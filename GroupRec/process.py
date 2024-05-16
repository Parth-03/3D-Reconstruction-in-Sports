'''
 @FileName    : process.py
 @EditTime    : 2022-09-27 16:18:51
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import torch
import numpy as np
import cv2
from tqdm import tqdm
import time


def extract_valid(data):
    batch_size, agent_num, d = data['keypoints'].shape[:3]
    valid = data['valid'].reshape(-1, )

    data['center'] = data['center']  #.reshape(batch_size*agent_num, -1)[valid == 1]
    data['scale'] = data['scale']  #.reshape(batch_size*agent_num,)[valid == 1]
    data['img_h'] = data['img_h']  #.reshape(batch_size*agent_num,)[valid == 1]
    data['img_w'] = data['img_w']  #.reshape(batch_size*agent_num,)[valid == 1]
    data['focal_length'] = data['focal_length']  #.reshape(batch_size*agent_num,)[valid == 1]

    data['valid_img_h'] = data['img_h'].reshape(batch_size * agent_num, )[valid == 1]
    data['valid_img_w'] = data['img_w'].reshape(batch_size * agent_num, )[valid == 1]
    data['valid_focal_length'] = data['focal_length'].reshape(batch_size * agent_num, )[valid == 1]
    data['has_3d'] = data['has_3d'].reshape(batch_size * agent_num, 1)[valid == 1]
    data['has_smpl'] = data['has_smpl'].reshape(batch_size * agent_num, 1)[valid == 1]
    data['verts'] = data['verts'].reshape(batch_size * agent_num, 6890, 3)[valid == 1]
    data['gt_joints'] = data['gt_joints'].reshape(batch_size * agent_num, -1, 4)[valid == 1]
    data['pose'] = data['pose'].reshape(batch_size * agent_num, 72)[valid == 1]
    data['betas'] = data['betas'].reshape(batch_size * agent_num, 10)[valid == 1]
    data['keypoints'] = data['keypoints'].reshape(batch_size * agent_num, 26, 3)[valid == 1]
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size * agent_num, 3)[valid == 1]

    imgname = (np.array(data['imgname']).T).reshape(batch_size * agent_num, )[valid.detach().cpu().numpy() == 1]
    data['imgname'] = imgname.tolist()

    return data


def extract_valid_demo(data):
    batch_size, agent_num, _, _, _ = data['img'].shape
    valid = data['valid'].reshape(-1, )

    data['center'] = data['center']
    data['scale'] = data['scale']
    data['img_h'] = data['img_h']
    data['img_w'] = data['img_w']
    data['focal_length'] = data['focal_length']

    data['valid_focal_length'] = data['focal_length'].reshape(batch_size * agent_num, )[valid == 1]

    return data


def to_device(data, device):
    imnames = {'imgname': data['imgname']}
    data = {k: v.to(device).float() for k, v in data.items() if k not in ['imgname']}
    data = {**imnames, **data}

    return data


def relation_demo(model, loader, device=torch.device('cpu')):
    print('-' * 10 + 'model demo' + '-' * 10)
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            batchsize = data['img'].shape[0]
            data = to_device(data, device)
            data = extract_valid_demo(data)

            # forward
            pred = model.model(data)

            results = {}
            results.update(imgs=data['imgname'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['valid_focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_demo_joint_results(results, i, batchsize)
            else:
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                model.save_demo_results(results, i, batchsize)


"""
Custom training function. 
"""


def process_batch(batch, model, device):
    batch_output = []
    for image in batch:
        input = torch.from_numpy(image).to(device)
        pred = model(input)
        batch_output.append(pred)
    return batch_output


def train(model, loader, labels, optimizer, device=torch.device('cpu'), num_epochs=10, batch_size=20):
    num_batches = loader.len // batch_size - 1
    print(num_batches)

    # Shuffle data. Do this by shuffling indices, using indices to get data and labels, thus ensuring a match.
    data_indices = np.arange(len(labels))
    # 3.) set up training framework (ensure use align humans)
    epoch_losses = []
    batch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(num_batches):
            if i != num_batches - 1:
                batch_labels = [labels[j] for j in data_indices[i * batch_size: (i + 1) * batch_size]]
                batch_imgs = [loader[j] for j in data_indices[i * batch_size: (i + 1) * batch_size]]
            else:  # add remainder
                batch_labels = [labels[j] for j in data_indices[i * batch_size:]]
                batch_imgs = [loader[j] for j in data_indices[i * batch_size:]]
            print(f"Processing batch {i + 1}/{num_batches} with {len(batch_imgs)} samples")
            optimizer.zero_grad()
            batch_output = process_batch(batch_imgs, model, device)

            batch_output, batch_labels = align_humans(batch_output, batch_labels)
            # 4.) import and use custom loss function
            # 4a.) use separate losses for shape and translation, then add losses
            batch_loss = smpl_params_loss(batch_output, batch_labels, ['transl'], ['trans'])

            if torch.isnan(batch_loss):
                print(f"NaN detected in loss at batch {i + 1}")

            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()

            # Record loss
            batch_losses.append(batch_loss.item())
            epoch_loss += batch_loss.item()
            print(f"Epoch {epoch + 1}, Batch {(i + 1)}, Loss: {batch_loss.item()}")

            # Clean up
            del batch_labels, batch_imgs, batch_output, batch_loss
            torch.cuda.empty_cache()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name} param norm: {param.norm()}")

        average_epoch_loss = epoch_loss / (num_batches)
        epoch_losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1} Average Loss: {average_epoch_loss}")

