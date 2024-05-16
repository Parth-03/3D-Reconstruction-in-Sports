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
from copy import copy


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
    for data in batch:
        data = to_device(data, device)
        data = extract_valid_demo(data)  # not really sure what this does or how it differs from extract _demo
        pred = model.model(data)
        batch_output.append(pred)
    return batch_output


# remove extra humans based on detection score
def align_humans(predictions, gts):
    predictions = copy(predictions)
    gts = copy(gts)
    aligned_preds = []
    aligned_gts = []
    for pred, gt in zip(predictions, gts):
        while (pred_length := pred['pred_cam_t'].shape[0]) > len(gt):
            # detection score doesn't exist, so have to do some random guess as to which is incorrect
            index = np.random.randint(pred_length)
            pred['pred_cam_t'] = torch.cat((pred['pred_cam_t'][:index], pred['pred_cam_t'][index + 1:]))
        while pred_length < len(gt):
            gt = gt[:len(gt) - 1]

        assert pred_length == len(gt)
        aligned_preds.append(pred)
        aligned_gts.append(gt)

    assert len(aligned_preds) == len(aligned_gts)
    return aligned_preds, aligned_gts


def smpl_params_loss(predictions, gts, fields, gt_fields, device) -> torch.Tensor:
    # second dimension is 3 because translations only have 3. Yeah this is some bad hardcoding...
    all_preds = []
    all_gts = []

    total_loss = 0

    for field in fields:
        curr_preds = torch.empty((0, 3), dtype=torch.float32).to(device)
        for pred_img in predictions:
            preds_humans = pred_img[field]
            curr_preds = torch.cat((curr_preds, preds_humans))
        all_preds.append(curr_preds)
    for field in gt_fields:
        curr_gts = []
        for humans in gts:
            curr_values = []
            for human in humans:
                curr_values.append(human[field])
            curr_gts.append(curr_values)
        all_gts.append(curr_gts)

    criterion = torch.nn.L1Loss()
    #criterion = torch.nn.MSELoss()
    for i, field in enumerate(fields):
        # convert to tensors
        all_gts[i] = torch.from_numpy(np.stack([item for sublist in all_gts[i] for item in sublist]))
        all_preds[i] = all_preds[i].to(device)
        all_gts[i] = all_gts[i].to(device)
        total_loss += criterion(all_preds[i], all_gts[i])
    return total_loss


def train(model, loader, labels, optimizer, device=torch.device('cpu'), num_epochs=10, batch_size=20):
    torch.set_grad_enabled(True)
    # freeze parameters
    for name, param in model.model.named_parameters():
        if 'x_attention_head' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    num_batches = loader.len // batch_size - 1
    print(num_batches)

    # Shuffle data. Do this by shuffling indices, using indices to get data and labels, thus ensuring a match.
    data_indices = np.arange(len(labels))
    rng = np.random.default_rng()
    rng.shuffle(data_indices)

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
            # output is an array of preds. For pred format, see relation_joint

            # batch output may have less shape and translation to align
            batch_output, batch_labels = align_humans(batch_output, batch_labels)
            print("aligned humans!")
            # 4.) import and use custom loss function
            # 4a.) use separate losses for shape and translation, then add losses
            batch_loss = smpl_params_loss(batch_output, batch_labels, ['pred_cam_t'], ['trans'], device)

            if torch.isnan(batch_loss):
                print(f"NaN detected in loss at batch {i + 1}")
            batch_loss.requires_grad = True
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=0.01)
            optimizer.step()

            # Record loss
            batch_losses.append(batch_loss.item())
            epoch_loss += batch_loss.item()
            print(f"Epoch {epoch + 1}, Batch {(i + 1)}, Loss: {batch_loss.item()}")

            # Clean up
            del batch_labels, batch_imgs, batch_output, batch_loss
            torch.cuda.empty_cache()

            # for name, param in model.model.named_parameters():
            #     if param.requires_grad:
            #        print(f"{name} param norm: {param.norm()}")

        average_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1} Average Loss: {average_epoch_loss}")
    model.save_model()
