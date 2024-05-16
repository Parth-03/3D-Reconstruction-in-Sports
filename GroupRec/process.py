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
import pickle
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

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


def smpl_trans_loss(predictions, gts, field, gt_field, device) -> torch.Tensor:

    total_loss = 0
    all_preds = torch.empty((0, 3), dtype=torch.float32).to(device)
    for pred_img in predictions:
        preds_humans = pred_img[field]
        all_preds = torch.cat((all_preds, preds_humans))

    all_gts = []
    for humans in gts:
        for human in humans:
            all_gts.append(human[gt_field])

    criterion = torch.nn.L1Loss()

    all_gts = torch.stack([torch.tensor(item).to(device) for item in all_gts])

    # sort tuples themselves
    preds, _ = torch.sort(all_preds)
    curr_gts, _ = torch.sort(all_gts)

    # sort tuples based on first element
    preds, _ = torch.sort(preds, 0)
    curr_gts, _ = torch.sort(curr_gts, 0)
    total_loss += criterion(preds, curr_gts)
    return total_loss


def train(model, loader, labels, device=torch.device('cpu'), num_epochs=10, batch_size=20):
    torch.set_grad_enabled(True)

    # freeze parameters
    for name, param in model.model.named_parameters():
        if 'cam_head' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    num_batches = loader.len % batch_size

    # Shuffle data. Do this by shuffling indices, using indices to get data and labels, thus ensuring a match.
    data_indices = np.arange(len(labels))
    rng = np.random.default_rng()
    rng.shuffle(data_indices)

    epoch_losses = []
    batch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f"Epoch: {epoch+1}")
        for i in range(num_batches):
            if i != num_batches - 1:
                batch_labels = [labels[j] for j in data_indices[i * batch_size: (i + 1) * batch_size]]
                batch_imgs = [loader[j] for j in data_indices[i * batch_size: (i + 1) * batch_size]]
            else:  # add remainder
                batch_labels = [labels[j] for j in data_indices[i * batch_size:]]
                batch_imgs = [loader[j] for j in data_indices[i * batch_size:]]
            print(f"Processing batch {i + 1}/{num_batches} with {len(batch_imgs)} samples")
            model.optimizer.zero_grad()
            batch_output = process_batch(batch_imgs, model, device)
            # output is an array of preds. For pred format, see relation_joint

            # batch output may have less shape and translation to align
            batch_output, batch_labels = align_humans(batch_output, batch_labels)
           # print("aligned humans!")
            # 4.) import and use custom loss function
            # 4a.) use separate losses for shape and translation, then add losses
            batch_loss = smpl_trans_loss(batch_output, batch_labels, 'pred_cam_t', 'trans', device)

            if torch.isnan(batch_loss):
                print(f"NaN detected in loss at batch {i + 1}")
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=0.01)
            model.optimizer.step()


            # Record loss
            batch_losses.append(batch_loss.item())
            epoch_loss += batch_loss.item()
            print(f"Loss: {batch_loss.item()}")

            # Clean up
            del batch_labels, batch_imgs, batch_output, batch_loss
            torch.cuda.empty_cache()

            # for name, param in model.model.named_parameters():
            #     if param.requires_grad:
            #        print(f"{name} param norm: {param.norm()}")

        average_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1} Average Loss: {average_epoch_loss}")
        model.save_model(epoch, 'train')  # save every epoch

    # automatically plot epoch
    # plt.plot(np.arange(num_epochs), epoch_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Loss')
    # plt.title('SMPL Trans Loss Over Time')
    # plt.show()
    with open("data/Panda/epoch_losses.pkl", 'wb') as file:
        pickle.dump(epoch_losses, file)

    with open("data/Panda/batch_losses.pkl", 'wb') as file:
        pickle.dump(batch_losses, file)

    model.save_model(num_epochs, 'final_train')  # save every epoch
