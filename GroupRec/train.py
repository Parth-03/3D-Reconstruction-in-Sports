'''
 @FileName    : demo.py
 @EditTime    : 2023-09-28 16:53:59
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description :
'''
import torch
import torch.optim as optim
from cmd_parser import parse_config
from modules import init, ModelLoader, DatasetLoader
from process import train
import pickle
import sys
import os
import numpy as np

sys.argv = ['', '--config=cfg_files/train.yaml']


def main(**args):
    dtype = torch.float32
    out_dir, logger, smpl = init(dtype=dtype, **args)
    device = torch.device(index=args.get('gpu_index'), type='cuda')

    # =================== Create the dataloader ================== #
    print("Making dataset...")
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    train_dataset = dataset.load_train_data()
    train_dataset.human_detection()
    print("Dataset created.")

    # write dataset to file so you don't have to preprocess everything again.
    with open("data/Panda/Panda_imgs.pkl", 'wb') as file:
        pickle.dump(train_dataset, file)
    # ============================================================ #

    # uncomment this and comment above if above already ran
    # with open("data/Panda/Panda_imgs.pkl", "rb") as file:
    #     train_dataset = pickle.load(file)
    # print(len(train_dataset))

    # =================== Processing labels ====================== #
    with open("data/Panda/Panda.pkl", "rb") as file:
        data_list = pickle.load(file)
    train_x_path = "data/Panda/images/Det"
    # ground truth. train_y is stripped down version of original labels, since dataset is missing some.
    # missing img paths, but should now be in order of the dataset.
    train_y = []
    i = 0
    for sub_dir, _, files in os.walk(train_x_path):
        for filename in files:
            if i == 3:
                break
            corresponding_dict = None
            parent_folder = os.path.basename(sub_dir)
            image_identifier = os.path.join(parent_folder, filename)
            for data in data_list:
                if os.path.join(os.path.basename(os.path.dirname(data["img_path"])),
                                os.path.basename(data["img_path"])) == image_identifier:
                    corresponding_dict = data
                    break
            if corresponding_dict is None:
                print("problem")
                print(image_identifier)
                continue
            y = corresponding_dict['params']
            train_y.append(np.array(y))
            print(i)
            i += 1
    assert len(train_y) == len(train_dataset)

    with open("data/Panda/Panda_labels.pkl", 'wb') as file:
        pickle.dump(train_y, file)
    data_labels = train_y
    # ========================================================= #
    #
    # with open("data/Panda/Panda_labels.pkl", 'rb') as file:
    #     data_labels = pickle.load(file)

    # Load model
    print("Loading model...")
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)
    print("Model loaded.")
    optimizer = optim.Adam(model.model.parameters(), lr=0.0001)
    train(model, train_dataset, data_labels, optimizer, device=device)


if __name__ == "__main__":
    args = parse_config()
    main(**args)
