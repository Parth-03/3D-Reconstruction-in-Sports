'''
 @FileName    : demo.py
 @EditTime    : 2023-09-28 16:53:59
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description :
'''
import torch
from cmd_parser import parse_config
from modules import init, ModelLoader, DatasetLoader
from process import train
import pickle
import sys

sys.argv = ['', '--config=cfg_files/train.yaml']


def main(**args):
    dtype = torch.float32
    out_dir, logger, smpl = init(dtype=dtype, **args)

    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    train_dataset = dataset.load_train_data()
    print(train_dataset)
    # Make sure you run this from GroupRec folder
    with open("data/Panda/Panda.pkl", "rb") as file:
        data_labels = pickle.load(file)

    # # # Global setting

    # device = torch.device(index=args.get('gpu_index'), type='cuda')
    # #

    # #
    # # Load model
    # model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)

    # # create data loader

    #
    # train_dataset.human_detection()


#  train()

#eval('%s' % task)(model, train_dataset, device=device)


if __name__ == "__main__":
    args = parse_config()
    main(**args)
