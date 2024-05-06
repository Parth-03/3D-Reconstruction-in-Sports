import argparse
import torch
import numpy as np

from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_loss(predictions, labels):
    

def train(dataloader, model, optimizer):
    model.train()
    loss_sum = 0.0
    loss_count = 0.0

    for epoch in range(num_epochs):
        for images, labels in dataloader:
            predictions = model(images)
            loss = compute_loss(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main(args):

    model = Model()
    model.to(device)
    print("=> Will use the (" + device.type + ") device.")

    optimizer = 

    # create dataset

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mutliHMR')

    parser.add_argument()

    print(parser.parse_args())
    main(parser.parse_args())