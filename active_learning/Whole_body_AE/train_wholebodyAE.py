"""Main script for training the whole body autoencoder.
Define the model, load the data, train the model, and save the model.
Using the whole body autoencoder, we compute the whole body hand-crrafted feature from the input pose and reconstruct it."""

import argparse
import os
# CUDA setting
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import pdb # import python debugger
import sys
import time
import random
import json
import datetime

# python general libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from cachetools import cached

# import original libraries
from AutoEncoder import WholeBodyAE
from Whole_body_hybrid import Wholebody
from hybrid_feature import compute_hybrid

def parse_args():
    """
    parse given arguments before active learning execution
    return: args parsed by parser
    """
    parser = argparse.ArgumentParser(description='Active Learning Script')
    parser.add_argument('--z', type=int, default="2", help='dimension of latent space')
    parser.add_argument('--epoch', type=int, default="300", help='number of epochs')
    parser.add_argument('--pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--kp_direct', action='store_true', help='whether to use keypoints as input of AE directly')
    args = parser.parse_args()
    return args

def plot(log_train, log_valid, save_root):
    # plot train and valid loss
    plt.figure()
    x_train = np.arange(0, len(log_train))
    x_valid = np.arange(0, len(log_valid)) * 2
    plt.plot(x_train, log_train, label="Train_loss")
    plt.plot(x_valid, log_valid, label="Valid_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # save plot
    plt.savefig(f'{save_root}/plot.png')
    plt.close()
    print("plot saved!")

if __name__ == '__main__':
    # setting
    opt = parse_args()
    NUM_EPOCHS = opt.epoch
    Z_DIM = opt.z
    SEED = 318
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # get current time
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # save log as json file
    type = "direct" if opt.kp_direct else "hybrid"
    # make directory if not exist
    save_root = f"exp/Whole_body_AE/{type}/zdim_{Z_DIM}/{now}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WholeBodyAE(z_dim=Z_DIM, kp_direct=opt.kp_direct).to(device)
    if opt.pretrained:
        pretrained_path = f"pretrained_models/Whole_body_AE/zdim_{Z_DIM}.pth"
        # load pretrained model
        model.load_state_dict(torch.load(pretrained_path))
    print("model: ", model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss() # MSE loss

    # define dataset and dataloader
    print("Loading dataset...")
    train_dataset = Wholebody(mode="train", kp_direct=opt.kp_direct)
    train_loader = DataLoader(train_dataset, batch_size=80000, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    print("train dataset: ", len(train_dataset))

    valid_dataset= Wholebody(mode="train_val", kp_direct=opt.kp_direct)
    valid_loader = DataLoader(valid_dataset, batch_size=8000, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    print(f"valid dataset: {len(valid_dataset)}\n")

    log = {"z_dim": Z_DIM, "epoch": NUM_EPOCHS, "pretrained": opt.pretrained, "kp_direct": opt.kp_direct, "Train_loss": [], "Valid_loss": []}
    best_loss = 0
    for epoch in range(opt.epoch):
        # learning rate decay
        if epoch == 100:
          optimizer.param_groups[0]['lr'] = 0.005
        if epoch == 200:
          optimizer.param_groups[0]['lr'] = 0.001

        # training
        model.train()
        train_loss = 0
        for i, feature in tqdm(enumerate(train_loader)):
            input = feature.to(device) # input: (batch_size, 42) vector
            output = model(input) # output: (batch_size, 42) vector. output is reconstructed feature
            loss = criterion(output, input) # loss: reconstruction error
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1}, train_loss: {train_loss/len(train_loader): 0.4f}')
        log["Train_loss"].append(train_loss/len(train_loader))

        # validation
        if epoch % 2 == 0: # validation every 5 epochs
            valid_loss = 0
            model.eval()
            with torch.no_grad():
                for i, feature in tqdm(enumerate(valid_loader)):
                    input = feature.to(device)
                    output = model(input)
                    loss = criterion(output, input) # loss: reconstruction error
                    valid_loss += loss.item()
            log["Valid_loss"].append(valid_loss/len(valid_loader))
            # save model if valid loss is the best
            if valid_loss < best_loss or epoch == 0:
                best_loss = valid_loss
                torch.save(model.state_dict(), f"{save_root}/WholeBodyAE_zdim{Z_DIM}.pth")
                print("\nbest model updated!")
                log['best_epoch'] = epoch
                log['best_loss'] = best_loss/len(valid_loader)
            print(f'Epoch: {epoch+1}, val_loss: {valid_loss/len(valid_loader): 0.4f}\n')
            if epoch % 100 == 0:
                plot(log["Train_loss"], log["Valid_loss"], save_root)
    with open(f'{save_root}/log.json', 'w') as f:
        json.dump(log, f)
    print("\nlog saved!")
    plot(log["Train_loss"], log["Valid_loss"], save_root)
    print("\ntraining finished!")