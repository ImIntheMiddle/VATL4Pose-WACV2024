import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import torchvision
import cv2
import pdb
import glob

from alphapose.utils.vis import vis_frame
from alphapose.utils.config import update_config
from alphapose.utils.transforms import get_func_heatmap_to_coord

"""visualize results (pose estimation, heatmap, etc.)"""
def read_video_list(video_list):
    """read video list from txt file.
    Args:
        video_list(list): path to video list txt file.
    """
    with open(video_list, "r") as f:
      video_list = f.readlines()
    video_list = [video.strip() for video in video_list]
    return video_list

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def save_batch_image_with_joints(batch_image, batch_heatmaps, batch_joints, batch_joints_vis, joint_pairs, file_name, nrow=8, padding=2, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())
        batch_image.add_(-min).div_(max - min + 1e-5)

    num_joints = batch_heatmaps.size(1)
    preds, _ = get_max_preds(batch_heatmaps.detach().cpu().numpy())
    preds *= 4.0 # to the original scale
    img = batch_image[0].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img = img.copy()
    # tone down the img
    img = img.astype(np.float32)
    img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    heatmaps = batch_heatmaps[0].mul(255).clamp(0, 255).byte().cpu().numpy()
    joints = batch_joints[0].cpu().numpy()
    joints_vis = batch_joints_vis[0].cpu().numpy()

    for joint_pair in joint_pairs:
        if joints_vis[joint_pair[0]] and joints_vis[joint_pair[1]]:
            cv2.line(img=img, pt1=(int(preds[0][joint_pair[0]][0]), int(preds[0][joint_pair[0]][1])), pt2=(int(preds[0][joint_pair[1]][0]), int(preds[0][joint_pair[1]][1])), color=[0, 0, 255], thickness=2)

    for j in range(num_joints):
        if joints_vis[j]:
            # print(int(preds[0][j][0]), int(preds[0][j][1])) # x, y
            cv2.circle(img=img, center=(int(preds[0][j][0]), int(preds[0][j][1])), radius=2, color=[0, 0, 255], thickness=3)

    cv2.imwrite(file_name, img)
    print("save image to {}".format(file_name))


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((3*heatmap_height,5*heatmap_width,3),dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heatmaps = batch_heatmaps[i].mul(255).clamp(0, 255).byte().cpu().numpy()

        resized_image = cv2.resize(image,(int(heatmap_width), int(heatmap_height)))
        cnt = 0
        for j in range(num_joints):
            if j in [3, 4]:
                continue
            height_begin = heatmap_height * (cnt//5)
            height_end = heatmap_height * (cnt//5 + 1)
            cv2.circle(resized_image,(int(preds[i][j][0]), int(preds[i][j][1])),1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,(int(preds[i][j][0]), int(preds[i][j][1])),1, [0, 0, 255], 1)

            width_begin = heatmap_width * (cnt%5)
            width_end = heatmap_width * (cnt%5 + 1)
            # print(height_begin, height_end, width_begin, width_end)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3
            # print(f"{j} clear")
            cnt += 1
        # grid_image[0:heatmap_height, 0:heatmap_width, :] = resized_image
    cv2.imwrite(file_name, grid_image)
    print("save image to {}".format(file_name))


def vis_item(cfg, item_path):
    #parse path
    ann_id = item_path.split('/')[-1].split('.')[0]
    # track_id: ann_idの下二桁
    track_id = ann_id[-2:]
    image_id = ann_id[:-2]
    round_cnt = item_path.split('/')[-2]
    video_id = item_path.split('/')[-5]
    strategy = item_path.split('/')[-6]
    model = item_path.split('/')[-7]
    rootdir = os.path.join('exp/vis', model, strategy, video_id, round_cnt, track_id)
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)

    # load npy file, which is a dictionary
    # unpack npy file and save heatmap
    item_dict = np.load(item_path, allow_pickle=True)
    # print(item_dict.item().keys())
    # print(item_dict.item()['img'].shape)
    # pdb.set_trace()
    heatmaps = item_dict.item()['heatmaps'] # [17, 64, 48]
    batch_image = item_dict.item()['img'] # list of images [channel, height, width]
    batch_joints = item_dict.item()['keypoints'] # list of joints (length = 51 = 17*3)
    batch_joints_vis = [1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1] # 17 joints
    joint_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 7], [6, 8], [7, 9], [8, 10], [0, 1], [0, 2], [1, 5], [1, 6]]
    # pdb.set_trace()

    batch_image =torch.Tensor(batch_image).view(1, 3, 256, 192)
    heatmaps = torch.Tensor(heatmaps).view(1, 17, 64, 48)
    batch_joints = torch.Tensor(batch_joints).view(1, 17, 3)
    batch_joints_vis = torch.Tensor(batch_joints_vis).view(1, 17, 1)

    # save images with joints
    os.makedirs(os.path.join(rootdir, "estimation_result"), exist_ok=True)
    save_batch_image_with_joints(batch_image, heatmaps, batch_joints, batch_joints_vis, joint_pairs, os.path.join(rootdir, "estimation_result", f'{image_id}.jpg'))
    # save heatmaps
    os.makedirs(os.path.join(rootdir, "heatmap"), exist_ok=True)
    save_batch_heatmaps(batch_image, heatmaps, os.path.join(rootdir, "heatmap", f'{image_id}.jpg'))

if __name__ == '__main__':
    cfg = update_config("configs/al_simple.yaml") # load config
    # strategy_list = ["Random", "HP", "TPC", "THC_L1", "WPU_hybrid"] # strategies
    strategy_list = ["Random"]
    round_list = ["Round0", "Round4"]
    root_dir = "exp/AL_Glory/SimplePose"
    video_id_list = read_video_list("configs/val_video_list.txt")
    video_id_list = ["000812"]
    for strategy in strategy_list:
        for video_id in video_id_list: # load each video's result from 'result.json' in each directory
            result_dir = os.path.join(root_dir, strategy, video_id) # exp/AL_Glory/SimplePose/HP/000000
            for round in round_list:
                results = glob.glob(f"{result_dir}/*/heatmap/{round}/*.npy")
                for item_path in results:
                    vis_item(cfg, item_path)