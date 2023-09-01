"""Script for multi-gpu training."""
import json
import os
import sys
import requests
import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from cachetools import cached

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.bbox import bbox_xyxy_to_xywh
from active_learning.al_metric import compute_OKS

num_gpu = torch.cuda.device_count()
num_cpu = int(os.cpu_count()/4)

norm_layer = nn.SyncBatchNorm if opt.sync else nn.BatchNorm2d

def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('\nCreate new model')
        logger.info('=> init weights')
        model._initialize()
    return model

def validate(m, opt, cfg, data_loader, eval_joints, ann_file):
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    with torch.no_grad():
        with tqdm(data_loader, dynamic_ncols=True, leave=True) as data_loader:
            for i, (idxs, inps, labels, label_masks, GTkpts, img_ids, ann_ids, bboxes, bboxes_ann, _, _) in enumerate(data_loader):
                output = m(inps[:, 0].cuda()) # input inps into model
                assert output.dim() == 4
                pred = output[:, eval_joints, :, :]

                for j in range(output.shape[0]):
                    bbox = bboxes[j].tolist()
                    bbox_ann = bbox_xyxy_to_xywh(bboxes_ann[j].tolist())
                    pose_coords, pose_scores = heatmap_to_coord(
                        pred[j][eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                    keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
                    keypoints = keypoints.reshape(-1).tolist()
                    GT_keypoints = GTkpts[j].reshape(-1).tolist() # ground truth keypoints
                    oks = float(compute_OKS(bbox_ann, keypoints, GT_keypoints))
                    data_dict = {"bbox": bbox, "image_id": int(img_ids[j]), "id": int(ann_ids[j]), "score": float(np.mean(pose_scores) + 1.25 * np.max(pose_scores)), "category_id": 1, "keypoints": keypoints, "GT_keypoints": GT_keypoints, "OKS": oks}
                    kpt_json.append(data_dict)
    sysout = sys.stdout
    with open(os.path.join(opt.work_dir, 'predicted_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'predicted_kpt.json'), ann_type='keypoints', ann_file=ann_file)
    logger.info(res)
    sys.stdout = sysout
    return res["AP"]

def main():
    m = preset_model(cfg) # Model Initialize
    m = nn.DataParallel(m).cuda()

    for mode in ["TEST"]: # Evaluate
        if mode == "TRAIN":
            DATA_DIR = cfg.DATASET.TRAIN
            DATA_ANN = cfg.DATASET.TRAIN.ANN
        elif mode == "VAL":
            DATA_DIR = cfg.DATASET.VAL
            DATA_ANN = cfg.DATASET.VAL.ANN
        elif mode == "TEST":
            DATA_DIR = cfg.DATASET.TEST
            DATA_ANN = cfg.DATASET.TEST.ANN
        else:
            raise NotImplementedError
        dataset = builder.build_dataset(DATA_DIR, preset_cfg=cfg.DATA_PRESET, train=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.VAL.BATCH_SIZE * num_gpu, shuffle=False, num_workers=num_cpu, pin_memory=True, drop_last=False)
        ann_file=os.path.join(cfg.DATASET.VAL.ROOT, DATA_ANN)
        APs = validate(m, opt, cfg, dataloader, dataset.EVAL_JOINTS, ann_file) * 100
        logger.info(f'\n##### Evaluate {mode} | metric: {APs} #####')

if __name__ == "__main__":
    logger.info(opt)
    logger.info(cfg)
    logger.info('Number of GPUs: {}'.format(num_gpu))
    logger.info('Number of CPUs: {}'.format(num_cpu))
    logger.info('******************************\n')

    # CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main()