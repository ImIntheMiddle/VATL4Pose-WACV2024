"""Script for multi-gpu training."""
import json
import os
import sys
import requests
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm
from cachetools import cached

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord

num_gpu = torch.cuda.device_count()
num_cpu = os.cpu_count()

if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d


def train(opt, train_loader, m, criterion, optimizer, writer):
    loss_logger = DataLogger()
    acc_logger = DataLogger()

    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')

    m.train()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, label_masks, img_ids, ann_ids, bboxes) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
        else:
            inps = inps.cuda().requires_grad_()
        if isinstance(labels, list):
            labels = [label.cuda() for label in labels]
            label_masks = [label_mask.cuda() for label_mask in label_masks]
        else:
            labels = labels.cuda()
            label_masks = label_masks.cuda()

        output = m(inps) # input inps into model

        loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))
        acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        if opt.board:
            board_writing(writer, loss_logger.avg, acc_logger.avg, opt.trainIters, 'Train')

        # Debug
        if opt.debug and not i % 10:
            debug_writing(writer, output, labels, inps, opt.trainIters)
            if i % 10==0:
                break

        # TQDM
        train_loader.set_description(
            'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                loss=loss_logger.avg,
                acc=acc_logger.avg)
        )

    train_loader.close()

    return loss_logger.avg, acc_logger.avg

def validate_gt(m, opt, cfg, heatmap_to_coord):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=cfg.VAL.BATCH_SIZE*num_gpu, shuffle=False, num_workers=num_cpu, drop_last=False, pin_memory=True)
    kpt_json = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)
    for i, (inps, labels, label_masks, img_ids, ann_ids, bboxes) in enumerate(gt_val_loader):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()

        output = m(inps) # input inps into model

        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for j in range(output.shape[0]):
            bbox = bboxes[j].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[j][gt_val_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[j].tolist()
            data['image_id'] = int(img_ids[j])
            data['ann_id'] = int(ann_ids[j])
            data['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    sysout = sys.stdout
    with open(os.path.join(opt.work_dir, 'predicted_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'predicted_kpt.json'), ann_type='keypoints', ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    sys.stdout = sysout
    return res

def main():
    logger.info('\n******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')
    logger.info('Number of GPUs: {}'.format(num_gpu))
    logger.info('Number of CPUs: {}'.format(num_cpu))
    logger.info('******************************\n')

    # CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Model Initialize
    m = preset_model(cfg)
    m = nn.DataParallel(m).cuda()

    criterion = builder.build_loss(cfg.LOSS).cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    writer = SummaryWriter('.tensorboard/{}-{}'.format(opt.exp_id, cfg.FILE_NAME))

    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=num_cpu, pin_memory=True)

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    opt.trainIters = 0
    best_score = 0
    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'\n############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss, miou = train(opt, train_loader, m, criterion, optimizer, writer)
        logger.epochInfo('Train', opt.epoch, loss, miou)
        line_notify(f'\n【{cfg.MODEL.TYPE}: 学習が進んだぞ!】\nEpoch: {opt.epoch}\nTrain Loss: {loss:.8f}\nTrain mIoU: {miou:.4f}\nです!')

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint
            torch.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
            # Prediction Test
            with torch.no_grad():
                gt_AP = validate_gt(m.module, opt, cfg, heatmap_to_coord)
                logger.info(f'\n##### Epoch {opt.epoch} | gt mAP: {gt_AP} #####')
                line_notify(f'\n【{cfg.MODEL.TYPE}: 途中経過のお知らせ!!】\nEpoch: {opt.epoch}\nVal mAP: {gt_AP}\nです!')
                if gt_AP > best_score:
                    best_score = gt_AP
                    torch.save(m.module.state_dict(), './exp/{}-{}/model_best.pth'.format(opt.exp_id, cfg.FILE_NAME))
                    line_notify(f'\n【{cfg.MODEL.TYPE}: ベストスコア更新!】\n"./exp/{opt.exp_id}-{cfg.FILE_NAME}/model_best.pth" に保存したよ!',
                                sticker_package_id=446, sticker_id=1991)

        # Time to add DPG
        if i == cfg.TRAIN.DPG_MILESTONE:
            torch.save(m.module.state_dict(), './exp/{}-{}/final.pth'.format(opt.exp_id, cfg.FILE_NAME))
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.TRAIN.LR
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
            # Reset dataset
            train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, dpg=True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=num_cpu)

    torch.save(m.module.state_dict(), './exp/{}-{}/final_DPG.pth'.format(opt.exp_id, cfg.FILE_NAME))


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

def line_notify(notification_message, image_path=None, sticker_id=None, sticker_package_id=None):
    line_notify_token = "" # LINE Notifyのトークン
    line_notify_api = 'https://notify-api.line.me/api/notify'
    print(f'message: {notification_message}')
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    if image_path is not None:
        files = {"imageFile": open(image_path, "rb")}
    if (sticker_id is not None) and (sticker_package_id is not None):
        data['stickerId'] = sticker_id
        data['stickerPackageId'] = sticker_package_id
    requests.post(line_notify_api, headers = headers, data = data)

if __name__ == "__main__":
    main()