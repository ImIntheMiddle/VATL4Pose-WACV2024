"""PoseTrack21 Dataset"""
import os
import numpy as np
import torch
import cv2
import copy
from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from .custom import CustomDataset
import pdb

@DATASET.register_module
class Posetrack21(CustomDataset): # alphapose/models/builder.py
    """ PoseTrack21 Dataset
    Parameters
    ----------
    root: str, default './data/PoseTrack21'
    for home-local: '/home-local/halo/PoseTrack21'
        Path to the PoseTrack21 dataset.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    num_joints = 17
    joint_pairs = [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    def __getitem__(self, idx):
        """Get item for the given index.
        process depends on whether get_prenext is True or False.
        Args:
            idx (int): Index of the item to be retrieved from the dataset.

        Returns: Result of the get item function corresponding to self.temporal.
        """
        if self.get_prenext:
            return self._get_temporal_img(idx)
        else:
            return self._get_single_img(idx)

    def __len__(self):
        return len(self._items)

    def _load_jsons(self):
        """Load all image paths and labels from annotation files into buffer."""
        _posetrack = self._lazy_load_ann_file()
        classes = [c['name'] for c in _posetrack.loadCats(_posetrack.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with PoseTrack21. "

        items, labels = [], []
        image_ids = sorted(_posetrack.getImgIds()) # return list of ids for each image. sorted by id
        num_frames = len(image_ids)

        for frame in _posetrack.loadImgs(image_ids): # iterate for each image id from start to end of sequence
            filename = frame['file_name'] # load image path
            abs_path = os.path.join(self._root, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            valid_objs = self._check_load_keypoints(_posetrack, frame) # load annotation of this frame
            if not valid_objs:
                continue
            for person_label in valid_objs: # num of items are relative to person, not frame
                item = {}
                item['path'] = abs_path
                item['img_id'] = frame['image_id']
                item['ann_id'] = person_label['ann_id']
                item['id'] = person_label['id']
                item['track_id'] = person_label['track_id']
                item['keypoint'] = person_label['keypoint']
                items.append(item)
                labels.append(person_label)

        # sort items and labels by 'ann_id'. Required for TPC & THC
        items_sorted = sorted(items, key=lambda x: x['id'])
        labels_sorted = sorted(labels, key=lambda x: x['id'])
        return items_sorted, labels_sorted # 結局このitemsとlabelsがちゃんと人ごとに並んでいればよい

    def _check_load_keypoints(self, _data, frame):
        """Check and load ground-truth keypoints"""
        ann_ids = _data.getAnnIds(imgIds=frame['image_id'])
        objs = _data.loadAnns(ann_ids)
        valid_objs = []
        width = int(frame['width'])
        height = int(frame['height'])

        for obj in objs:
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            if xmax <= xmin or ymax <= ymin: # require non-zero box area
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            if max(obj['keypoints']) == 0: # invalid keypoint annotations
                continue
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0
            if np.sum(joints_3d[:, 0, 1]) < 1: # no visible keypoint
                continue

            ann_id = int(obj['id']) # ann_id is unique for each human in each frame
            id = int(str(ann_id)[-2:] + str(frame['image_id'])) # idの下二桁を取り出し，img_idと結合したものをann_idとする
            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax), # (xmin, ymin, xmax, ymax)
                'width': width,
                'height': height,
                'joints_3d': joints_3d,
                'keypoint': obj['keypoints'],
                'id': id,
                'ann_id': ann_id,
                'track_id': obj['track_id']
            })
        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32),
                    'keypoint': np.zeros((self.num_joints * 3), dtype=np.float32),
                    'id': -1,
                    'ann_id': -1,
                    'track_id': -1
                    })
        return valid_objs

    def _get_temporal_img(self, idx):
        assert type(self._items[idx]) == dict
        img_path = self._items[idx]['path']
        img_id = self._items[idx]['img_id'] # get image id
        track_id = self._items[idx]['track_id'] # get annotation id, unique for each body
        ann_id = self._items[idx]['ann_id'] # get annotation id, unique for each body
        GTkpt = self._items[idx]['keypoint'] # get ground truth of current frame

        # load current, previous and next image
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # load ground truth of current frame, bbox of previous and next frame
        label = copy.deepcopy(self._labels[idx])
        bbox_ann = torch.Tensor(copy.deepcopy(self._labels[idx]['bbox']))
        # transform img and gt annotation into input_img and training label
        img, label, label_mask, bbox_crop = self.transformation(img, label) # no augmentation

        # load previous and next image, then transform them
        if idx == 0: # if first frame
            # make dummy tensor
            img_pre = torch.zeros_like(img)
            isPrev = False
        else:
            pre_img_path = self._items[idx-1]['path']
            pre_track_id = self._items[idx-1]['track_id']
            if pre_track_id != track_id: # if different person is in previous frame
                img_pre = torch.zeros_like(img)
                isPrev = False
            else:
                img_pre = cv2.cvtColor(cv2.imread(pre_img_path), cv2.COLOR_BGR2RGB)
                bbox_pre = copy.deepcopy(self._labels[idx - 1]['bbox'])
                # torch.from_numpy(img_pre).to(self._device) # move to device
                img_pre, _ = self.transformation.test_transform(img_pre, bbox_pre)
                isPrev = True
        if idx == len(self._items)-1: # if last frame
            img_next = torch.zeros_like(img)
            isNext = False
        else:
            next_img_path = self._items[idx+1]['path']
            next_track_id = self._items[idx+1]['track_id']
            if next_track_id != track_id: # if different persin is in next frame
                img_next = torch.zeros_like(img)
                isNext = False
            else:
                img_next = cv2.cvtColor(cv2.imread(next_img_path), cv2.COLOR_BGR2RGB)
                bbox_next = copy.deepcopy(self._labels[idx + 1]['bbox'])
                # torch.from_numpy(img_next).to(self._device) # move to device
                img_next, _ = self.transformation.test_transform(img_next, bbox_next)
                isNext = True
        stacked_inp = torch.stack([img, img_pre, img_next], dim=0)
        GTkpt = torch.tensor(GTkpt)
        return idx, stacked_inp, label, label_mask, GTkpt, img_id, ann_id, bbox_crop, bbox_ann, isPrev, isNext # for temporal model

    def _get_single_img(self, idx):
        assert type(self._items[idx]) == dict
        img_path = self._items[idx]['path']
        img_id = self._items[idx]['img_id']
        ann_id = self._items[idx]['ann_id']
        GTkpt = self._items[idx]['keypoint']
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        label = copy.deepcopy(self._labels[idx]) # load ground truth
        bbox_ann = torch.Tensor(copy.deepcopy(self._labels[idx]['bbox']))

        # transform img and gt annotation into input_img and training label
        # torch.from_numpy(img).to(self._device) # move to device
        input, label, label_mask, bbox_crop = self.transformation(img, label) # no augmentation. bbox format: (xmin, ymin, xmax, ymax)
        stacked_inp = torch.stack([input, torch.zeros_like(input), torch.zeros_like(input)], dim=0) # only use current image
        GTkpt = torch.tensor(GTkpt)
        return idx, stacked_inp, label, label_mask, GTkpt, img_id, ann_id, bbox_crop, bbox_ann, False, False # isPrev and isNext are always False

    def my_collate_fn(self, batch):
        # batch is a list of tuple
        # each tuple is (idx, stacked_inp, label, label_mask, GTkpt, img_id, ann_id, bbox, isPrev, isNext)
        # stacked_inp: (3, 3, 256, 256)
        # label: (1, 17, 2)
        # label_mask: (1, 17, 1)
        # GTkpt: (51)
        # bbox: (4)
        idx, stacked_inp, label, label_mask, GTkpt, img_id, ann_id, bbox_crop, bbox_ann, isPrev, isNext = zip(*batch)
        stacked_inp = torch.stack(stacked_inp, dim=0)
        label = torch.stack(label, dim=0)
        label_mask = torch.stack(label_mask, dim=0)
        GTkpt = torch.stack(GTkpt, dim=0)
        bbox_crop = torch.stack(bbox_crop, dim=0)
        bbox_ann = torch.stack(bbox_ann, dim=0)
        isPrev = torch.tensor(isPrev, dtype=torch.bool)
        isNext = torch.tensor(isNext, dtype=torch.bool)
        return idx, stacked_inp, label, label_mask, GTkpt, img_id, ann_id, bbox_crop, bbox_ann, isPrev, isNext