"""PoseTrack21 Dataset"""
import os
import numpy as np
import cv2
from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from .custom import CustomDataset

@DATASET.register_module
class Posetrack21(CustomDataset): # alphapose/models/builder.py
    """ PoseTrack21 Dataset
    Parameters
    ----------
    root: str, default './data/PoseTrack21'
        Path to the PoseTrack21 dataset.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    num_joints = 17
    joint_pairs = [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    def __init__(self, get_prenext=False):
        super(Posetrack21, self).__init__(get_prenext)
        self.get_prenext = get_prenext
        self._get_item_func = self._get_temporal_img if self.get_prenext else self._get_single_img

    def __getitem__(self, idx):
        """Get item for the given index.
        process depends on whether temporal is True or False.
        Args:
            idx (int): Index of the item to be retrieved from the dataset.

        Returns: Result of the get item function corresponding to self.temporal.
        """
        return self._get_item_func(idx)

    def __len__(self):
        return len(self._items)

    def _load_jsons(self):
        """Load all image paths and labels from annotation files into buffer."""
        _posetrack = self._lazy_load_ann_file()

        # calss invalid check
        classes = [c['name'] for c in _posetrack.loadCats(_posetrack.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with PoseTrack21. "

        items = []
        labels = []
        image_ids = sorted(_posetrack.getImgIds()) # return list of ids for each image. sorted by id
        num_frames = len(image_ids)

        for frame in _posetrack.loadImgs(image_ids): # iterate for each image id from start to end of sequence

            filename = frame['file_name'] # load image path
            abs_path = os.path.join(self._root, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))

            frame_label = self._check_load_keypoints(_posetrack, frame) # load annotation of this frame
            if not frame_label:
                continue

            # num of items are relative to person, not frame
            for person_ann in frame_label:
                items.append(abs_path)
                labels.append(person_ann)

        return items, labels

    def _check_load_keypoints(self, _data, frame):
        """Check and load ground-truth keypoints"""
        ann_ids = _data.getAnnIds(imgIds=frame['id'])
        objs = _data.loadAnns(ann_ids)

        # check valid bboxes
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

            person = int(obj['person_id'])

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d
                'person': person
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                    'person': -1
                })
        return valid_objs

    def _get_temporal_img(self, idx):
        if type(self._items[idx]) == dict:
            img_path = self._items[idx]['path']
            pre_img_path = self._items[idx-1]['path']
            next_img_path = self._items[idx+1]['path']
            img_id = self._items[idx]['id'] # get image id
        else:
            img_path = self._items[idx]
            pre_img_path = self._items[idx - 1]
            next_img_path = self._items[idx + 1]
            img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # load current, previous and next image
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_pre = cv2.cvtColor(cv2.imread(pre_img_path), cv2.COLOR_BGR2RGB)
        img_next = cv2.cvtColor(cv2.imread(next_img_path), cv2.COLOR_BGR2RGB)

        # load ground truth of current frame, bbox of previous and next frame
        label = copy.deepcopy(self._labels[idx])
        bbox_pre = copy.deepcopy(self._labels[idx - 1]['bbox'])
        bbox_next = copy.deepcopy(self._labels[idx + 1]['bbox'])

        # transform img and gt annotation into input_img and training label
        img, label, _, bbox = self.transformation(img, label) # no augmentation
        img_pre = self.tranformation.test_transform(img_pre, bbox_pre)
        img_next = self.tranformation.test_transform(img_next, bbox_next)

        return img, label, img_id, bbox, img_pre, img_next

    def _get_single_img(self, idx):
        if type(self._items[idx]) == dict:
            img_path = self._items[idx]['path']
            img_id = self._items[idx]['id'] # get image id
        else:
            img_path = self._items[idx]
            img_id = int(os.path.splitext(os.path.basename(img_path))[0])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # load ground truth
        label = copy.deepcopy(self._labels[idx])

        # transform img and gt annotation into input_img and training label
        img, label, _, bbox = self.transformation(img, label) # no augmentation
        return img, label, img_id, bbox