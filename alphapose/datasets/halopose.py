"""HaloPose val Dataset (from PoseTrack21/val)"""
import os
import numpy as np
import cv2
from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from .custom import CustomDataset

@DATASET.register_module
class Halo(CustomDataset): # alphapose/models/builder.py
    """ HaloPose val Dataset from PoseTrack21/val
    Parameters
    ----------
    root: str, default './data/HaloPose'
        Path to the HaloPose dataset.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    num_joints = 17
    joint_pairs = [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    def __getitem__(self, idx):
        """Get image path and id for the previous frame, current frame, and next frame

        Args:
            idx (int): Index of the item to be retrieved from the dataset.

        Returns:
            tuple: img, keypoint label, image id, gt_bbox
        """
        if type(self._items[idx]) == dict:
            # preimg_path = self._items[idx]['path']
            # preimg_id = self._items[idx]['id']
            img_path = self._items[idx]['path']
            img_id = self._items[idx]['id'] # get image id
            # nextimg_path = self._items[idx+2]['path']
            # nextimg_id = self._items[idx+2]['id']
        else:
            # preimg_path = self._items[idx]
            # preimg_id = int(os.path.splitext(os.path.basename(preimg_path))[0])
            img_path = self._items[idx]
            img_id = int(os.path.splitext(os.path.basename(img_path))[0])
            # nextimg_path = self._items[idx+2]
            # nextimg_id = int(os.path.splitext(os.path.basename(nextimg_path))[0])

        # load ground truth, 3images, bbox
        label = copy.deepcopy(self._labels[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform img and gt annotation into input_img and training label
        img, label, _, bbox = self.transformation(img, label) # no augmentation
        return img, label, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _load_jsons(self):
        """Load all image paths and labels from annotation files into buffer."""
        _halo = self._lazy_load_ann_file()

        # calss invalid check
        classes = [c['name'] for c in _halo.loadCats(_halo.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with PoseTrack21. "

        items = []
        labels = []
        image_ids = sorted(_halo.getImgIds()) # return list of ids for each image. sorted by id
        num_frames = len(image_ids)

        for imgcnt, frame in enumerate(_halo.loadImgs(image_ids)): # iterate for each image id from start to end of sequence

            filename = frame['file_name'] # load image path
            abs_path = os.path.join(self._root, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))

            frame_label = self._check_load_keypoints(_halo, frame) # load annotation of this frame
            if not frame_label:
                continue

            # num of items are relative to person, not frame
            for person_ann in frame_label:
                items.append(abs_path)
                labels.append(person_ann)

        return items, labels

    def _check_load_keypoints(self, _halo, frame):
        """Check and load ground-truth keypoints"""
        ann_ids = _halo.getAnnIds(imgIds=frame['id'], iscrowd=False)
        objs = _halo.loadAnns(ann_ids)

        # check valid bboxes
        valid_objs = []
        width = frame['width']
        height = frame['height']

        for obj in objs:
            if max(obj['keypoints']) == 0: # invalid keypoint annotations
                continue

            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            if xmax <= xmin or ymax <= ymin: # require non-zero box area
                continue

            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
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

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs
