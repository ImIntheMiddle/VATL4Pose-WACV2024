import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import os
import json
import numpy as np

from .hybrid_feature import compute_hybrid


class Wholebody(Dataset):
    def __init__(self, mode: str, kp_direct=False, retrain_video_id=None) -> None:
        super().__init__()
        self.mode = mode # train or train_val
        self.retrain_video_id = retrain_video_id
        root = Path(f"data/PoseTrack21/activelearning/{self.mode}")
        if retrain_video_id is not None:
            self.file = os.path.join(root, f"{retrain_video_id}_mpii_test.json")
        else:
            self.file = os.path.join(root, f"000000_integrated_{self.mode}.json") # json file
        # self.ann = {'bbox':[], 'keypoints':[]} # bbox, keypoints
        self.num = 0 # number of human body in this json file
        self.items = []
        with open(self.file, "r") as f:
            data = json.load(f)
            item = {}
            for annotation in data["annotations"]:
                if sum(annotation['keypoints'][2::3]) == 0: # annotation must include at least one visible keypoint
                    continue
                self.num += 1 # count the number of visible human body
                id = int(annotation['id'])
                ann_id = int(str(id)[-2:] + str(annotation['image_id'])) # idの下二桁を取り出し，img_idと結合したものをann_idとする
                item["ann_id"] = ann_id # append ann_id
                if kp_direct: # if True, use keypoints as input of AE directly
                    item["feature"] = annotation['keypoints'] # append keypoints. size: 17*3 = 51
                else: # if False, use hand-crafted feature as input of AE. size: 17*2 + 8 = 42
                    item["feature"] = compute_hybrid(annotation['bbox'], annotation['keypoints']) # append hybrid feature
                self.items.append(item)
        self.items = sorted(self.items, key=lambda x:x["ann_id"]) # sort by ann_id, ascending order
        print(f"loaded {self.num} human body from {self.file}")

    # ここで取り出すデータを指定
    def __getitem__(self, index: int) -> torch.Tensor:
        input = self.items[index]["feature"] # select data following index(key of dictionary)
        return torch.tensor(input, dtype=torch.float32) # return data as tensor

    # この method がないと DataLoader を呼び出す際にエラーを吐く
    def __len__(self) -> int:
        return self.num

# test
import tqdm

if __name__ == '__main__':
    print("test start!")
    # for kp_direct
    dataset_trainval_direct = Wholebody(mode="train_val", kp_direct=True)
    print("len: ", len(dataset_trainval_direct))
    print("data sample: ", dataset_trainval_direct[0])
    for i in tqdm.tqdm(range(len(dataset_trainval_direct))):
        tmp = dataset_trainval_direct[i]
    print("test success for mode: train_val, kp_direct=True!\n")

    dataset_train_direct = Wholebody(mode="train", kp_direct=True)
    print("len: ", len(dataset_train_direct))
    print("data sample: ", dataset_train_direct[0])
    for i in tqdm.tqdm(range(len(dataset_train_direct))):
        tmp = dataset_train_direct[i]
    print("test success for mode: train, kp_direct=True!\n")

    print("load dataset...")
    dataset_trainval = Wholebody(mode="train_val")
    print("len: ", len(dataset_trainval))
    print("data sample: ", dataset_trainval[0])
    for i in tqdm.tqdm(range(len(dataset_trainval))):
        tmp = dataset_trainval[i]
    print("test success for mode: train_val!\n")

    print("load dataset...")
    dataset_train = Wholebody(mode="train")
    print("len: ", len(dataset_train))
    print("data sample: ", dataset_train[0])
    for i in tqdm.tqdm(range(len(dataset_train))): # 10000回繰り返してみる
        tmp = dataset_train[i]
    print("test success for mode: train!\n")