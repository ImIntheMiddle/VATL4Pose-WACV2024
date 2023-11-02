import json
import cv2
import os
from pathlib import Path

def make_annotation(mode, path):
  """Format the annotation files of jrdb-pose for active learning.
  Args:
    mode (str): 'train' or 'val' or 'test'.
  Return: None.
  """
  print(f'Started to format {mode} annotations...\n')
  video_list = f'configs/jrdb-pose/jrdb_{mode}.txt'
  label_root = 'data/jrdb-pose/jrdb2022/labels'
  with open(video_list, 'r') as f:
    seq_list = [seq.strip() for seq in f.readlines()]
  seq_cnt = 0
  for seq in seq_list:
    json_dict = {}
    json_dict['images'] = []
    json_dict['annotations'] = []
    json_dict['categories'] = []
    print(f"No.{seq_cnt+1}: Processing {seq}...")
    seq_id = f'{seq_cnt:02d}' # sequence id (2 digits)

    # detection label
    with open(f'{label_root}/labels_2d_stitched/{seq}.json', 'r') as det:
      d_det = json.load(det)
      # pose label
      with open(f'{label_root}/labels_2d_pose_stitched_coco/{seq}.json', 'r') as pose:
        d_pose = json.load(pose)
        json_dict['categories'] = d_pose['categories']
        image_with_ann = []
        for k, d_ann in enumerate(d_pose['annotations']):
          ann_dict = {}
          # image
          image_id = d_ann['image_id']
          if image_id >= 150:
            continue
          d_image = d_pose['images'][image_id-1]
          image_path_origin = str(f"{d_image['file_name'].split('/')[-1]}")
          new_image_id = int(f'1{seq_id}{image_id:05d}') # 7-digits image id (1 + 2-digits sequence id + 5-digits image id)
          if image_id not in image_with_ann:
            image_with_ann.append(image_id)
            image_dict = {}
            image_dict["id"] = new_image_id
            image_dict["image_id"] = new_image_id
            image_dict["vid_id"] = seq_id
            image_dict["file_name"] = "images/" + d_image["file_name"]
            image_dict["is_labeled"] = True
            image_dict["has_labeled_person"] = True
            if k == 0:
              im = cv2.imread(str(path / f"{image_dict['file_name']}"))
              h, w, _ = im.shape
            image_dict["height"], image_dict["width"] = h, w
            json_dict['images'].append(image_dict)

          # annotation
          track_id = d_ann['track_id']
          ann_dict["track_id"] = track_id
          ann_dict["image_id"] = new_image_id
          ann_dict["category_id"] = d_ann["category_id"]
          ann_dict["num_keypoints"] = d_ann["num_keypoints"]
          ann_dict["is_crowd"] = 0
          ann_dict["id"] =  int(str(new_image_id) + str(track_id).zfill(3)) # 3-digits track id

          # keypoints
          d_keypoints = d_ann['keypoints']
          new_keypoints = d_keypoints.copy()
          # modify the keypoints visibility
          for i in range(2, len(d_keypoints), 3):
            if d_keypoints[i] == 0: # invisible
              new_keypoints[i] = 0
            else: # visible
              new_keypoints[i] = 1.0
          ann_dict["keypoints"] = new_keypoints

          # bbox and area
          # print(d_det["labels"].keys())
          for person in d_det["labels"][image_path_origin]:
            if person["label_id"] == "pedestrian:" + str(track_id):
              ann_dict["bbox"] = person["box"]
              ann_dict["area"] = person["attributes"]["area"]
              json_dict['annotations'].append(ann_dict)
              break
    seq_cnt += 1

    # save new annotation file
    with open(f'data/jrdb-pose/activelearning/{mode}/{seq_id}_jrdb-pose.json', 'w') as f:
      json.dump(json_dict, f) # save the json file
      print(f'--> Annotation of {seq} saved to...  {f.name} !!\n')

if __name__ == '__main__':
  path = Path('data/jrdb-pose')
  mode = ['train', 'val', 'test']
  for m in mode:
    make_annotation(m, path)
    print('Done!\n')