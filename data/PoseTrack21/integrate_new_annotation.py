import json
import cv2
import os
from pathlib import Path

def integrate_annotations(path, mode):
    """integrate the annotations of the training set and the validation set of training phase.
    get the all json files in rootdir and combine them into one json file.
    Note: skip the sequences whose images has no annotation.

    Args:
        mode (str): 'train' or 'val'.
    Return: None.

    """
    rootdir = path / f'{mode}/' # sequence from PoseTrack21
    integrated_json = {}
    integrated_json['images'] = []
    integrated_json['annotations'] = []
    integrated_json['categories'] = []
    seq_list = [os.path.basename(file) for file in sorted(rootdir.glob('*.json')) if '000000' not in os.path.basename(file)]
    ann_cnt = 0
    for seq_cnt, seq_name in enumerate(seq_list):
        print(f'\n\n{seq_name}\n')
        with open(rootdir/seq_name, 'r') as f:
            data = json.load(f)
            img_with_ann = []
            if seq_cnt == 0:
                integrated_json['categories'] = data['categories']
            for img in data['images']: # make the list of images with annotation
                if img['is_labeled']:
                    img_with_ann.append(img['image_id'])
                    integrated_json['images'].append(img)
                    print(f"add {img['file_name']}")

            for ann in data['annotations']:
                if ann['image_id'] in img_with_ann:
                    ann['iscrowd'] = 0
                    ann['area'] = ann['bbox'][2] * ann['bbox'][3]
                    integrated_json['annotations'].append(ann)
                    ann_cnt += 1

    print(f'\n\n{ann_cnt} annotations in total.\n')
    with open(rootdir / f'000000_integrated_{mode}.json', 'w') as f: # 000000_integrated_mode.json
        json.dump(integrated_json, f)
        print(f'\n{mode} saved to...  {f.name} !!\n')

if __name__ == '__main__':
    path = Path('data/PoseTrack21/activelearning')
	# integrate_annotations(mode='train')
	# integrate_annotations(mode='train_val')
    integrate_annotations(path, mode='val')
    print('Done!')