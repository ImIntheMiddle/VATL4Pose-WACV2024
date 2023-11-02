import json
import cv2
import os
from pathlib import Path

def integrate_annotations(mode, path):
    """integrate the annotations of the training set and the validation set.
    get the all json files in rootdir and combine them into single json file.
    *Note: skip the sequences whose images has no annotation.

    Args:
        mode (str): 'train' or 'val'.
        path (str): path to the directory of the json files.
    Return: None.
    """
    print(f'\nStarted to integrate {mode} annotations...')
    rootdir = path / f'{mode}/' # sequence from JRDB-Pose
    integrated_json = {}
    integrated_json['images'] = []
    integrated_json['annotations'] = []
    integrated_json['categories'] = []

    # exclude the already integrated json files
    seq_list = sorted(rootdir.glob('*.json'))
    seq_list = [seq for seq in seq_list if not seq.name.startswith('integrated')]

    for seq_cnt, seq_file in enumerate(seq_list):
        seq_filename = os.path.basename(seq_file)
        print(f'Processing {seq_filename}')
        with open(seq_file, 'r') as f:
            data = json.load(f)
            img_with_ann = []
            if seq_cnt == 0:
                integrated_json['categories'] = data['categories']
            for img in data['images']: # make the list of images with annotation
                if img['is_labeled']:
                    img_with_ann.append(img['image_id'])
                    integrated_json['images'].append(img)
                    # print(f"add {img['file_name']}")

            for ann in data['annotations']:
                if ann['image_id'] in img_with_ann:
                    ann['iscrowd'] = 0
                    # ann['area'] = ann['bbox'][2] * ann['bbox'][3]
                    integrated_json['annotations'].append(ann)

    with open(rootdir / f'integrated_{mode}.json', 'w') as f:
        json.dump(integrated_json, f)
        print(f'\n--> integrated {mode} annotation was saved to.. {f.name} !')

if __name__ == '__main__':
    path = Path('data/jrdb-pose/activelearning')
    integrate_annotations(mode='train', path=path)
    integrate_annotations(mode='val', path=path)
    integrate_annotations(mode='test', path=path)
    print('Done!')