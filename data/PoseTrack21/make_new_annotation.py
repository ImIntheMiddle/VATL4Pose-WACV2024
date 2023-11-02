import json
import cv2
import os
from pathlib import Path

def make_dense_annotation_val():
	"""
	Extract the annotations of the validation set whose images has dense annotation.
	"""
	rootdir = path / f'posetrack_data/val/' # sequence from PoseTrack21
	seq_cnt = 0
	for file in sorted(rootdir.glob('*.json')):
		nframes_dense = 0
		seq_dict = {}
		seq_dict['images'] = []
		seq_dict['annotations'] = []
		seq_dict['categories'] = []
		seq_name = os.path.basename(file)
		print(f'process {seq_name}\n') # filename without extension
		seq_cnt += 1
		with open(file, 'r') as f:
			data = json.load(f)
			# print(data['images'][0]['nframes'])
			center_frame = int(data['images'][0]['nframes'] / 2) # get the center frame
			vid_id = data['images'][0]['vid_id']
			center_id = int(f'1{vid_id}{center_frame:04d}')
			# print(center_id)
			for image in data['images']:
				if (image['image_id']>center_id-17) & (image['image_id']<center_id+17) & (image['is_labeled']==True): # get the 30~ frames around the center frame
						nframes_dense += 1
						image_path = path / f"{image['file_name']}"
						print(f"add {image['file_name']}")
						im = cv2.imread(str(image_path))
						height, width, _ = im.shape
						image['width'] = width
						image['height'] = height
						seq_dict['images'].append(image)

			for annotation in data['annotations']:
				if annotation['image_id'] in [image['image_id'] for image in seq_dict['images']]:
					seq_dict['annotations'].append(annotation)
			seq_dict['categories'] = data['categories']
			print(f'{nframes_dense} frames densely annotated in {file}') # print the number of frames with dense label

		with open(path / f'activelearning/val/{seq_name}', 'w') as f: # convert to json file
			json.dump(seq_dict, f) # save the json file
			print(f'\nseq_id {seq_name} saved to...  {f.name} !!')

	print(f'\n{seq_cnt} sequences saved to...  {path}/activelearning/val/ !!\n')

def make_annotation_train(mode):
	"""Add the information of width and height to the annotation of the training set.

	Args:
		mode (str): 'train' or 'train_val'. 'train' means the training set, 'train_val' means the validation set of training phase.
	"""
	rootdir = path / f'posetrack_data/{mode}/' # sequence from PoseTrack21
	seq_cnt = 0
	for file in sorted(rootdir.glob('*.json')):
		nframes_dense = 0
		seq_dict = {}
		seq_dict['images'] = []
		seq_dict['annotations'] = []
		seq_dict['categories'] = []
		seq_name = os.path.basename(file)
		print(seq_name) # filename without extension
		seq_cnt += 1
		with open(file, 'r') as f:
			data = json.load(f)
			vid_id = data['images'][0]['vid_id']
			for image in data['images']:
				image_path = path / f"{image['file_name']}"
				print(f"add {image['file_name']}")
				im = cv2.imread(str(image_path))
				height, width, _ = im.shape
				image['width'] = width
				image['height'] = height
				seq_dict['images'].append(image)

			seq_dict['annotations'] = data['annotations']
			seq_dict['categories'] = data['categories']

		with open(path / f'activelearning/{mode}/{seq_name}', 'w') as f: # convert to json file
			json.dump(seq_dict, f) # save the json file
			print(f'\nseq_id {seq_name} saved to...  {f.name} !!')

	print(f'\n{seq_cnt} sequences saved to...  {path}/activelearning/{mode}_test/ !!\n')

if __name__ == '__main__':
	path = Path('data/PoseTrack21')
	# make_annotation_train(mode='train')
	# make_annotation_train(mode='train_val')
	# make_dense_annotation_val()
	print('Done!')