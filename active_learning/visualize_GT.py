import numpy as np
import cv2
from matplotlib import pyplot as plt

def visualize_GT(self, img, GTkpts, ann_id):
        # visualize GT skeleton
        joint_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 7], [6, 8], [7, 9], [8, 10], [0, 1], [0, 2], [1, 5], [1, 6]]
        