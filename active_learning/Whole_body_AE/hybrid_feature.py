import numpy as np
import torch
import pdb


def compute_angle(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> float:
    epsilon = 1e-6 # avoid division by zero
    m1 = (y1 - y0) / (x1 - x0 + epsilon) # slope of (left joint, center joint)
    m2 = (y2 - y1) / (x2 - x1 + epsilon) # slope of (center joint, right joint)
    tan_abs = np.abs((m1 - m2) / (1 + m1 * m2 + epsilon)) # absolute value of tan of angle of each joint triangle
    # print("tan_abs: ", tan_abs)
    return np.arctan(tan_abs) # compute angle of each joint triangle using arctan of slopes

def compute_hybrid(bbox: list[float], keypoints: list[float]) -> torch.Tensor:
  """Compute hybrid feature of a human body. include 38 values (x, y, vis, angle, center of gravity)

  Args:
      bbox (list[float]): bbox of a human body. [x, y, w, h]
      keypoints (list[float]): keypoints of a human body. include 15 keypoints, each of which has 3 values (x, y, vis)

  Returns:
      torch.Tensor: feature of a human body. include 38 values (x, y, vis, angle, center of gravity). input of autoencoder
  """
  height = bbox[3] # vertical height of human body
  assert height > 0, "height of human body must be positive!"
  x_keypoints = keypoints[0::3] # x of keypoints
  y_keypoints = keypoints[1::3] # y of keypoints
  scores = keypoints[2::3] # visibility of keypoints

  # center of gravity feature
  assert sum(scores) > 0, "at least one visible keypoint is required!"

  cg_x = np.average(x_keypoints, weights=scores) # center of gravity of x. np.average(x, weights=w) = sum(x*w) / sum(w)
  cg_y = np.average(y_keypoints, weights=scores) # center of gravity of y
  normed_x_feature = (np.array(x_keypoints) - cg_x) / height # (x of keypoints - center of gravity of x) / height of human body
  normed_y_feature = (np.array(y_keypoints) - cg_y) / height # (y of keypoints - center of gravity of y) / height of human body
  # print("normed_x_feature.shape: ", normed_x_feature.shape) # (17,)
  # print("normed_x_feature: ", normed_x_feature)
  # print("normed_y_feature.shape: ", normed_y_feature.shape) # (17,)
  # print("normed_y_feature: ", normed_y_feature)

  # angle feature. angle of each joint triangle. in total 8 angles
  # angle_triangles = np.array([[6, 4, 10], [4, 6, 8], [3, 5, 7], [5, 3, 9], [9, 10, 12], [10, 9, 11], [10, 12, 14], [9, 11, 13]]) # angle of each joint triangle (left, center, right)
  angle_triangles = np.array([[8, 6, 12], [6, 8, 10], [5, 7, 9], [7, 5, 11], [11, 12, 14], [12, 11, 13], [12, 14, 16], [11, 13, 15]])
  angle_feature = np.zeros(8) # initialize angle feature
  for i in range(8):
    x0, y0 = x_keypoints[angle_triangles[i][0]], y_keypoints[angle_triangles[i][0]]
    x1, y1 = x_keypoints[angle_triangles[i][1]], y_keypoints[angle_triangles[i][1]]
    x2, y2 = x_keypoints[angle_triangles[i][2]], y_keypoints[angle_triangles[i][2]]
    # print("x0, y0, x1, y1, x2, y2: ", x0, y0, x1, y1, x2, y2)
    angle_feature[i] = compute_angle(x0, y0, x1, y1, x2, y2) # compute angle of each joint triangle

  # print("angle_feature.shape: ", angle_feature.shape) # (8,)
  # print("angle_feature: ", angle_feature)
  # print("angle in degree: ", angle_feature * 180 / np.pi, "degree")

  feature = np.hstack((normed_x_feature, normed_y_feature, angle_feature)) # combine normed_x_feature, normed_y_feature, angle_feature
  # print(feature.shape) # (42,)
  # print(feature)
  # assert feature.shape[0] == 38, "feature must have 38 values!"
  return feature

# test code for compute_hybrid
if __name__ == "__main__":
  sample_bbox = [10, 20, 30, 40]
  sample_keypoints = [ # 17 keypoints, each of which has 3 values (x, y, vis)
        411.0, 296.0, 0.0,
        397.7706599832915, 324.5295867768595, 1.0,
        394.74, 280.64, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        405.0146198830409, 339.11983471074376, 0.0,
        377.37593984962405, 339.11983471074376, 1.0,
        403.73708461707747, 377.9917443864447, 0.0,
        378.0, 385.0, 1.0,
        431.68074658890845, 361.15543076883813, 0.0,
        390.0, 427.0, 1.0,
        388.0, 437.5, 0.0,
        368.0, 438.0, 1.0,
        404.0, 517.5, 1.0,
        384.0, 518.0, 1.0,
        396.0, 582.0, 1.0,
        372.0, 581.5, 1.0
  ]
  sample_feature = compute_hybrid(sample_bbox, sample_keypoints)
  assert sample_feature.shape == (38,)
  assert type(sample_feature) == torch.Tensor