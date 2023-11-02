import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio


def single_person_distance(gt, pred, normalizer, conf):
    '''

    :param gt:
    :param pred:
    :return:
    '''

    num_persons = 1
    dist = -np.ones(num_persons,)

    for person in range(num_persons):
        # -1 indicates GT not present
        if gt[person, 2] == -1:
            dist[person] = -1.0

        # 0 indicates occluded
        elif gt[person, 2] == 0 and (not conf.experiment_settings['occlusion']):
            dist[person] = -1.0

        elif (gt[person, 2] == 1 or (gt[person, 2] == 0 and conf.experiment_settings['occlusion'])) \
              and pred[person, 2] == -1:
            dist[person] = np.inf

        else:
            # Visible joint, occluded with occlusion == True,
            assert (gt[person, 2] == 1 or (gt[person, 2] == 0 and conf.experiment_settings['occlusion']))\
                   and pred[person, 2] == 1,  "Check conditions again"

            dist[person] = np.linalg.norm(gt[person, :2].astype(np.float32) - pred[person, :2].astype(np.float32))
            dist[person] /= normalizer[person]

    return dist


def PercentageCorrectKeypoint(pred_df, gt_df, config, jnts):
    """
    @params:
    - pred_df: prediction dataframe for the model outputs structured in the predefined manner
    - gt_df: ground truth dataframe in the same structure as the pred. The columns should be
        (image_id, amp, joint, u, v, cam_type)
    - conf: The configuration for the current experiment

    This function accepts the two dataframes and based on the joints, computes the distance between
    corresponding predictions. Then based on the threshold levels, computes the TP, FP and FN
    rates for the dataset and dumps the results in the output directories.
    """

    # Get the image names and joints over which the loop should be running
    joint_names = jnts

    # Create distace datarame
    distance_df = pd.DataFrame()
    distance_df['name'] = gt_df.name
    distance_df['dataset'] = gt_df.dataset
    distance_df['normalizer'] = gt_df.normalizer
    distance_df['joint'] = gt_df.joint
    distance_df['gt_uv'] = gt_df.uv
    distance_df['pred_uv'] = pred_df.uv
    distance_df['distance'] = distance_df.apply(lambda row: single_person_distance(
        row.gt_uv, row.pred_uv, row.normalizer, config), axis=1)

    threshold = np.linspace(0, 1, num=20)
    pck_dict = {}
    pck_dict['threshold'] = threshold

    # Placeholder for numpy average
    pck_dict['average'] = np.zeros_like(threshold)
    num_jnts = len(joint_names)

    # Start the loop for each joint
    for jnts in tqdm(joint_names):

        pck_dict[jnts] = []

        # get the sub tables for each of predictions and ground truths
        distance_sub = distance_df[distance_df.joint == jnts]
        distance_sub = distance_sub[distance_df.distance >= 0.0]

        total_gt = len(distance_sub.index)

        for th in threshold:
            distance_th = distance_sub[distance_sub.distance < th]
            pck_dict[jnts].append(len(distance_th.index) / total_gt)

        pck_dict['average'] += np.array(pck_dict[jnts])

    pck_dict['average'] /= num_jnts
    pck_csv = pd.DataFrame(pck_dict)
    return pck_csv