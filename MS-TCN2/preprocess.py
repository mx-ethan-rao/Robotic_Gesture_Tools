import os
import os.path as osp
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random

mapping = {
    0: "SIL",
    1: "reaching_for_needle_with_right_hand",
    2: "positioning_needle",
    3: "pushing_needle_through_tissue",
    4: "transferring_needle_from_left_to_right",
    5: "moving_to_center_with_needle_in_grip",
    6: "pulling_suture_with_left_hand",
    7: "pulling_suture_with_right_hand",
    8: "orienting_needle",
    9: "using_right_hand_to_help_tighten_suture",
    10: "loosening_more_suture",
    11: "dropping_suture_at_end_and_moving_to_end_points",
    12: "reaching_for_needle_with_left_hand",
    13: "making_C_loop_around_right_hand",
    14: "reaching_for_suture_with_right_hand",
    15: "pulling_suture_with_both_hands"
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='JIGSAWS')
parser.add_argument('--subdataset', default='Knot_Tying')
parser.add_argument('--vpath', default='/data/mingxing/JIGSAWS')
parser.add_argument('--val_mode', default='LOUO')
parser.add_argument('--output', default='/data/mingxing/MS_TCN_data/JIGSAWS')
parser.add_argument('--visual_feat', default='visual_features')
args = parser.parse_args()

root = args.vpath
output = args.output
val_mode = args.val_mode
subdataset = args.subdataset
visual_feat_path = osp.join(root, args.visual_feat)
# mapping = osp.join(output, 'mapping.txt')


# make features and ground truth
data_paths = []
data_paths.extend(glob.glob(osp.join(visual_feat_path, '**', '*_capture1'), recursive=True))
data_paths = [osp.relpath(f, visual_feat_path) for f in data_paths]
for path in tqdm(data_paths):
    label = np.load(osp.join(root, 'action_ids', '_'.join(path.split('/')) + '.npy'))
    frames_path = os.listdir(osp.join(visual_feat_path, path))
    frames_path.sort(key=lambda x: int(x[4:-4]))
    data = [np.load(osp.join(visual_feat_path, path, frame_feat)) for frame_feat in frames_path]
    non_zero_indices = np.where(label != 0)
    label = label[non_zero_indices]
    data = np.array(data)[non_zero_indices]
    os.makedirs(osp.join(output, 'features'), exist_ok=True)
    fname = path.rsplit('/', 1)[1]
    np.save(osp.join(output, 'features', f'{fname}.npy'), data.T)
    os.makedirs(osp.join(output, 'groundTruth'), exist_ok=True)
    with open(osp.join(output, 'groundTruth', f'{fname}.txt'), 'w') as f:
        for i in label:
            f.write(mapping[int(i)] + '\n')

# make splits
# if val_mode == 'LOUO':
experiment_list = os.listdir(osp.join(output, 'groundTruth'))
os.makedirs(osp.join(output, 'splits'), exist_ok=True)

if val_mode == 'LOUO':
    LOUO_groups = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    # LOUO_group = experiment_list[0].split('_')[-1][0]
    # LOUO_group = 'B'
    for g in LOUO_groups:
        train_list = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[1][0] != g and experiment.rsplit('_', 2)[0] == subdataset]
        test_list = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[1][0] == g and experiment.rsplit('_', 2)[0] == subdataset]
        with open(osp.join(output, 'splits', f'train.split.{subdataset}.{val_mode}.{g}.bundle'), 'w') as f:
            [f.write(f"{s}\n") for s in train_list]
        with open(osp.join(output, 'splits', f'test.split.{subdataset}.{val_mode}.{g}.bundle'), 'w') as f:
            [f.write(f"{s}\n") for s in test_list]
elif val_mode == 'LOSO':
    LUSO_groups = ['1', '2', '3', '4', '5']
    # LUSO_group = experiment_list[0].split('.')[0][-1]
    for g in LUSO_groups:
        train_list = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[1][-1] != g and experiment.rsplit('_', 2)[0] == subdataset]
        test_list = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[1][-1] == g and experiment.rsplit('_', 2)[0] == subdataset]
        with open(osp.join(output, 'splits', f'train.split.{subdataset}.{val_mode}.{g}.bundle'), 'w') as f:
            [f.write(f"{s}\n") for s in train_list]
        with open(osp.join(output, 'splits', f'test.split.{subdataset}.{val_mode}.{g}.bundle'), 'w') as f:
            [f.write(f"{s}\n") for s in test_list]
else:
    experiment = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[0] == subdataset]
    random.seed(12345)
    random.shuffle(experiment_list)
    ratio = 0.8
    train_idx = int(len(experiment_list) * ratio)
    train_list = experiment_list[:train_idx]
    test_list = experiment_list[train_idx:]
    with open(osp.join(output, 'splits', f'train.split.{subdataset}.{val_mode}.bundle'), 'w') as f:
        [f.write(f"{s}\n") for s in train_list]
    with open(osp.join(output, 'splits', f'test.split.{subdataset}.{val_mode}.bundle'), 'w') as f:
        [f.write(f"{s}\n") for s in test_list]   










