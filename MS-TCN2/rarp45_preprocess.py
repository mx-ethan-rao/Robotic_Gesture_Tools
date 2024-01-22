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
    0: "Other",
    1: "picking-up_the_needle",
    2: "positioning_the_needle_tip",
    3: "pushing_the_needle_through_the_tissue",
    4: "pulling_the_needle_out_of_the_tissue",
    5: "tying_a_knot",
    6: "cutting_the_suture",
    7: "returning_dropping_the_needle",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='RARP-45')
# parser.add_argument('--subdataset', default='Knot_Tying')
parser.add_argument('--vpath', default='/data/mingxing/RARP-45')
# parser.add_argument('--val_mode', default='LOUO')
parser.add_argument('--output', default='/data/mingxing/MS-TCN2/RARP-45')
parser.add_argument('--test_splits', default='test_split_files.npy')
parser.add_argument('--visual_feat', default='visual_features_resnet')
args = parser.parse_args()

root = args.vpath
output = args.output
# val_mode = args.val_mode
# subdataset = args.subdataset
visual_feat_path = osp.join(root, args.visual_feat)
test_splits = np.load(osp.join(root, 'splits', args.test_splits))
test_splits = [osp.relpath(f, root) for f in test_splits]
test_splits = ['_'.join(f.split('/')) for f in test_splits]
test_splits = [f.replace('segments.', '') for f in test_splits]
# visual_feat_path = args.visual_feat
# mapping = osp.join(output, 'mapping.txt')


# make features and ground truth
data_paths = []
data_paths.extend(glob.glob(osp.join(visual_feat_path, '**', 'EndoscopeImageMemory_0_sync'), recursive=True))
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
    fname = '_'.join(path.split('/'))
    np.save(osp.join(output, 'features', f'{fname}.npy'), data.T)
    os.makedirs(osp.join(output, 'groundTruth'), exist_ok=True)
    with open(osp.join(output, 'groundTruth', f'{fname}.txt'), 'w') as f:
        for i in label:
            f.write(mapping[int(i)] + '\n')

# make splits
# if val_mode == 'LOUO':
experiment_list = os.listdir(osp.join(output, 'groundTruth'))
os.makedirs(osp.join(output, 'splits'), exist_ok=True)

# if val_mode == 'LOUO':
#     LOUO_groups = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
#     # LOUO_group = experiment_list[0].split('_')[-1][0]
#     # LOUO_group = 'B'
#     for g in LOUO_groups:
#         train_list = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[1][0] != g and experiment.rsplit('_', 2)[0] == subdataset]
#         test_list = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[1][0] == g and experiment.rsplit('_', 2)[0] == subdataset]
#         with open(osp.join(output, 'splits', f'train.split.{subdataset}.{val_mode}.{g}.bundle'), 'w') as f:
#             [f.write(f"{s}\n") for s in train_list]
#         with open(osp.join(output, 'splits', f'test.split.{subdataset}.{val_mode}.{g}.bundle'), 'w') as f:
#             [f.write(f"{s}\n") for s in test_list]
# elif val_mode == 'LOSO':
#     LUSO_groups = ['1', '2', '3', '4', '5']
#     # LUSO_group = experiment_list[0].split('.')[0][-1]
#     for g in LUSO_groups:
#         train_list = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[1][-1] != g and experiment.rsplit('_', 2)[0] == subdataset]
#         test_list = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[1][-1] == g and experiment.rsplit('_', 2)[0] == subdataset]
#         with open(osp.join(output, 'splits', f'train.split.{subdataset}.{val_mode}.{g}.bundle'), 'w') as f:
#             [f.write(f"{s}\n") for s in train_list]
#         with open(osp.join(output, 'splits', f'test.split.{subdataset}.{val_mode}.{g}.bundle'), 'w') as f:
#             [f.write(f"{s}\n") for s in test_list]
# else:
# experiment = [experiment for experiment in experiment_list if experiment.rsplit('_', 2)[0] == subdataset]
# random.seed(12345)
# random.shuffle(experiment_list)
# ratio = 0.8
# train_idx = int(len(experiment_list) * ratio)
train_list = [experiment for experiment in experiment_list if experiment not in test_splits]
test_list = [experiment for experiment in experiment_list if experiment in test_splits]

with open(osp.join(output, 'splits', f'train.split.0.bundle'), 'w') as f:
    [f.write(f"{s}\n") for s in train_list]
with open(osp.join(output, 'splits', f'test.split.0.bundle'), 'w') as f:
    [f.write(f"{s}\n") for s in test_list]   










