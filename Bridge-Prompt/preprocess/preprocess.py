import os
import os.path as osp
import glob
import argparse
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='JIGSAWS')
parser.add_argument('--vpath', default='/data/mingxing/JIGSAWS')
parser.add_argument('--num_frames', default=16, type=int)
parser.add_argument('--n_split', default=1, type=int)
args = parser.parse_args()

overlap = [1, 1, 1]
dss = [4, 8, 16]
n_split = args.n_split
num_frames = args.num_frames
root = args.vpath
frames_output = osp.join(root, 'frames')

file_suffix = ['*.mp4', '*.avi', '*.webm']
video_files = []
label_files = []


fps = '15'
if args.dataset == '50salads' or args.dataset == 'JIGSAWS':
    fps = '30'

for s in file_suffix:
    video_files.extend(glob.glob(osp.join(root, '**', s), recursive=True))
label_files.extend(glob.glob(osp.join(root, '**', 'transcriptions','*.txt'), recursive=True))
video_file_names = [osp.relpath(f, root) for f in video_files]
label_file_names = [osp.relpath(f, root) for f in label_files]

# extract frames
for video in video_file_names:
    if not osp.exists(osp.join(frames_output, video).rsplit('.', 1)[0]):
        os.makedirs(osp.join(frames_output, video).rsplit('.', 1)[0], exist_ok=True)
        cmd = "ffmpeg -i " + osp.join(root, video) + " -vsync vfr -r " + fps + " " +\
              osp.join(frames_output, video).rsplit('.', 1)[0] + "/img_%05d.jpg"
        print(cmd)
        os.system(cmd)


# prepare for training splits
train_data = [l.replace('transcriptions', 'video').replace('.txt', '_capture1') for l in label_file_names]
new_train_list = []
for i in range(len(dss)):
    for dat, lpath in zip(train_data, label_files):
        vpath = osp.join(frames_output, dat)
        vlen = len([f for f in os.listdir(vpath) if osp.isfile(os.path.join(vpath, f))])
        labels = pd.read_csv(lpath, sep='\s+', header=None)
        label_all = np.zeros(vlen)
        for index, row in labels.iterrows():
            label_all[row[0]:row[1] + 1] = int(row[2][1:])
        os.makedirs(osp.join(root, 'action_ids'), exist_ok=True)
        label_prefix = '_'.join(dat.split('/'))
        if not osp.exists(osp.join(root, 'action_ids', f'{label_prefix}.npy')):
            np.save(osp.join(root, 'action_ids', f'{label_prefix}.npy'), label_all)

        start_idxs = np.arange(0, vlen, int(num_frames * overlap[i] * dss[i]))
        for idx in start_idxs:
            new_train_list.append([dat, idx, dss[i]])

os.makedirs(osp.join(root, 'splits'), exist_ok=True)
np.save(osp.join(root, 'splits', 'train_split.npy'), np.array(new_train_list))

# make feat extractor splits
frames_file = []
for s in video_file_names:
    frames_file.extend(glob.glob(osp.join(frames_output, s.rsplit('.', 1)[0], 'img*.jpg'), recursive=True))
frames_file = [osp.relpath(f, frames_output) for f in frames_file]
np.save(osp.join(root, 'splits', f'{args.dataset}_exfm.npy'), np.array(frames_file))


