import os
import os.path as osp
import glob
import argparse
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='RARP-45')
parser.add_argument('--vpath', default='/data/mingxing/RARP-45/')
parser.add_argument('--num_frames', default=16, type=int)
parser.add_argument('--filter_labels', default=False, type=bool)
parser.add_argument('--keep_labels', default=3, type=int, help='Number of labels to keep')
parser.add_argument('--out', default='/data/mingxing/RARP-45/', help='Output dir')
parser.add_argument('--ratio', default=0.7, type=float, help='Proportion for training')

args = parser.parse_args()

overlap = [1, 1, 1, 1]
dss = [3, 6, 15, 30]
num_frames = args.num_frames
root = args.vpath
filter_labels = args.filter_labels
keep_labels = args.keep_labels
frames_output = osp.join(root, f'frames')
output = args.out
ratio = args.ratio
os.makedirs(output, exist_ok=True)

# file_suffix = ['*.mp4', '*.avi', '*.webm']
video_files = []
label_files = []


fps = '15'
if args.dataset == 'RARP-45':
    fps = '60'


video_files.extend(glob.glob(osp.join(root, '**', 'EndoscopeImageMemory_0_sync.avi'), recursive=True))
label_files.extend(glob.glob(osp.join(root, '**','EndoscopeImageMemory_0_sync.segments.txt'), recursive=True))
video_file_names = [osp.relpath(f, root) for f in video_files]
label_file_names = [osp.relpath(f, root) for f in label_files]

# extract frames
# for video in video_file_names:
#     if not osp.exists(osp.join(frames_output, video).rsplit('.', 1)[0]):
#         os.makedirs(osp.join(frames_output, video).rsplit('.', 1)[0], exist_ok=True)
#         cmd = "ffmpeg -i " + osp.join(root, video) + " -vsync vfr -r " + fps + " " +\
#               osp.join(frames_output, video).rsplit('.', 1)[0] + "/img_%05d.jpg"
#         print(cmd)
#         os.system(cmd)


# prepare for training splits
train_idx = int(len(label_file_names) * ratio)
video_file_prefix = [f.replace('.avi', '') for f in video_file_names]
train_data = video_file_prefix
label_files_train = label_files
new_train_list = []
for i in range(len(dss)):
    for dat, lpath in zip(train_data, label_files_train):
        vpath = osp.join(frames_output, dat)
        vlen = len([f for f in os.listdir(vpath) if osp.isfile(os.path.join(vpath, f))])
        labels = pd.read_csv(lpath, sep='\s+', header=None)
        label_all = np.zeros(vlen)
        for index, row in labels.iterrows():
            label_all[row[0]:row[1]] = int(row[2])

        # filter labels            
        if filter_labels and i == 0:
            delete_frames_indices = []
            valid_labels = list(range(keep_labels))
            for idx, l in enumerate(label_all):
                if l not in valid_labels:
                    delete_frames_indices.append(idx)
            label_all = np.delete(label_all, delete_frames_indices, axis=0)
            for frame_idx in delete_frames_indices:
                os.remove(osp.join(vpath, 'img_{:05}.jpg'.format(frame_idx + 1)))
            vlen = vlen - len(delete_frames_indices)
            
        os.makedirs(osp.join(output, 'action_ids'), exist_ok=True)
        label_prefix = '_'.join(dat.split('/'))
        if not osp.exists(osp.join(output, 'action_ids', f'{label_prefix}.npy')):
            np.save(osp.join(output, 'action_ids', f'{label_prefix}.npy'), label_all)
        
        if dat in video_file_prefix[:train_idx]:
            start_idxs = np.arange(0, vlen, int(num_frames * overlap[i] * dss[i]))
            for idx in start_idxs:
                new_train_list.append([dat, idx, dss[i]])

os.makedirs(osp.join(output, 'splits'), exist_ok=True)
np.save(osp.join(output, 'splits', 'train_split.npy'), np.array(new_train_list))
# np.save(osp.join(output, 'splits', 'test_split_files.npy'), np.array(label_files[train_idx:]))


# make feat extractor splits
frames_file = []
for s in video_file_names:
    frames_file.extend(glob.glob(osp.join(frames_output, s.rsplit('.', 1)[0], 'img*.jpg'), recursive=True))
frames_file = [osp.relpath(f, frames_output) for f in frames_file]
np.save(osp.join(output, 'splits', f'{args.dataset}_exfm.npy'), np.array(frames_file))


