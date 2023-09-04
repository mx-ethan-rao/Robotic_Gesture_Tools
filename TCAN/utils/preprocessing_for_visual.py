import cv2
import numpy as np
import torch
import pandas as pd
import os
import os.path as osp
import random
from tqdm import tqdm
from multiprocessing import Pool


def generate_dataset(dir_data_root, dataset_name, seq_len, valid_len, task, fps, train_scheme, ratio = [0.7, 0.2, 0.1], seed=2023):
    experiment_list = []
    for filename in os.listdir(osp.join(dir_data_root, dataset_name, 'transcriptions')):
        if os.path.isfile(osp.join(dir_data_root, dataset_name, 'transcriptions', filename)):
            experiment_list.append(filename)

    # shuffle the experiment_list
    random.seed(seed)
    random.shuffle(experiment_list)

    if train_scheme == 'LOUO':
        LOUO_group = experiment_list[0].split('_')[-1][0]
        # LOUO_group = 'B'
        if task == 'train':
            experiment_list = [experiment for experiment in experiment_list if experiment.split('_')[-1][0] != LOUO_group]
        else:
            experiment_list = [experiment for experiment in experiment_list if experiment.split('_')[-1][0] == LOUO_group]
    elif train_scheme == 'LUSO':
        LUSO_group = experiment_list[0].split('.')[0][-1]
        # LUSO_group = '5'
        if task == 'train':
            experiment_list = [experiment for experiment in experiment_list if experiment.split('.')[0][-1] != LUSO_group]
        else:
            experiment_list = [experiment for experiment in experiment_list if experiment.split('.')[0][-1] == LUSO_group]
    else:
        train_idx = int(len(experiment_list) * ratio[0])
        valid_idx = int(len(experiment_list) * (ratio[0] +  ratio[1]))
        if task == 'train':
            experiment_list = experiment_list[:train_idx]
        elif task == 'valid':
            experiment_list = experiment_list[train_idx: valid_idx]
        else:
            experiment_list = experiment_list[valid_idx:]
    _save_data(dir_data_root, dataset_name, task, seq_len, valid_len, experiment_list, fps)

def _save_data(dir_data_root, dataset_name, task, seq_len, valid_len, experiment_list, fps):
    processed_video_root = osp.join(dir_data_root, dataset_name, 'processed_video', task)
    os.makedirs(processed_video_root, exist_ok=True)
    label_all = []
    indices_all = []
    for idx, experiment in enumerate(experiment_list):
        print('Experiment {}/{}'.format(idx, len(experiment_list)))
        data_all_tokens, label_all_tokens = _get_single_experiment_data(dir_data_root, dataset_name, seq_len, valid_len, experiment, fps)
        # with Pool(processes=fps) as pool:
        for idx, elem in enumerate(tqdm(data_all_tokens, ncols=80)):
            # pool.apply_async(torch.save, (elem, osp.join(processed_video_root, '{}_{}_data.pt'.format(experiment.split('.')[0], idx))))
            # torch.save(elem, osp.join(processed_video_root, '{}_{}_data.pt'.format(experiment.split('.')[0], idx)))
            save_tensor_as_video(elem, osp.join(processed_video_root, '{}_{}_data.avi'.format(experiment.split('.')[0], idx)), fps=fps)
        label_all.extend(label_all_tokens)
        indices_all.extend(['{}_{}'.format(experiment.split('.')[0], i) for i in range(len(label_all_tokens))])
    
    # reindex label
    unique_labels = torch.unique(torch.stack(label_all))
    mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
    # Apply the mapping to get the reindexed tensor
    label_all = [torch.tensor(mapping[element.item()]) for element in label_all]
    torch.save(label_all, osp.join(processed_video_root, 'labels.pt'))
    torch.save(indices_all, osp.join(processed_video_root, 'label_indices.pt'))


def _get_single_experiment_data(dir_data_root, dataset_name, seq_len, valid_len, suffix, fps):
    prefix = suffix.split('.')[0]

    # load data
    data_left = load_video_to_tensor(osp.join(dir_data_root, dataset_name, 'video', '{}_capture{}.avi'.format(prefix, '1')))
    data_right = load_video_to_tensor(osp.join(dir_data_root, dataset_name, 'video', '{}_capture{}.avi'.format(prefix, '2')))

    # load the labels
    labels = pd.read_csv(osp.join(dir_data_root, dataset_name, 'transcriptions', suffix), sep='\s+', header=None)
    label_all = torch.zeros(max(data_left.size(0), data_right.size(0)))
    for index, row in labels.iterrows():
        label_all[row[0]:row[1] + 1] = int(row[2][1:])

    label_all.to(torch.float32)

    non_zero_indices = torch.nonzero(label_all, as_tuple=True)

    downsampling_rate = 30 // fps
    data_left, data_right, label_all = data_left[non_zero_indices][::downsampling_rate], data_right[non_zero_indices][::downsampling_rate], label_all[non_zero_indices][::downsampling_rate]

    num_data = data_left.size(0) - seq_len
    data_left_tokens, data_right_token, label_all_tokens = [], [], []
    for i in range(num_data):
        data_left_tokens.append(data_left[i:i + seq_len])
        data_right_token.append(data_right[i:i + seq_len])
        label_all_tokens.append(label_all[i + seq_len])

    # to be discussed how to combine left-hand data and right-hand data
    # data = torch.vstack([data_left, data_right])
    # return data_left_tokens + data_right_token, label_all_tokens + label_all_tokens
    return data_left_tokens, label_all_tokens

def load_video_to_tensor(filename):
    cap = cv2.VideoCapture(filename)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Convert list of frames into a numpy array and scale to [0, 1]
    # frames_array = np.array(frames, dtype=np.float32) / 255.0
    frames_array = np.array(frames)
    frames_tensor = torch.from_numpy(frames_array)
    # [number_of_frames, height, width, channels]
    return frames_tensor

def save_tensor_as_video(tensor, output_path, fps=30.0, codec='XVID'):
    """
    Save a 4D PyTorch tensor as an AVI video file.

    :param tensor: 4D tensor with shape [number of frames, height, width, channels]
    :param output_path: Path to the output AVI file
    :param fps: Frames per second for the output video
    :param codec: Codec to be used for the output video
    """
    # Convert tensor to numpy array and scale to 0-255
    video_data = tensor.numpy().astype(np.uint8)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_data.shape[2], video_data.shape[1]))

    # Write frames to video
    for frame in video_data:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release the video writer
    out.release()

dir_data_root = '/data/mingxing/JIGSAWS'
dataset_name = 'Knot_Tying'
fps = 10
seq_len = 2 * fps
valid_len = 2 * fps
train_scheme = 'LOUO'

generate_dataset(dir_data_root, dataset_name, seq_len, valid_len, task='train', fps=fps, train_scheme=train_scheme)
generate_dataset(dir_data_root, dataset_name, seq_len, valid_len, task='valid', fps=fps, train_scheme=train_scheme)
generate_dataset(dir_data_root, dataset_name, seq_len, valid_len, task='test', fps=fps, train_scheme=train_scheme)