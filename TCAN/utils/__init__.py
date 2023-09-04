import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from utils.dataset import RawDataset, KinematicDataset, VisualDataset, TwoStreamDataset
from utils.utils import save_model, load_model, save_visual_info, record, draw_attn, Wandb_logger

def load_data(dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus):
    dataset = RawDataset(dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus)
    return dataset

def load_dataloader(dataset, signal_type, batch_size, num_workers=4, shuffle=True):
    if signal_type == 'kinematic':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    elif signal_type == 'visual'  or signal_type == 'both':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset._collate_fn)
    return dataloader
