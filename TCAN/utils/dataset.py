import os
import os.path as osp
import torch
import pickle
import cv2
import unidecode
import observations
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from torch.utils import data
from torch.autograd import Variable
from torchvision import datasets, transforms
import glob
import random

import logging
from IPython import embed


class KinematicDataset(data.Dataset):
    def __init__(self, dir_data_root, dataset_name, task, fps, seq_len, valid_len, train_scheme='LOUO', is_permute=False, ratio = [0.7, 0.2, 0.1], seed=1111):
        super(KinematicDataset, self).__init__()
        self.is_permute = is_permute
        if is_permute:
            torch.manual_seed(seed)
            self.permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
        # self.dataset_name = dataset_name
        experiment_list = []
        for filename in os.listdir(osp.join(dir_data_root, dataset_name, 'transcriptions')):
            if os.path.isfile(osp.join(dir_data_root, dataset_name, 'transcriptions', filename)):
                experiment_list.append(filename)

        # shuffle the experiment_list
        # random.seed(seed)
        # random.shuffle(experiment_list)

        if train_scheme == 'LOUO':
            LOUO_group = experiment_list[0].split('_')[-1][0]
            LOUO_group = 'C'
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
        self.data_all, self.label_all = self._get_data(dir_data_root, dataset_name, seq_len, valid_len, experiment_list, fps)

    def _get_data(self, dir_data_root, dataset_name, seq_len, valid_len, experiment_list, fps):
        data_all = []
        label_all = []
        for experiment in experiment_list:
            data_all_tokens, label_all_tokens = self._get_single_experiment_data(dir_data_root, dataset_name, seq_len, valid_len, experiment, fps)
            data_all.extend(data_all_tokens)
            label_all.extend(label_all_tokens)
        unique_labels = torch.unique(torch.stack(label_all))
        mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
        # Apply the mapping to get the reindexed tensor
        label_all = [torch.tensor(mapping[element.item()]) for element in label_all]

        # to be set as hyperparameters
        # ratio = 0.8
        # index_80_percent = int(len(data_all) * ratio)
        # if task == 'train':
        #     return data_all[:index_80_percent], label_all[:index_80_percent]
        # else:
        #     return data_all[index_80_percent:], label_all[index_80_percent:]
        return data_all, label_all

    def _get_single_experiment_data(self, dir_data_root, dataset_name, seq_len, valid_len, suffix, fps):

        # load data
        data = pd.read_csv(osp.join(dir_data_root, dataset_name, 'kinematics/AllGestures', suffix), sep='\s+', header=None)

        # Convert the dataframe into numpy
        numpy_data = data.values.astype('float32')
        # Convert the numpy array to a PyTorch Tensor
        data_all = torch.from_numpy(numpy_data)

        # load labels
        labels = pd.read_csv(osp.join(dir_data_root, dataset_name, 'transcriptions', suffix), sep='\s+', header=None)
        label_all = torch.zeros(data_all.shape[0])
        for index, row in labels.iterrows():
            label_all[row[0]:row[1] + 1] = int(row[2][1:])

        label_all.to(torch.float32)

        non_zero_indices = torch.nonzero(label_all, as_tuple=True)

        downsampling_rate = 30 // fps
        data_all, label_all = data_all[non_zero_indices][::downsampling_rate], label_all[non_zero_indices][::downsampling_rate]

        # reindex the label
        # unique_labels = torch.unique(label_all)
        # mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
        # # Apply the mapping to get the reindexed tensor
        # label_all = torch.tensor([mapping[element.item()] for element in label_all])

        num_data = data_all.size(0) - seq_len
        data_all_tokens, label_all_tokens = [], []
        for i in range(num_data):
            data_all_tokens.append(data_all[i:i + seq_len])
            label_all_tokens.append(label_all[i + seq_len])
        return data_all_tokens, label_all_tokens
        
        # elif signal_type == 'visual':
        #     prefix = suffix.split('.')[0]

        #     # load data
        #     data_left = self._load_video(osp.join(dir_data_root, dataset_name, 'video', '{}_capture{}.avi'.format(prefix, '1')))
        #     data_right = self._load_video(osp.join(dir_data_root, dataset_name, 'video', '{}_capture{}.avi'.format(prefix, '2')))

        #     # load the labels
        #     labels = pd.read_csv(osp.join(dir_data_root, dataset_name, 'transcriptions', suffix), sep='\s+', header=None)
        #     label_all = torch.zeros(max(data_left.size(0), data_right.size(0)))
        #     for index, row in labels.iterrows():
        #         label_all[row[0]:row[1] + 1] = int(row[2][1:])

        #     label_all.to(torch.float32)

        #     non_zero_indices = torch.nonzero(label_all, as_tuple=True)

        #     data_left, data_right, label_all = data_left[non_zero_indices], data_right[non_zero_indices], label_all[non_zero_indices]

        #     num_data = data_left.size(0) - seq_len
        #     data_left_tokens, data_right_token, label_all_tokens = [], [], []
        #     for i in range(num_data):
        #         data_left_tokens.append(data_left[i:i + seq_len])
        #         data_right_token.append(data_right[i:i + seq_len])
        #         label_all_tokens.append(label_all[i + seq_len])

        #     # to be discussed how to combine left-hand data and right-hand data
        #     # data = torch.vstack([data_left, data_right])
        #     return data_left_tokens + data_right_token, label_all_tokens + label_all_tokens

        
    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        data = self.data_all[index]
        label = self.label_all[index]
        return data, label
    
    # def _load_video(self, filename):
    #     cap = cv2.VideoCapture(filename)
    #     frames = []

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         # Convert BGR to RGB
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frames.append(frame)

    #     cap.release()

    #     # Convert list of frames into a numpy array and scale to [0, 1]
    #     frames_array = np.array(frames, dtype=np.float32) / 255.0
    #     frames_tensor = torch.from_numpy(frames_array)
    #     return frames_tensor
    
class VisualDataset(data.Dataset):
    def __init__(self, dir_data_root, dataset_name, task, fps, seq_len, valid_len, train_scheme='LOUO', is_permute=False, ratio = [0.7, 0.2, 0.1], seed=1111):
        super(VisualDataset, self).__init__()
        self.root = osp.join(dir_data_root, dataset_name, f'{train_scheme}_processed_video', task)
        if not osp.exists(self.root):
            self.is_permute = is_permute
            if is_permute:
                torch.manual_seed(seed)
                self.permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
            experiment_list = []
            for filename in os.listdir(osp.join(dir_data_root, dataset_name, 'transcriptions')):
                if os.path.isfile(osp.join(dir_data_root, dataset_name, 'transcriptions', filename)):
                    experiment_list.append(filename)

            # shuffle the experiment_list
            # random.seed(seed)
            # random.shuffle(experiment_list)

            if train_scheme == 'LOUO':
                LOUO_group = experiment_list[0].split('_')[-1][0]
                LOUO_group = 'C'
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
            self._save_data(dir_data_root, dataset_name, task, seq_len, valid_len, experiment_list, fps, train_scheme)
        self.label_all = torch.load(osp.join(self.root, 'labels.pt'))
        self.label_indices = torch.load(osp.join(self.root, 'label_indices.pt'))
        self.transform =  transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((240, 320)),
                                # transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])
        self.video_files = self.index_videoFile()
    
    def _save_data(self, dir_data_root, dataset_name, task, seq_len, valid_len, experiment_list, fps, train_scheme):
        processed_video_root = osp.join(dir_data_root, dataset_name, f'{train_scheme}_processed_video', task)
        os.makedirs(processed_video_root, exist_ok=True)
        label_all = []
        indices_all = []
        for idx, experiment in enumerate(experiment_list):
            print('Experiment {}/{}'.format(idx, len(experiment_list)))
            data_all_tokens, label_all_tokens = self._get_single_experiment_data(dir_data_root, dataset_name, seq_len, valid_len, experiment, fps)
            # with Pool(processes=fps) as pool:
            for idx, elem in enumerate(tqdm(data_all_tokens, ncols=80)):
                # pool.apply_async(torch.save, (elem, osp.join(processed_video_root, '{}_{}_data.pt'.format(experiment.split('.')[0], idx))))
                # torch.save(elem, osp.join(processed_video_root, '{}_{}_data.pt'.format(experiment.split('.')[0], idx)))
                self.save_tensor_as_video(elem, osp.join(processed_video_root, '{}_{}_data.avi'.format(experiment.split('.')[0], idx)), fps=fps)
            label_all.extend(label_all_tokens)
            indices_all.extend(['{}_{}'.format(experiment.split('.')[0], i) for i in range(len(label_all_tokens))])
        
        # reindex label
        unique_labels = torch.unique(torch.stack(label_all))
        mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
        # Apply the mapping to get the reindexed tensor
        label_all = [torch.tensor(mapping[element.item()]) for element in label_all]
        torch.save(label_all, osp.join(processed_video_root, 'labels.pt'))
        torch.save(indices_all, osp.join(processed_video_root, 'label_indices.pt'))
        
    def index_videoFile(self):
        video_files_list = glob.glob(osp.join(self.root, '*data.avi'))
        video_files_dict = dict()
        for video in video_files_list:
            idx = '_'.join(os.path.basename(video).split('_')[:-1])
            video_files_dict[idx] = video
        video_files_list = [video_files_dict[idx_name] for idx_name in self.label_indices]
        return video_files_list
        
    def _get_single_experiment_data(self, dir_data_root, dataset_name, seq_len, valid_len, suffix, fps):
        prefix = suffix.split('.')[0]

        # load data
        data_left = self.load_video_to_tensor(osp.join(dir_data_root, dataset_name, 'video', '{}_capture{}.avi'.format(prefix, '1')))
        data_right = self.load_video_to_tensor(osp.join(dir_data_root, dataset_name, 'video', '{}_capture{}.avi'.format(prefix, '2')))

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
    
    def load_video_to_tensor(self, filename):
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
    
    def save_tensor_as_video(self, tensor, output_path, fps, codec='XVID'):
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

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_files[idx])
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
            if self.transform:
                frame = self.transform(frame)
                frame = frame.permute(1, 2, 0)
            frames.append(frame)
        cap.release()
        # frames_array = np.array(frames, dtype=np.float32) / 255.0
        # frames_tensor = torch.from_numpy(frames_array)
        frames_tensor = torch.stack(frames)
        return frames_tensor, self.label_all[idx]
    
    def _collate_fn(self, batch_data):
        data = [tup[0] for tup in batch_data]
        labels = [tup[1] for tup in batch_data]
        return torch.stack(data), torch.tensor(labels)
    
class TwoStreamDataset(data.Dataset):
    def __init__(self, dir_data_root, dataset_name, task, fps, seq_len, valid_len, train_scheme='LOUO', is_permute=False, ratio = [0.7, 0.2, 0.1], seed=1111):
        super(TwoStreamDataset, self).__init__()
        self.k_sig = KinematicDataset(dir_data_root, dataset_name, task, fps, seq_len, valid_len, train_scheme, is_permute, ratio, seed)
        self.v_sig = VisualDataset(dir_data_root, dataset_name, task, fps, seq_len, valid_len, train_scheme, is_permute, ratio, seed)
    
    def __getitem__(self, idx):
        k_data, k_label = self.k_sig[idx]
        v_data, v_label = self.v_sig[idx]
        return k_data, k_label, v_data

    def _collate_fn(self, batch_data):
        k_data = [tup[0] for tup in batch_data]
        k_labels = [tup[1] for tup in batch_data]
        v_data = [tup[2] for tup in batch_data]
        # v_labels = [tup[3] for tup in batch_data]
        return (torch.stack(k_data), torch.stack(v_data)), torch.stack(k_labels)

    def __len__(self):
        return len(self.v_sig.video_files)





class RawDataset(data.Dataset):
    def __init__(self, dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus=True, is_permute=False, seed=1111):
        super(RawDataset, self).__init__()  
        self.is_permute = is_permute
        if is_permute:
            torch.manual_seed(seed)
            self.permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.data_all, self.label_all, self.n_dict, self.dictionary = self._get_data(dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus)
       
    def __getitem__(self, index):
        data = Variable(self.data_all[index])
        if self.dataset_name == 'mnist':
            data = data.view(1, 784).float()
            # data = data.view(784).long()
            if self.is_permute:
                data = data[:, self.permute]
        label = Variable(self.label_all[index])
        return data, label

    def __len__(self):
        return len(self.data_all)
    
    def _get_data(self, dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus):
        dir_data = os.path.join(dir_data_root, dataset_name)
        if dataset_name == 'penn':
            if os.path.exists(dir_data + "/corpus") and is_corpus:
                corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
            else:
                corpus = Corpus_word(dir_data)
                pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))
                
        elif dataset_name == 'char_penn':
            dir_data = os.path.join(dir_data_root, dataset_name)
            if os.path.exists(dir_data + "/corpus") and is_corpus:

                corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
            else:
                file, testfile, valfile = getattr(observations, 'ptb')(dir_data)
                corpus = Corpus_char(file + " " + valfile + " " + testfile)
                corpus.train = char_tensor(corpus, file)
                corpus.valid = char_tensor(corpus, valfile)
                corpus.test = char_tensor(corpus, testfile)
                pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))

        elif dataset_name == 'mnist':
            corpus = Corpus_mnist(dir_data)

        elif dataset_name == 'wikitext-2':
            if os.path.exists(dir_data + "/corpus") and is_corpus:
                corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
            else:
                corpus = Corpus_word(dir_data)
                pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))
        
        elif dataset_name == 'wikitext-103':
            if os.path.exists(dir_data + "/corpus") and is_corpus:
                corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
            else:
                corpus = Corpus_word(dir_data)
                pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))

        n_dict = len(corpus.dictionary)
        dictionary = corpus.dictionary
        if task == 'train':
            data_task = corpus.train
        elif task == 'valid':
            data_task = corpus.valid
        elif task == 'test':
            data_task = corpus.test

        if self.dataset_name == 'mnist':
            if task == 'valid':
                task = 'test'
            # return getattr(data_task, '{}_data'.format(task))[:640], getattr(data_task, '{}_labels'.format(task))[:640], n_dict
            return getattr(data_task, '{}_data'.format(task)), getattr(data_task, '{}_labels'.format(task)), n_dict

        num_data = data_task.size(0) // valid_len
        data_all, label_all = [], []
        for i in range(num_data):
            if i*valid_len+seq_len+1 > data_task.size(0):
                break
            data_all.append(data_task[i*valid_len:i*valid_len+seq_len])
            label_all.append(data_task[i*valid_len+1:i*valid_len+seq_len+1])

        return data_all, label_all, n_dict, dictionary


def char_tensor(corpus, string):
    tensor = torch.zeros(len(string)).long()
    for i in tqdm(range(len(string)), ncols=80):
        tensor[i] = corpus.dictionary.char2idx[string[i]]
    return Variable(tensor)


class Dictionary_char(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    def add_word(self, char):
        self.counter[char] += 1

    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self):
        return len(self.idx2char)


class Corpus_char(object):
    def __init__(self, string):
        self.dictionary = Dictionary_char()
        for c in string:
            self.dictionary.add_word(c)
        self.dictionary.prep_dict()
        self.train = None
        self.valid = None
        self.test = None


class Corpus_word(object):
    def __init__(self, path):
        self.dictionary = Dictionary_word()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class Dictionary_word(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus_mnist(object):
    def __init__(self, path):
        self.dictionary = list(range(10))
        self.train = datasets.MNIST(root=path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
        self.valid = datasets.MNIST(root=path, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
        self.test = datasets.MNIST(root=path, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))


if __name__ == '__main__':
    dir_data_root = '../data'
    dataset_name = 'char_penn'
    task = 'train'
    batch_size = 16
    seq_len = 80
    valid_len = 40
    rawdataset = RawDataset(dir_data_root, dataset_name, task, seq_len, valid_len)
    embed()
    total = 0
    for _ in rawdataset:
        total += 1
    print(total)

