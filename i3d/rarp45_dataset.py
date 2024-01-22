import os
import os.path
import numpy as np
import random
import torch
import json
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import torch.utils.data as data_utl
import cv2

class RARP45(data_utl.Dataset):
    def __init__(self,
                 root='/data/mingxing/RARP-45',
                 transform=None, mode='val',
                 num_frames=16, ds=1, ol=0.5,
                 small_test=False,
                 frame_dir='/data/mingxing/RARP-45/frames/',
                 label_dir='/data/mingxing/RARP-45/action_ids/',
                 class_dir='/data/mingxing/RARP-45/rarp_45_mapping.json',
                 pretrain=True, n_split=1):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        # self.ds = ds
        # self.overlap = ol
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.class_dir = class_dir
        self.pretrain = pretrain
        self.n_split = n_split

        # if self.mode == 'train':
        with open(self.class_dir, 'r') as f:
            self.classes = json.load(f)
            self.classes = {int(k): v for k, v in self.classes.items()}
        # else:
        #     with open(self.ext_class_dir, 'r') as f:
        #         self.classes = json.load(f)
        #         self.classes = {int(k): v for k, v in self.classes.items()}

        if not self.small_test:
            if self.mode == 'train':
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'train_split.npy'))
            else:
                self.train_split = np.load(
                    os.path.join(root, 'splits',
                                 f'test_split.npy'))
        else:
            self.train_split = np.load(
                os.path.join(root, 'splits', f'smalltest_split1_nf{self.num_frames}_ol{self.overlap}_ds{self.ds}.npy'))

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        ds = videoname[2]
        seq_idx = np.arange(self.num_frames) * int(ds) + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        videoname = self.train_split[index]
        vsplt = videoname[0].split('/')
        vname_splt = np.copy(vsplt)
        # if vsplt[1] == 'stereo':
        #     vname_splt[1] = 'stereo01'
        #     vname_splt[2] = vsplt[2][:-4]
        vpath = os.path.join(self.frame_dir, vsplt[0], vsplt[1], vsplt[2], vsplt[3], vsplt[4])
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        vlabel = np.load(
            os.path.join(self.label_dir, '_'.join(vname_splt) + '.npy'))
        diff = vlabel.size - vlen
        # if diff > 0:
        #     vlabel = vlabel[:-diff]
        # elif diff < 0:
        #     vlabel = np.pad(vlabel, (0, -diff), 'constant', constant_values=(0, vlabel[-1]))
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x[4:-4]))
        frame_index = self.frame_sampler(videoname, vlen)
        # seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]
        seq = self.load_rgb_frames(vpath, frame_index, path_list)
        vid = vlabel[frame_index]
        vid = np.array([self.one_hot_encode(int(label)-1, 15) for label in vid])
        # if self.pretrain:
        #     vid = torch.from_numpy(vid)
        #     vid = torch.unique_consecutive(vid)
        #     vid = vid.numpy()
        #     vid = np.pad(vid, (0, 10 - vid.shape[0]), 'constant', constant_values=(0, -1))

        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        # seq = torch.stack(seq, 1)
        # seq = seq.permute(1, 0, 2, 3)
        return torch.from_numpy(seq.transpose([3,0,1,2])), torch.from_numpy(vid).transpose(1,0).to(torch.float32)

    def __len__(self):
        # return 2
        return len(self.train_split)
    
    def load_rgb_frames(self, vpath, frame_index, path_list):
        frames = []
        for i in frame_index:
            img = cv2.imread(os.path.join(vpath, path_list[i]))[:, :, [2, 1, 0]]
            w,h,c = img.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            img = (img/255.)*2 - 1
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)
    
    def one_hot_encode(self, label, num_classes):
        """
        Convert an ordinal index label to one-hot encoding.
        
        Parameters:
        - label (int): The label (index) to be converted to one-hot encoding.
        - num_classes (int): Total number of possible classes or labels.
        
        Returns:
        - list: A one-hot encoded vector.
        """
        if label >= num_classes:
            raise ValueError("Label should be less than number of classes")

        one_hot = [0] * num_classes
        one_hot[label] = 1
        return one_hot

class RARP45_FRAMES(data.Dataset):
    def __init__(self,
                root='/data/mingxing/RARP-45',
                small_test=False,
                frame_dir='/data/mingxing/RARP-45/frames/',
                save_feat_dir='/data/mingxing/RARP-45/i3d_visual_features',
                num_frames=32,
                transforms=None):
        self.root = root
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.save_feat_dir = save_feat_dir
        self.num_frames = num_frames
        self.transform = transforms

        self.data_lst = np.load(
            os.path.join(root, 'splits', 'RARP-45_exfm.npy'))

    # def frame_sampler(self, videoname, vlen):
    #     start_idx = int(videoname[1])
    #     seq_idx = np.arange(self.num_frames) + start_idx
    #     seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
    #     return seq_idx

    def __getitem__(self, index):
        videoname = self.data_lst[index]
        seq = self.load_rgb_frames(self.frame_dir, videoname)
        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        fname = os.path.join(self.save_feat_dir, videoname.replace('.jpg', '.npy'))
        return torch.from_numpy(seq.transpose([3,0,1,2])), fname

    def __len__(self):
        # return 1
        return len(self.data_lst)

    def load_rgb_frames(self, frame_dir, videoname):
        frames = []
        img = cv2.imread(os.path.join(frame_dir, videoname))[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
        return np.asarray(frames, dtype=np.float32)                   