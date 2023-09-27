import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('-load_model', default='./models/002500.pt', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

# from charades_dataset_full import Charades as Dataset
from JIGSAWS_dataset import JIGSAWS_FRAMES



def run(max_steps=64e3, mode='rgb', root='/ssd2/charades/Charades_v1_rgb', split='charades/charades.json', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = JIGSAWS_FRAMES(transforms=test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)    

    dataloaders = {'val': val_dataloader}

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(16)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in ['val']:
        i3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        for data in tqdm(dataloaders[phase]):
            # get the inputs
            inputs, fname = data
            # if os.path.exists(fname):
            #     continue

            b,c,t,h,w = inputs.shape
            if t > 1600:
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(fname, np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                with torch.no_grad():
                    inputs = Variable(inputs.cuda())
                    features = i3d.extract_features(inputs)
                    os.makedirs(os.path.dirname(fname[0]), exist_ok=True)
                    np.save(fname[0], features.view(-1).cpu().detach().numpy())


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
