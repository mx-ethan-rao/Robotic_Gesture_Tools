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

# from pytorch_i3d import InceptionI3d
from resnet import generate_model
from transforms_ss import *


# from charades_dataset_full import Charades as Dataset
from JIGSAWS_dataset import JIGSAWS_FRAMES


def get_augmentation(training, scale_size=224, dataset='JIGSAWS'):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    # scale_size = config.data.input_size * 256 // 224
    if training:

        unique = torchvision.transforms.Compose([GroupMultiScaleCrop(scale_size, [1, .875, .75, .66]),
                                                 GroupRandomHorizontalFlip(is_sth='some' in dataset),
                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
                                                 GroupGaussianBlur(p=0.0),
                                                 GroupSolarization(p=0.0)]
                                                )
    else:
        unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                 GroupCenterCrop(scale_size)])

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std),
                                            Unstack(roll=False)]
                                    )
    return torchvision.transforms.Compose([unique, common])

def most_frequent_using_bincount(tensor):
    results = []
    for row in tensor:
        row_int = row.to(torch.int64)
        counts = torch.bincount(row_int)
        most_common = torch.argmax(counts).item()
        results.append(most_common)
    return torch.tensor(results, dtype=torch.int64)



def run(max_steps=64e3, mode='rgb', root='/ssd2/charades/Charades_v1_rgb', split='charades/charades.json', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = JIGSAWS_FRAMES(transforms=get_augmentation(training=False))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)    

    dataloaders = {'val': val_dataloader}

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    # if mode == 'flow':
    #     i3d = InceptionI3d(400, in_channels=2)
    # else:
    #     i3d = InceptionI3d(400, in_channels=3)
    resnet3d = generate_model(model_depth=50)
    # i3d.replace_logits(15)
    resnet3d.load_state_dict(torch.load(load_model))
    resnet3d.cuda()

    for phase in ['val']:
        resnet3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        for data in tqdm(dataloaders[phase]):
            # get the inputs
            inputs, fname = data
            # if os.path.exists(fname):
            #     continue

            # b,c,t,h,w = inputs.shape
                # wrap them in Variable
            with torch.no_grad():
                inputs = Variable(inputs.cuda())
                _, features = resnet3d(inputs)
                os.makedirs(os.path.dirname(fname[0]), exist_ok=True)
                np.save(fname[0], features.view(-1).cpu().detach().numpy())


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
