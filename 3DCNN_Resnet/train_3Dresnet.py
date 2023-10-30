import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('-save_model', default='./models', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms
from transforms_ss import *
from resnet import generate_model
from torch.nn import CrossEntropyLoss


import numpy as np

from pytorch_i3d import InceptionI3d

# from charades_dataset import Charades as Dataset
from JIGSAWS_dataset import JIGSAWS

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

def run(init_lr=0.1, max_steps=2500, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', batch_size=8*5, save_model=''):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    # val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)
    dataset = JIGSAWS(transform=get_augmentation(training=True), mode='train') 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)


    dataloaders = {'train': dataloader}
    datasets = {'train': dataset}

    
    # setup the model
    # if mode == 'flow':
    #     i3d = InceptionI3d(400, in_channels=2)
    #     i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    # else:
    #     i3d = InceptionI3d(400, in_channels=3)
    #     i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    resnet3d = generate_model(model_depth=50)
    # i3d.replace_logits(15)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    resnet3d.cuda()
    resnet3d = nn.DataParallel(resnet3d)

    lr = init_lr
    optimizer = optim.SGD(resnet3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                resnet3d.train(True)
            else:
                resnet3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(most_frequent_using_bincount(labels).cuda())

                outputs, _ = resnet3d(inputs)
                criterion = CrossEntropyLoss()
                loss = criterion(outputs, labels)
                # tot_cls_loss += cls_loss.item()

                avg_loss = loss/num_steps_per_update
                tot_loss += avg_loss.item()
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 50 == 0:
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        # save model
                        torch.save(resnet3d.module.state_dict(), os.path.join(save_model, str(steps).zfill(6)+'.pt'))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
    


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
