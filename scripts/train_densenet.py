#!/usr/bin/env python
# coding: utf-8

import os,sys,inspect
import numpy as np
import argparse
import pandas as pd
import torch
import torchvision, torchvision.transforms
import yaml
import random
from torch import nn
import utils
import datasets
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--main_dir', '-m', default='/jet/home/nmurali/asc170022p/nmurali/projects/misc/TMLR23_Dynamics_of_Spurious_Features')    
parser.add_argument('--config', '-c', default='configs/nih.yaml')
args = parser.parse_args()
main_dir = args.main_dir

# ============= Load config =============
config_path = os.path.join(main_dir, args.config)
config = yaml.safe_load(open(config_path)) 
print("Training Configuration: ")
print(config)  
config['output_dir'] = os.path.join(main_dir,
                                    config['output_dir'],
                                    config['expt_name'])
config['class_names'] = config['class_names'].split(",")

# ============= Dataset ====================
df = pd.read_csv(config['data_file'])
df_train = df.loc[(df['split']=='train')]
train_inds = np.asarray(df_train.index)
df_val = df.loc[(df['split']=='val')]
val_inds = np.asarray(df_val.index)
print("train: ", train_inds.shape, "test: ", val_inds.shape)

transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config['size'], config['size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(center_crop()),
        torchvision.transforms.Lambda(normalize())
    ])

dataset = datasets.ChestXRayDataset(csvpath=config['data_file'], class_names=config['class_names'], transform=transforms, seed=config['seed'])
    
# ============= Seed ====================    
np.random.seed(config['seed'])
random.seed(config['seed'])
torch.manual_seed(config['seed'])
if config['cuda']:
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============= Model ==================== 
model = models.DenseNet(num_classes=config['num_classes'], in_channels=config['channel'], drop_rate = config['drop_rate'],**models.get_densenet_params(config['model'])) 

# ============= Training ====================
utils.train(model, dataset, config, train_inds, val_inds)
print("Done")