#!/usr/bin/env python
# coding: utf-8

# ===================== Import Libraries =====================
import matplotlib.pyplot as plt  
from tqdm import tqdm   
import pandas as pd
import numpy as np
import os, sys
import torch, torchvision
import pickle
import time
import argparse
import random
import math
from PIL import Image
import seaborn as sns
import models, datasets
from torch import nn
from utils import *

# ===================== Register Hook Function =====================
feature_maps = []
def hook_feat_map(mod, inp, out):
    out = torch.nn.functional.interpolate(out,(8,8))
    feature_maps.append(torch.reshape(out, (out.shape[0],-1)))

# ===================== Parse Arguments =====================
parser = argparse.ArgumentParser()
parser.add_argument('main_repo_dir', type=str, help='path where repo is initialized') 
parser.add_argument('expt_name', type=str, help='characteristic name for experiment (used for naming output files)') 
parser.add_argument('ckpt_path', type=str, help='path to model checkpoint used for computing PD') 
parser.add_argument('csv_train_embs', type=str, help='csv file used for computing train-embeddings (used later by KNN for computing PD); make sure it has equal number of positives and negatives') 
parser.add_argument('csv_plot_pd', type=str, help='csv file used for plotting PD') 
parser.add_argument('--df_path_col', type=str, default='path', help='col name having file paths for images in the ChestXRay csv files')
parser.add_argument('--cls_name', type=str, default='Pneumothorax', help='column name for target class in the ChestXRay csv files') 
parser.add_argument('--img_size', type=int, default=128, help='size of input ChestXRay images') 
parser.add_argument('--seed', type=int, default=0, help='seed for experiment') 
parser.add_argument('--K', type=int, default=29, help='value of K for KNN') 
parser.add_argument('--knn_pos_thresh', type=float, default=0.62, help='KNN considers K neighbours, and if the mean vote is larger than this thresh we take class as positive')
parser.add_argument('--knn_neg_thresh', type=float, default=0.38, help='same as above, but if mean vote less than this thresh we treat point as negative class')
parser.add_argument('--lp_norm', type=int, default=1, help='1 or 2. used to calculate KNN distances')
parser.add_argument('--grayscale', action='store_true', help='if true, then images are converted to grayscale before computing pd. Use this for GithubCovid Images')
parser.add_argument('--num_imgs_for_pd', type=int, default=1000, help='number of images from csv_pd to consider for computing pd')
args = parser.parse_args()

# ===================== Load Model =====================
model = torch.load(args.ckpt_path).to('cuda')
model = register_hooks(model, hook_feat_map)
print('Model Loaded. \n')

# ===================== Read training data for KNN (used later for PD plots) =====================
df_train_embs = pd.read_csv(args.csv_train_embs)
print(f'Number of Embeddings: {len(df_train_embs)} \n')

# ===================== Assertions (Sanity Checks) =====================
assert(os.path.exists(args.main_repo_dir))
assert(os.path.exists(args.csv_train_embs))
assert(os.path.exists(args.csv_plot_pd))
assert(os.path.exists(args.ckpt_path))
assert(args.cls_name in df_train_embs.columns)
assert(args.df_path_col in df_train_embs.columns)
assert(args.img_size > 0)
assert(args.seed >= 0)
assert(args.K > 0)
assert(args.knn_pos_thresh > 0 and args.knn_pos_thresh < 1)
assert(args.knn_neg_thresh > 0 and args.knn_neg_thresh < 1)
assert(args.lp_norm == 1 or args.lp_norm == 2)
assert(args.num_imgs_for_pd > 0)
print('All Assertions Passed! \n')

# ===================== Dataset Transformations =====================
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.img_size,args.img_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(center_crop()),
    torchvision.transforms.Lambda(normalize())
])

dataset = datasets.ChestXRayDataset(df=df_train_embs, class_names=[args.cls_name], transform=transforms)
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=64,
                                     shuffle=False,
                                     num_workers=4, 
                                     pin_memory=True)
print('Dataset Loaded \n')


# ===================== Obtain/Store Embeddings for Training Data Subset =====================
if not os.path.exists(os.path.join(args.main_repo_dir,'output/')):
    os.makedirs(os.path.join(args.main_repo_dir,'output'))
handle = open(os.path.join(args.main_repo_dir,f'output/{args.expt_name}_tr_embs.pkl'), "wb")

with torch.no_grad():
    for b_idx,batch in tqdm(enumerate(tqdm(loader))):
        
        imgs = batch['img'].to('cuda')
        labels = batch['lab']
        paths = batch['file_name']
        
        feature_maps = []
        _ = model(imgs)
        
        info_dict = {'batch_idx':b_idx,'num_batches':len(loader),'feats':feature_maps,'labels':labels,'paths':paths}
        pickle.dump(info_dict, handle)  
        
        # free up GPU memory
        del feature_maps, info_dict
        torch.cuda.empty_cache()     
        
handle.close()
print('Embeddings for Training Data Subset Stored. \n')

# ===================== Compute PD =====================
train_embs_path = os.path.join(args.main_repo_dir,f'output/{args.expt_name}_tr_embs.pkl')
pd_pkl_path = compute_pd(args.main_repo_dir, args.ckpt_path, train_embs_path, args.csv_plot_pd, args.expt_name, args.cls_name, args.K, args.knn_pos_thresh, args.knn_neg_thresh, args.lp_norm, args.seed, args.grayscale, args.img_size, args.num_imgs_for_pd, args.df_path_col)
print('PD Computed! \n')

# ===================== Plot PD =====================
with open(pd_pkl_path, 'rb') as handle:
    batch_info = pickle.load(handle)
batch_info['pd'] = np.array(batch_info['pd'])
batch_info['labels'] = np.array(batch_info['labels'])
batch_info['preds'] = np.array(batch_info['preds'])
pos_pd_arr = (batch_info['pd']>=0)
batch_info['pd'][~pos_pd_arr] = 140 # undefined samples are assigned a pd of 140 (just for visualization purposes)
pd_mean = np.mean(batch_info['pd'][pos_pd_arr&(batch_info['pd']<=120)])
correct_preds_arr = (batch_info['preds']==batch_info['labels'])
y_lim = np.bincount(batch_info['pd']).max() 
y_step = math.ceil(y_lim/4)

sns.set(style="darkgrid")
plt.figure(figsize=(7,7))
plt.title(f'PD Plot for {args.expt_name}', fontsize=40)
plt.ylabel('No. of Images', fontsize=40)
plt.xlabel('Layer', fontsize=40)
plt.xlim((0,140))
plt.ylim((0,y_lim))
plt.xticks([0,40,80,120], fontsize=35)
plt.yticks([y_step,y_step*2,y_step*3,y_step*4], fontsize=35)

sns_hist = sns.histplot(batch_info['pd'],bins=30)
sns_hist.vlines(pd_mean,0,y_lim,color='peru',linestyle='dashed',linewidth=4)
sns_hist.get_children()[29].set_color("red")
sns_hist.get_children()[29].set_width(1)
sns_hist.figure.savefig(os.path.join(args.main_repo_dir, f"output/{args.expt_name}_pd_plot.svg"),bbox_inches='tight')

print('PD Plot Saved! \n')




















