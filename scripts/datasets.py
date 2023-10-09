from PIL import Image
from matplotlib import pyplot as plt
from os.path import join
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os,sys,os.path
import pandas as pd
import collections
import pprint
import torch
import torch.nn.functional as F
import torchvision
import skimage.transform
from random import *
import random as rand

class Dataset():
    def __init__(self):
        pass
    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.class_names,counts))
    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()
    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")
        
class SubsetDataset(Dataset):
    def __init__(self, dataset, idxs=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.class_names = dataset.class_names
        
        self.idxs = idxs
        
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        
        self.csv = self.csv.reset_index(drop=True)
        
        if hasattr(self.dataset, 'which_dataset'):
            self.which_dataset = self.dataset.which_dataset[self.idxs]
    
    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n","\n  ")
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


    
class ChestXRayDataset(Dataset):

    def __init__(self, class_names, transform, df=None, csvpath=None, data_aug=None, seed=0, unique_patients=True):

        super(ChestXRayDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.class_names = class_names        
        self.transform = transform
        self.data_aug = data_aug

        if df is not None:
            self.csv = df
        elif csvpath is not None:
            self.csvpath = csvpath
            self.csv = pd.read_csv(self.csvpath)
        else:
            raise Exception("Must pass in either df or csvpath")

        self.csv = self.csv.fillna(0)
        self.labels = []
        for name in self.class_names:
            if name in self.csv.columns:
                mask = self.csv[name]   
            self.labels.append(mask.values)    
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        self.labels[self.labels == -1] = 0 # make all the -1 values into 0 to keep things simple
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self), self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = str(self.csv.iloc[idx]["path"])
        img = Image.open(img_path)
        if len(img.getbands())>1:   # needed for NIH dataset as some images are RGBA 4 channel
            img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)            
        if self.data_aug is not None:
            img = self.data_aug(img)   

        return {"img":img, "lab":self.labels[idx], "idx":idx, "file_name" : img_path}


    


