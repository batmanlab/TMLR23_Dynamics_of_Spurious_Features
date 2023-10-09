import os, sys
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join
from torch.autograd import Variable
import pdb
import numpy as np
import pandas as pd
import torch, torchvision
import torch.nn.functional as F
import sklearn.metrics
import sklearn, sklearn.model_selection
import datasets
from random import random as rand
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import seaborn as sns

# ================== Class Definitions ==================

class center_crop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y,x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]
    
    def __call__(self, img):
        return self.crop_center(img)

class normalize(object):
    def normalize_(self, img, maxval=255):
        img = (img)/(maxval)
        return img
    
    def __call__(self, img):
        return self.normalize_(img)

# ================== Function Definitions ==================

# input: densenet121 model, provide a hook function
# output: returns a model with hooks registered for all 58 layers
def register_hooks(model, hook):
    
    for idx,layer in enumerate(model.features.denseblock1):
        if idx%2==0:
            layer.register_forward_hook(hook)
        
    for idx,layer in enumerate(model.features.denseblock2):
        if idx%2==0:
            layer.register_forward_hook(hook)
        
    for idx,layer in enumerate(model.features.denseblock3):
        if idx%2==0:
            layer.register_forward_hook(hook)
        
    for idx,layer in enumerate(model.features.denseblock4):
        if idx%2==0:
            layer.register_forward_hook(hook)
        
    return model


def to_cpu(arr):
    for idx,x in enumerate(arr):
        arr[idx] = x.to('cpu')
    return arr

def print_memory_profile(s):
    # print GPU memory
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(s)
    print(t/1024**3,r/1024**3,a/1024**3)
    print('\n')

def compute_pred_depth(arr):
    last = arr[-1]

    if last==99 or arr[-2]==99:    # uncertain pd if last or penultimate layers are uncertain
        return -25  

    p_depth = 4
    for i in range(len(arr)-1):
        ele = arr[-1-(i+1)]
        if ele!=last:
            p_depth = (len(arr)-(i+1))*4 + 4
            break
    
    return p_depth

def adjust_learning_rate(cfg, optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = cfg.lr
    if epoch in [10,15,20,30]:
        print("Old lr: ", lr)
        lr /= 10
        print("New lr: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def uniform_binning(y_conf,bin_size=0.10):
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    lower_bounds = upper_bounds- bin_size
    y_bin = []
    n_bins = len(upper_bounds)
    for s in y_conf:
        if (s <= upper_bounds[0]) :
            y_bin.append(0)
        elif (s > lower_bounds[n_bins-1]) :
            y_bin.append(n_bins-1)
        else:
            for i in range(1,n_bins-1):
                if (s > lower_bounds[i]) & (s <=upper_bounds[i] ):
                    y_bin.append(i)
                    break
    y_bin = np.asarray(y_bin)
    return y_bin

def compute_pd(main_repo_dir, ckpt_path, train_embs_path, data_csv_path, expt_name='', cls_name='Pneumothorax', K=29, knn_pos_thresh=0.62, knn_neg_thresh=0.38, lp_norm=1, seed=0, grayscale=False, img_size=128, num_imgs=100, df_path_col='path'):

    # ===================== Seed =====================   
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ===================== Create Pandas DataFrame for Validation Images =====================   
    df_val = pd.read_csv(data_csv_path)
    df_split_vals = df_val['split'].unique()
    assert(('val' in df_split_vals) and ('train' in df_split_vals))

    df_val = df_val[df_val['split']=='val']
    df_val = df_val.sample(n=num_imgs,random_state=seed)

    # ===================== Assertions (Sanity Checks) =====================
    assert(os.path.exists(ckpt_path))
    assert(os.path.exists(train_embs_path))
    assert(os.path.exists(data_csv_path))
    assert(cls_name in df_val.columns)
    assert(df_path_col in df_val.columns)
    assert(K>0)
    assert(knn_pos_thresh>0 and knn_pos_thresh<1)
    assert(knn_neg_thresh>0 and knn_neg_thresh<1)
    assert(lp_norm==1 or lp_norm==2)
    assert(img_size>0)
    assert(num_imgs>0)

    feature_maps = []
    def hook_feat_map(mod, inp, out):
        out = torch.nn.functional.interpolate(out,(8,8))
        feature_maps.append(torch.reshape(out, (out.shape[0],-1)))

    # ===================== Load Model =====================
    feature_maps = []
    model = torch.load(ckpt_path).to('cuda')
    model = register_hooks(model, hook_feat_map)

    # ===================== Dataset Transformations =====================
    if grayscale: # for GithubCovid
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size,img_size)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(center_crop()),
            torchvision.transforms.Lambda(normalize())
        ])
    else: # for NIH, MIMIC-CXR, CheXpert
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size,img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(center_crop()),
            torchvision.transforms.Lambda(normalize())
        ])

    # ===================== Storing Batch Statistics =====================
    batch_info = {}
    batch_info['readme'] = '---- K=%d ---- data_csv=%s ---- ckpt_path=%s ---- train_embs_path=%s ----' %(K,data_csv_path,ckpt_path,train_embs_path)
    batch_info['paths'] = [] # paths of test images
    batch_info['preds'] = [] # corresponding model predictions
    batch_info['labels'] = [] # labels of the test images
    batch_info['pd'] = [] # corresponding prediction depths
    batch_info['layers_knn_mean'] = [] # for each test image we have a list of knn means for every layer
    batch_info['layers_knn_mode'] = [] # for each test image we have a list of knn mode for every layer

    feature_maps = []
    def hook_feat_map(mod, inp, out):
        out = torch.nn.functional.interpolate(out,(8,8))
        feature_maps.append(torch.reshape(out, (out.shape[0],-1)))

    # ===================== Loop over val images and collect statistics =====================
    for df_idx, img_path in enumerate(tqdm(df_val[df_path_col])):        
        batch_info['paths'].append(img_path)
        with Image.open(img_path) as img:
            with torch.no_grad():
                img = transforms(img).unsqueeze(0).to('cuda')
                if img.shape[1]==4:
                    img = img[:,0,:,:].unsqueeze(0)
                feature_maps = []
                out = model(img)
                print('Model output: ')
                print(torch.sigmoid(out))
                batch_info['preds'].append(round(float(torch.sigmoid(out)),2))
                batch_info['labels'].append(df_val.iloc[df_idx][cls_name])

                # the below two lists are to store KNN distances (of neighbors) and labels across layers of DenseNet-121
                # we need to loop over the validation batches stored in pickle file and update the list across batches
                nbr_dist = [torch.empty((0)).to('cuda')]*len(feature_maps) # distance of neighbours
                nbr_labs = [torch.empty((0))]*len(feature_maps) # labels of neighbours

                with open(train_embs_path, 'rb') as handle:
                    # loop over val batches in pkl data
                    for pkl_idx in range(10000):
                        info_dict = pickle.load(handle)

                        # loop over layers in densenet
                        for layer_id,feat in enumerate(feature_maps):
                            X_i = feat.unsqueeze(1) 
                            X_j = info_dict['feats'][layer_id].unsqueeze(0)  
                            if lp_norm==2:
                                D_ij = ((X_i - X_j) ** 2).sum(-1)  
                            elif lp_norm==1:
                                D_ij = (abs(X_i - X_j)).sum(-1)  
                            else:
                                raise('Invalid lp_norm in arguments!')

                            ind_knn = torch.topk(-D_ij,K,dim=1)  # Samples <-> Dataset, (N_test, K)
                            lab_knn = info_dict['labels'][ind_knn[1]]  # (N_test, K) array of integers in [0,9]

                            # append knn preds for this layer (along with those in past batches)
                            nbr_dist[layer_id] = torch.cat((nbr_dist[layer_id],ind_knn[0]),dim=1)
                            nbr_labs[layer_id] = torch.cat((nbr_labs[layer_id],lab_knn.squeeze(2)),dim=1)

                        break_flag = (pkl_idx==info_dict['num_batches']-2) or (pkl_idx==info_dict['num_batches']-1)

                        # free GPU memory
                        del info_dict
                        torch.cuda.empty_cache()

                        if break_flag:
                            break # end of pickle objects                
                    

        for test_id in range(len(nbr_labs[0])):    
            knn_preds_mode = []  # layer-wise final KNN classification preds         
            knn_preds_mean = []  # layer-wise final KNN classification preds  

            for layer_id in range(len(feature_maps)):
                topk_inds = torch.topk(nbr_dist[layer_id],K)  # Samples <-> Dataset, (N_test, K)
                topk_labs = nbr_labs[layer_id][test_id][topk_inds[1][test_id]].unsqueeze(0)
                knn_preds_mode.append(int(topk_labs.squeeze().mode()[0]))
                knn_preds_mean.append(round(float(topk_labs.mean(dim=1)),2))

            print('Test Image: %d' %(test_id))
            print(knn_preds_mode,knn_preds_mean)
            print('\n')
            batch_info['layers_knn_mean'].append(knn_preds_mean)
            batch_info['layers_knn_mode'].append(knn_preds_mode)
            if knn_pos_thresh==0.5 and knn_neg_thresh==0.5:
                batch_info['pd'].append(compute_pred_depth(knn_preds_mode))
            else:
                arr = knn_preds_mean
                for arr_idx,ele in enumerate(arr):
                    if ele>knn_pos_thresh:
                        arr[arr_idx] = 1
                    elif ele<knn_neg_thresh:
                        arr[arr_idx] = 0
                    else:
                        arr[arr_idx] = 99   # its between pos thresh and neg thresh, means uncertain value
                print('Using pos and neg thresh we get: ')
                print(arr)
                batch_info['pd'].append(compute_pred_depth(arr))


    # ===================== Save results =====================
    if not os.path.exists(os.path.join(main_repo_dir,'output/')):
        os.makedirs(os.path.join(main_repo_dir,'output'))
    with open(os.path.join(main_repo_dir,f'output/{expt_name}_val_pd.pkl'), 'wb') as handle:
        pickle.dump(batch_info, handle)

    return os.path.join(main_repo_dir,f'output/{expt_name}_val_pd.pkl')

def train(model, dataset, cfg, train_inds, test_inds):   
    
    device = 'cuda' if cfg['cuda'] else 'cpu'

    print("Saving everying at path:")
    print(cfg['output_dir'])

    if not exists(cfg['output_dir']):
        os.makedirs(cfg['output_dir'])

    # Dataset 
    train_dataset = datasets.SubsetDataset(dataset, train_inds)
    valid_dataset = datasets.SubsetDataset(dataset, test_inds)    
    
    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg['batch_size'],
                                               shuffle=cfg['shuffle'],
                                               num_workers=cfg['threads'], 
                                               pin_memory=cfg['cuda'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg['batch_size'],
                                               shuffle=cfg['shuffle'],
                                               num_workers=cfg['threads'], 
                                               pin_memory=cfg['cuda'])

    # Optimizer
    if cfg['opt'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], amsgrad=True, weight_decay=1e-5)
    if cfg['opt'] == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    elif cfg['opt'] == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-5)

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []

    model.to(device)
    
    for epoch in range(start_epoch, cfg['num_epochs']):
        avg_loss = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader)
            
        auc_valid, task_aucs, task_outputs, task_targets = valid_test_epoch(cfg=cfg, 
                                     name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader)
        
        if np.mean(auc_valid) > best_metric:
            try:
                os.remove(join(cfg['output_dir'], f'best-auc{best_metric:4.4f}.pt')) # remove previous best checkpoint
            except:
                pass
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg['output_dir'], f'best-auc{np.mean(auc_valid):4.4f}.pt'))

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(cfg['output_dir'], f'e{epoch + 1}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        if epoch%cfg['save_freq']==0:
            torch.save(model, join(cfg['output_dir'], f'e{epoch + 1}-auc{np.mean(auc_valid):4.4f}.pt'))   
        

    return metrics, best_metric, weights_for_best_validauc
      
def train_epoch(cfg, epoch, model, device, train_loader, optimizer, limit=None):
    print(f'===================================TRAINING EPOCH-{epoch+1}===========================================')
    model.train()
    
    t = tqdm(train_loader)
    num_batches = len(train_loader)

    for batch_idx, samples in enumerate(t):
        if limit and (batch_idx > limit):
            print("breaking out")
            break

        optimizer.zero_grad()
        images = samples["img"].to(device)
        targets = samples["lab"].to(device)

        if len(targets.shape) == 1 and cfg['num_classes'] != 1:
            targets = F.one_hot(targets, num_classes=cfg['num_classes'])
            
        outputs = model(images)

        wt = torch.tensor([cfg['pos_weights']]).to('cuda')
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=wt)
        loss = criterion(outputs, targets).to(device)
                
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()
        
        loss.backward()

        loss = loss.detach().cpu().numpy()
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {loss:4.4f}')

        optimizer.step()

    print(f'=====================================================================================================================') 
    return loss

def valid_test_epoch(cfg, name, epoch, model, device, data_loader, limit=None):
    print(f'===================================VALIDATING EPOCH-{epoch+1}===========================================')
    model.eval()

    task_outputs=[]
    task_targets=[]
        
    with torch.no_grad():
        t = tqdm(data_loader)
        val_loss = torch.zeros(1).to(device).float()
        
        # iterate dataloader
        for batch_idx, samples in enumerate(t):
            index = epoch*len(t)+batch_idx
            if limit and (batch_idx > limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)

            if len(targets.shape) == 1 and cfg['num_classes'] != 1:
                targets = F.one_hot(targets, num_classes=cfg['num_classes'])
            
            outputs = model(images)

            wt = torch.tensor([cfg['pos_weights']]).to('cuda')
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=wt)
                
            val_loss += criterion(outputs, targets).to(device)

            task_outputs.append(outputs.detach().cpu().numpy())
            task_targets.append(targets.detach().cpu().numpy())

        val_loss /= batch_idx  
        val_loss = val_loss.detach().cpu()[0]
        t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {val_loss:4.4f}')
        print(f'Epoch {epoch + 1} - {name} - Loss = {val_loss:4.4f}')

        task_outputs = np.concatenate(task_outputs)
        task_targets = np.concatenate(task_targets)

        # compute val metrics
        val_auc = sklearn.metrics.roc_auc_score(task_targets[:,0], task_outputs[:,0])        
        print(f'Epoch {epoch + 1} - {name}: ')
        print(f'AUC = {val_auc:4.4f}')
        print(f'=====================================================================================================================')

    return val_auc, val_loss, task_outputs, task_targets