import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import cv2
import PIL.Image as Image
import os 
import random
import numpy as np 
import torch.nn.functional as F
import json

from pycocotools import mask as mask_utils

import warnings
warnings.filterwarnings("ignore")

def LoadImg(path) :
    return Image.open(path).convert('RGB')

def reshape_img_war(img, size=(480, 480)):
    C = 1
    if len(img.shape) == 2:
        H, W = img.shape
        img = img[..., None]
    else:
        H, W, C = img.shape
    
    temp = np.zeros((max(H, W), max(H, W), C), dtype=np.uint8)

    if H > W:
        L = (H - W) // 2
        temp[:, L:-L] = img
    elif W > H:
        L = (W - H) // 2
        temp[L:-L] = img
    else:
        temp = img

    temp = cv2.resize(temp, size, interpolation=cv2.INTER_NEAREST)

    return temp

def scale_mask(mask, resolution, stride_net=16):
    width, height = resolution
    width = int(width / stride_net)
    height = int(height / stride_net)
    scaled_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return scaled_mask

class ImageFolderTrain(Dataset):

    def __init__(self, data_dir, transform, prob_style, prob_dir, tps_grid, img_size=(480, 480), 
                        reverse=False, num_pairs=-1, train=True):
        ### each image directory should contain the same number of pairs
        
        self.data_dir = data_dir
        self.train = train
        self.reverse = reverse

        self.pairs = self._load_all_pairs(data_dir, num_pairs=num_pairs)
        self.mask_annotations = self._load_mask_annotations(data_dir)
        
        self.nb_pair = len(self.pairs) 
        self.transform = transform
        self.prob_style = prob_style
        self.prob_dir = prob_dir
        
        self.stride_net = 16
        self.nb_feat_w = img_size[0] // self.stride_net
        self.nb_feat_h = img_size[1] // self.stride_net
    
    def _load_all_pairs(self, data_dir, num_pairs):

        if self.train:
            if self.reverse:
                pairs_json = 'train_exoego_pairs.json'
            else:
                pairs_json = 'train_egoexo_pairs.json'
        else:
            if self.reverse:
                pairs_json = 'val_exoego_pairs.json'
            else:
                pairs_json = 'val_egoexo_pairs.json'

        print('LOADING: ', pairs_json)
        
        pairs = []
        for d in data_dir:
            with open(os.path.join(d, pairs_json), 'r') as fp:
                pairs.extend(json.load(fp))
        
        if num_pairs > 0:
            pairs = pairs[:num_pairs]

        return pairs
    
    def _load_mask_annotations(self, data_dir):

        d = data_dir[0]
        with open(f'{d}/split.json', 'r') as fp:
            splits = json.load(fp)
        valid_takes = splits['train'] + splits['val']

        annotations = {}
        for take in valid_takes:
            with open(f'{d}/{take}/annotation.json', 'r') as fp:
                annotations[take] = json.load(fp)

        return annotations

    def _get_negatives(self, take_id, obj_name):

        idx = random.randint(0, self.nb_pair-1)
        while take_id in self.pairs[idx][2] or obj_name in self.pairs[idx][2]:
            idx = random.randint(0, self.nb_pair-1)

        return self.pairs[idx][2]

    def load_pair_img(self, idx):
        
        root_dir = self.data_dir[0]
        if self.reverse:
            img_pth2, mask_pth2, img_pth1, mask_pth1 = self.pairs[idx]
        else:
            img_pth1, mask_pth1, img_pth2, mask_pth2 = self.pairs[idx]

        negative = False
        if torch.rand(1).item() < 0.25:
            img_pth2 = self._get_negatives(img_pth1)
            negative = True

        img1 = cv2.imread(os.path.join(root_dir, img_pth1))[..., ::-1]
        img1 = Image.fromarray(reshape_img_war(img1))

        mask1 = cv2.imread(os.path.join(root_dir, mask_pth1))[..., 0] / 255.
        mask1 = reshape_img_war(mask1)

        img2 = cv2.imread(os.path.join(root_dir, img_pth2))[..., ::-1]
        img2 = Image.fromarray(reshape_img_war(img2))

        if not negative:
            mask2 = cv2.imread(os.path.join(root_dir, mask_pth2))[..., 0] / 255.
            mask2 = reshape_img_war(mask2)
        else:
            mask2 = np.zeros((480, 480)) # default input size

        return img1, mask1, img2, mask2, negative

    def _split_img_path(self, img_p):
        root, take_id, cam, obj, _type, idx = img_p.split('//')
        return root, take_id, cam, obj, idx

    def _get_mask(self, rle_obj):
        return mask_utils.decode(rle_obj)

    def load_pair(self, idx):

        root_dir = self.data_dir[0]
        if self.reverse:
            img_pth2, mask_pth2, img_pth1, mask_pth1 = self.pairs[idx]
        else:
            img_pth1, mask_pth1, img_pth2, mask_pth2 = self.pairs[idx]

        root, take_id, cam, obj, idx = self._split_img_path(img_pth1)
        negative = False
        if torch.rand(1).item() < 0.1:
            img_pth2 = self._get_negatives(take_id, obj)
            negative = True
        
        root2, take_id2, cam2, obj2, idx2 = self._split_img_path(img_pth2)
        vid_idx = int(idx)
        vid_idx2 = int(idx2)

        img1 = cv2.imread(f"{root_dir}/{take_id}/{cam}/{vid_idx}.jpg")[..., ::-1]
        img1 = Image.fromarray(reshape_img_war(img1))

        img2 = cv2.imread(f"{root_dir}/{take_id2}/{cam2}/{vid_idx2}.jpg")[..., ::-1]
        img2 = Image.fromarray(reshape_img_war(img2))

        mask_annotation1 = self.mask_annotations[take_id] 
        mask_annotation2 = self.mask_annotations[take_id2] 

        mask1 = self._get_mask(mask_annotation1['masks'][obj][cam][idx])
        mask1 = reshape_img_war(mask1)

        if not negative:
            if idx in mask_annotation2['masks'][obj2][cam2]:
                mask2 = self._get_mask(mask_annotation2['masks'][obj2][cam2][idx])
                mask2 = reshape_img_war(mask2)
            else:
                mask2 = np.zeros((480, 480))
                negative = True
        else:
            mask2 = np.zeros((480, 480))

        return img1, mask1, img2, mask2, negative

    def load_img_random_style(self, img_dir, idx) :
        

        pth1 = os.path.join(img_dir, '{:d}_a.jpg'.format(idx))
        pth2 = os.path.join(img_dir, '{:d}_b.jpg'.format(idx))
        
        return pth1, pth2
    
    def __getitem__(self, idx):
        ## load image

        I1, FM1, I2, FM2, negative = self.load_pair(idx)

        M1 = scale_mask(FM1, FM1.shape[:2])
        M2 = scale_mask(FM2, FM2.shape[:2])

        target1 = FM1
        target2 = FM2

        mask1 = M1
        mask2 = M2
        
        mask1 = torch.from_numpy( mask1 ).unsqueeze(0)
        mask2 = torch.from_numpy( mask2 ).unsqueeze(0) 
            
        T1 = self.transform(I1)
        T2 = self.transform(I2)
        
        random_mask2 = torch.BoolTensor(mask2.size()).fill_(False) 
        random_mask1 = torch.BoolTensor(mask1.size()).fill_(False) 
        
        if torch.rand(1).item() > 0.5 and mask2.sum() > 0: 
            random_mask2 = mask2 < 0.5
        
        mask1 = mask1.type(torch.FloatTensor)
        mask2 = mask2.type(torch.FloatTensor)

        FM1 = torch.from_numpy(FM1).unsqueeze(0).type(torch.FloatTensor)
        FM2 = torch.from_numpy(FM2).unsqueeze(0).type(torch.FloatTensor)

        target1 = torch.from_numpy(target1).unsqueeze(0).type(torch.FloatTensor)
        target2 = torch.from_numpy(target2).unsqueeze(0).type(torch.FloatTensor)
            
        return {'T1' : T1,
                'T2' : T2,
                'RM1' :random_mask1,
                'RM2' :random_mask2,
                'M1' : mask1,
                'M2' : mask2,
                'FM1' : FM1,
                'FM2' : FM2,
                'target1': target1,
                'target2': target2,
                'idx': idx,
                'negative': negative,
                }
                

    def __len__(self):
        return self.nb_pair
        
## Train Data loader
def TrainDataLoader(img_dir_list, transform, batch_size, prob_style, prob_dir, tps_grid,  img_size=(480, 480), reverse=False):

    trainSet = ImageFolderTrain(img_dir_list, transform, prob_style, prob_dir, tps_grid, img_size, reverse=reverse, train=True)
    trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=8, drop_last = True)

    return trainLoader

def ValDataLoader(img_dir_list, transform, batch_size, prob_style, prob_dir, tps_grid,  img_size=(480, 480), reverse=False):

    valSet = ImageFolderTrain(img_dir_list, transform, prob_style, prob_dir, tps_grid, img_size, reverse=reverse, num_pairs=500, train=False)
    valLoader = DataLoader(dataset=valSet, batch_size=batch_size, shuffle=True, num_workers=8, drop_last = True)

    return valLoader

def getDataloader(train_dir_list, batch_size, prob_style, prob_dir, tps_grid, img_size=(480, 480), reverse=False) : 
    
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    
    
    trainLoader = TrainDataLoader(train_dir_list, transformINet, batch_size, prob_style, prob_dir, tps_grid, img_size, reverse=reverse)
    valLoader = ValDataLoader(train_dir_list, transformINet, batch_size, prob_style, prob_dir, tps_grid, img_size, reverse=reverse)

    return  trainLoader, valLoader
