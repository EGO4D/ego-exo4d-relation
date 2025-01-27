# coding=utf-8
import os 
import sys 
sys.path.append('model/')
import numpy as np 
from PIL import Image
import cv2
import json

import torch 
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import transformer # model

import tqdm

from pycocotools import mask as mask_utils
import utils

MASKThresh = 0.5

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

def get_model(model_path):
    ## set gpu
    device = torch.device('cuda')

    backbone = models.resnet50(pretrained=False)   
    resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3']
    resnet_module_list = [getattr(backbone,l) for l in resnet_feature_layers]
    last_layer_idx = resnet_feature_layers.index('layer3')
    backbone = torch.nn.Sequential(*resnet_module_list[:last_layer_idx+1])

    ## load pre-trained weight
    pos_weight = 0.1
    feat_weight = 1
    dropout = 0.1
    activation = 'relu'
    mode = 'small'
    layer_type = ['I', 'C', 'I', 'C', 'I', 'N']
    drop_feat = 0.1
    feat_dim=1024

    ## model
    netEncoder = transformer.TransEncoder(feat_dim,
                                        pos_weight = pos_weight,
                                        feat_weight = feat_weight,
                                        dropout = dropout,
                                        activation = activation,
                                        mode = mode,
                                        layer_type = layer_type,
                                        drop_feat = drop_feat) 

    netEncoder.to(device)

    print ('Loading net weight from {}'.format(model_path))
    param = torch.load(model_path)
    backbone.load_state_dict(param['backbone'])
    netEncoder.load_state_dict(param['encoder'])
    backbone.eval()
    netEncoder.eval()
    backbone.to(device)
    netEncoder.to(device)

    return backbone, netEncoder 

def get_tensors(I1np, I2np, M1np):

    # masking Image1
    I1np = I1np 

    I1 = I1np
    I2 = I2np

    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    tensor1 = I1 
    tensor2 = I2 
    tensor3 = M1np


    tensor1 = transformINet(tensor1).unsqueeze(0).cuda()
    tensor2 = transformINet(tensor2).unsqueeze(0).cuda()
    tensor3 = torch.from_numpy(tensor3).unsqueeze(0).type(torch.FloatTensor).cuda()

    return I1, I2, tensor1, tensor2, tensor3

def forward_pass(backbone, netEncoder, tensor1, tensor2, tensor3):
    with torch.no_grad() : 
        feat1 = backbone( tensor1 ) ## feature
        feat1 = F.normalize(feat1, dim=1) ## l2 normalization
        feat2 = backbone( tensor2 ) ## features 
        feat2 = F.normalize(feat2, dim=1) ## l2 normalization

        fmask = backbone(tensor3.unsqueeze(0).repeat(1, 3, 1, 1))
        fmask = F.normalize(fmask, dim=1)

        out1, out2, out3 = netEncoder(feat1, feat2, fmask) ## predictions
        m1_final, m2_final, m3_final = out1[0, 2].cpu().numpy(), out2[0, 2].cpu().numpy(), out3.item()
    
    return m1_final, m2_final, m3_final


def load_frame(path, frame_idx, ret_size=False):
    img = cv2.imread(os.path.join(path, f'{frame_idx}.jpg'))[..., ::-1]
    orig_size = img.shape[:-1]
    img = Image.fromarray(reshape_img_war(img))
    if ret_size:
        return img, orig_size
    return img

def egoexo(backbone, netEncoder, annotations, ego, exo, obj, take, anno_path, pred_json):

    pred_json['masks'][obj][f'{ego}_{exo}'] = {}
    for idx in annotations['masks'][obj][ego].keys():

        ego_frame = load_frame(path=f'{anno_path}/{take}/{ego}/', frame_idx=idx)
        ego_mask = mask_utils.decode(annotations['masks'][obj][ego][idx])
        ego_mask = reshape_img_war(ego_mask)

        exo_frame, exo_size = load_frame(path=f'{anno_path}/{take}/{exo}/', frame_idx=idx, ret_size=True)

        Ix, Iy, tensor1, tensor2, tensor3 = get_tensors(ego_frame, exo_frame, ego_mask)
        mx, my, confidence = forward_pass(backbone, netEncoder, tensor1, tensor2, tensor3)

        y_step = (my > MASKThresh)
        y_step = utils.remove_pad(y_step, orig_size=exo_size)

        exo_pred = mask_utils.encode(np.asfortranarray(y_step.astype(np.uint8)))
        exo_pred['counts'] = exo_pred['counts'].decode('ascii')
        pred_json['masks'][obj][f'{ego}_{exo}'][idx] = {'pred_mask': exo_pred, 'confidence': confidence}

def exoego(backbone, netEncoder, annotations, ego, exo, obj, take, anno_path, pred_json):

    pred_json['masks'][obj][f'{exo}_{ego}'] = {}
    for idx in annotations['masks'][obj][exo].keys():

        exo_frame = load_frame(path=f'{anno_path}/{take}/{exo}/', frame_idx=idx)
        exo_mask = mask_utils.decode(annotations['masks'][obj][exo][idx])
        exo_mask = reshape_img_war(exo_mask)

        ego_frame = load_frame(path=f'{anno_path}/{take}/{ego}/', frame_idx=idx)

        Ix, Iy, tensor1, tensor2, tensor3 = get_tensors(exo_frame, ego_frame, exo_mask)
        mx, my, confidence = forward_pass(backbone, netEncoder, tensor1, tensor2, tensor3)

        y_step = (my > MASKThresh)

        ego_pred = mask_utils.encode(np.asfortranarray(y_step.astype(np.uint8)))
        ego_pred['counts'] = ego_pred['counts'].decode('ascii')
        pred_json['masks'][obj][f'{exo}_{ego}'][idx] = {'pred_mask': ego_pred, 'confidence': confidence}

def main(model_path, takes, anno_path, out_path, setting='ego-exo', save_inter=False):

    print('TOTAL TAKES: ', len(takes))

    backbone, netEncoder = get_model(model_path)
    
    results = {}
    for take in tqdm.tqdm(takes):

        with open(f'{anno_path}/{take}/annotation.json', 'r') as fp:
            annotations = json.load(fp)

        pred_json = {'masks': {}, 'subsample_idx': annotations['subsample_idx']}

        for obj in annotations['masks']:

            pred_json['masks'][obj] = {}

            cams = annotations['masks'][obj].keys()

            exo_cams = [x for x in cams if 'aria' not in x]
            ego_cams = [x for x in cams if 'aria' in x]

            for ego in ego_cams:
                for exo in exo_cams:
                    # ego -> exo
                    if setting == 'ego-exo':
                        egoexo(backbone=backbone, netEncoder=netEncoder, annotations=annotations,
                                ego=ego, exo=exo, obj=obj, take=take, anno_path=anno_path, pred_json=pred_json)
                    elif setting == 'exo-ego':
                        exoego(backbone=backbone, netEncoder=netEncoder, annotations=annotations,
                                ego=ego, exo=exo, obj=obj, take=take, anno_path=anno_path, pred_json=pred_json)
                    else:
                        raise Exception(f"Setting {setting} not recognized.")

        results[take] = pred_json

        if save_inter:
            os.makedirs(f'{out_path}/{take}', exist_ok=True)
            with open(f'{out_path}/{take}/pred_annotations.json', 'w') as fp:
                json.dump(pred_json, fp)

    return results

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--splits_path', type=str, required=True, help="Path to json of take splits")
    parser.add_argument('--split', type=str, required=True, help="Split to evaluate on")
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--setting', required=True, choices=['ego-exo', 'exo-ego'], help="ego-exo or exo-ego")
    parser.add_argument('--save_inter', action='store_true', help="Store intermediate take wise results")

    args = parser.parse_args()

    with open(args.splits_path, "r") as fp:
        splits = json.load(fp)
    
    results = main(args.ckpt_path,
                splits[args.split],
                args.data_path,
                args.out_path,
                setting=args.setting,
                save_inter=args.save_inter)

    os.makedirs(args.out_path, exist_ok=True)
    with open(f"{args.out_path}/{args.setting}_{args.split}_results.json", "w") as fp:
        json.dump({args.setting: {'results': results}}, fp)

