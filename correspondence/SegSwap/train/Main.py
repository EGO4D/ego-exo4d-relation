# coding=utf-8
import argparse
import json
import itertools
import numpy as np
import os 
import torch

from datetime import datetime

import sys 
sys.path.append('../model')
import transformer # model

import Dataloader # dataloader
import Train # train

import torchvision.models as models

import logging
from tensorboardX import SummaryWriter

import sys

def get_logger(logdir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


## optimiser
parser.add_argument('--n-epoch', type=int, default=100, help='nb of epochs') 
parser.add_argument('--lr-schedule', nargs='+', type=int, default=[50], help='lr schedule') 
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--gpu', type=str, default='0', help='gpu devices')


## input / output dir 
parser.add_argument('--out-dir', type=str, help='output directory')
parser.add_argument('--train-dir', type=str, nargs='+', default=['/space_sdd/train_100k_2object_480p', '/space_sdd/train_100k_oneobject_zeropad_480p'], help='train directory with 480 x 480 resolution')

parser.add_argument('--prob-dir', type=float, nargs='+', default=[0.5, 0.5], help='probability to sample each directory')


parser.add_argument('--batch-pos', type=int, default=5, help='nb of positive samples in a batch size')
parser.add_argument('--batch-neg', type=int, default=15, help='nb of negative samples in a batch size')

parser.add_argument('--feat-pth', type=str, default='../evalBrueghel/Moco_resnet50_feat_1Scale_640p.pkl', help='validation feature')

parser.add_argument('--warp-mask', action='store_true', help='train the warped mask loss?')

parser.add_argument('--warmUpIter', type = int, default=1000, help='total iterations for learning rate warm')
parser.add_argument('--resume-pth', type = str, default=None, help='resume path')
parser.add_argument('--resume-epoch', type=int, default=0, help='resume epoch')

## paramter transformer
parser.add_argument('--mode', type=str, choices=['tiny', 'small', 'base', 'large'], default='small', help='different size of transformer encoder')
parser.add_argument('--pos-weight', type=float, default=0.1, help='weight for positional encoding')
parser.add_argument('--feat-weight', type=float, default=1, help='weight of feature')

parser.add_argument('--dropout', type=float, default=0.1, help='dropout in the transformer layer')
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'], help='activation in the transformer layer')

parser.add_argument('--prob-style', type=float, default=0.5, help='probability to sample a stylised image')

parser.add_argument('--layer-type', type=str, nargs='+', default=['I', 'C', 'I', 'C', 'I', 'N'], help='which type of layers: I is for inner image attention, C is for cross image attention, N is None')

parser.add_argument('--drop-feat', type=float, default=0.1, help='drop feature to make the task difficult')
parser.add_argument('--tps-grid', type=int, nargs='+', default=[4, 6], help='tps grid')

parser.add_argument('--eta-corr', type=float, default=8, help='loss on the correspondence term')

parser.add_argument('--iter-epoch', type=int, default=1000, help='iteration of each train epoch')
parser.add_argument('--iter-epoch-val', type=int, default=100, help='iteration of each val epoch')

parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
parser.add_argument('--reverse', action='store_true', help='train with ego -> exo instead of exo -> exo')

args = parser.parse_args()
#print (args)

writer = SummaryWriter(os.path.join(args.out_dir, 'tb_logs'))

## set gpu

device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

out_dir = 'checkpoints'
if not os.path.exists(out_dir) : 
    os.mkdir(out_dir)

args.out_dir = os.path.join(out_dir, args.out_dir) if 'checkpoints' not in args.out_dir else args.out_dir

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

logger = get_logger(args.out_dir)
logger.info(args)

## feature net
msg = 'Load MocoV2 pre-trained ResNet-50 feature...'
# print (msg)
logger.info(msg)

resume_path = '../model/moco_v2_800ep_pretrain_torchvision.pth.tar'
param = torch.load(resume_path)['model']
new_param = {}
for key in param.keys() : 
    if 'fc'  in key : 
        continue
    new_param[key] = param[key]

backbone = models.resnet50(pretrained=False)   
backbone.load_state_dict(new_param, strict=False)
resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3']
resnet_module_list = [getattr(backbone,l) for l in resnet_feature_layers]
last_layer_idx = resnet_feature_layers.index('layer3')
backbone = torch.nn.Sequential(*resnet_module_list[:last_layer_idx+1])
feat_dim=1024
backbone.cuda()


## model
netEncoder = transformer.TransEncoder(feat_dim,
                                      pos_weight = args.pos_weight,
                                      feat_weight = args.feat_weight,
                                      dropout= args.dropout,
                                      activation=args.activation,
                                      mode=args.mode,
                                      layer_type = args.layer_type,
                                      drop_feat = args.drop_feat) 

netEncoder.cuda()

## dataloader

trainLoader, valLoader = Dataloader.getDataloader(args.train_dir, args.batch_pos, args.prob_style, args.prob_dir, args.tps_grid, reverse=args.reverse)


## log  
history = {'valmAP':[], 'trainAcc':[], 'trainPosAcc':[], 'trainNegAcc':[], 'trainLoss':[], 'trainLossMask':[], 'trainLossDice':[], 'trainLossCls':[], 'trainMask':[]}
val_history = {'valmAP':[], 'valMIoU': [], 'valAcc':[], 'valPosAcc':[], 'valNegAcc':[], 'valLoss':[], 'valLossMask':[], 'valLossDice':[], 'valLossCls':[], 'valMask':[]}

optim = torch.optim.Adam
optimizer = optim(itertools.chain(*[netEncoder.parameters()]), args.lr,  weight_decay = args.weight_decay)

## resume
if args.resume_pth : 
    param = torch.load(args.resume_pth)
    optim = torch.optim.Adam
    
    backbone.load_state_dict(param['backbone'])
    netEncoder.load_state_dict(param['encoder'])
    optimizer.load_state_dict(param['optimiser']) 
    msg = 'Loading net weight and optimiser from {}'.format(args.resume_pth)
    logger.info(msg)
        
    for g in optimizer.param_groups:
        g['lr'] = args.lr

## lr schedule, lr 2e-4 than reduced to 1e-5                               
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_schedule, gamma=0.05) 

# Loss
Loss = torch.nn.BCELoss(reduction='none')
# Classification loss
ClsLoss = torch.nn.BCELoss(reduction='none')

# warmup
if args.resume_pth is None:
    backbone, netEncoder, optimizer, history = Train.trainEpoch(trainLoader, backbone, netEncoder, optimizer, history, Loss, ClsLoss, args.batch_pos, args.batch_neg, args.warp_mask, logger, args.eta_corr, args.warmUpIter, 0, args.lr, writer, warmup=True)

    msg = '{} \t Warmup {:d} iters |  Pos : {:.3f}%  Neg : {:.3f}% \t '.format(datetime.now().time(), args.warmUpIter, np.mean(history['trainPosAcc'])  * 100, np.mean(history['trainNegAcc']) * 100)    
    logger.info(msg)
        
    param = {'backbone':backbone.state_dict(), 
            'encoder':netEncoder.state_dict(),
            'optimiser':optimizer.state_dict()
            
            }

    torch.save(param, os.path.join(args.out_dir, 'net_warmup.pth'))

for epoch in range(args.resume_epoch, args.n_epoch) : 
    
    backbone, netEncoder, optimizer, history = Train.trainEpoch(trainLoader, backbone, netEncoder, optimizer, history, Loss, ClsLoss, args.batch_pos, args.batch_neg, args.warp_mask, logger, args.eta_corr, args.iter_epoch, epoch, args.lr, writer, warmup=False)
    msg = '{} TRAINING Epoch {:d} | Loss : {:.3f}  Acc : {:.3f}%  Pos : {:.3f}%  Neg : {:.3f}% '.format(datetime.now().time(), epoch, history['trainLoss'][-1], history['trainAcc'][-1] * 100, history['trainPosAcc'][-1] * 100, history['trainNegAcc'][-1] * 100)
    logger.info(msg)

    backbone, netEncoder, val_history = Train.evalEpoch(valLoader, backbone, netEncoder, val_history, Loss, ClsLoss, args.batch_pos, args.batch_neg, args.warp_mask, logger, args.eta_corr, args.iter_epoch_val, epoch, args.lr, writer, warmup=False)
    msg = '{} VAL Epoch {:d} | Loss : {:.3f} MIoU : {:.3f} Acc : {:.3f}%  Pos : {:.3f}%  Neg : {:.3f}% '.format(datetime.now().time(), epoch, val_history['valLoss'][-1], val_history['valMIoU'][-1], val_history['valAcc'][-1] * 100, val_history['valPosAcc'][-1] * 100, val_history['valNegAcc'][-1] * 100)
    logger.info(msg)
    
    param = {'backbone':backbone.state_dict(), 
             'encoder':netEncoder.state_dict(),
             'optimiser':optimizer.state_dict()
             }
    torch.save(param, os.path.join(args.out_dir, 'netLast.pth'))

    if epoch % 1 == 0:
        param = {'backbone':backbone.state_dict(), 
             'encoder':netEncoder.state_dict(),
             'optimiser':optimizer.state_dict()
             }
        torch.save(param, os.path.join(args.out_dir, f'net_ep{epoch}.pth'))
    
    
    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f :
        json.dump(history, f)
    lr_schedule.step()
    