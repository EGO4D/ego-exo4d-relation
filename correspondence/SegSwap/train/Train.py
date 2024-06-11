
from datetime import datetime
import numpy as np 
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import gc

import losses

def getIoU(gt_mask, pred_mask): 
    intersection = torch.logical_and(gt_mask, pred_mask > 0.1).sum(dim=(1, 2, 3))
    union = torch.logical_or(gt_mask, pred_mask).sum(dim=(1, 2, 3))
    return (intersection / union).mean()

def stable_clamp(out) : 
    eps = 1e-7
    out = torch.clamp(out, min=eps, max=1-eps)
    return out

def trainEpoch(trainLoader, backbone, netEncoder, optimizer, history, Loss, ClsLoss, batch_pos, batch_neg, warp_mask, logger, eta_corr, iter_epoch, epoch, lr, writer, warmup=False) : 
    
    backbone.eval()
    netEncoder.train()
    
    
    loss_log = []
    loss_mask_log = []
    loss_dice_log = []
    loss_cls_log = []
    
    acc_log = []
    acc_pos_log = []
    acc_neg_log = []

    trainLoader_iter = iter(trainLoader)
    
    for batch_id in tqdm(range(iter_epoch)):   
        
        try:
            batch = next(trainLoader_iter)
        except:
            trainLoader_iter = iter(trainLoader)
            batch = next(trainLoader_iter)

        ## put all into cuda
        T1 = batch['T1'].cuda()
        T2 = batch['T2'].cuda()
        
        
        RM1 = batch['RM1'].cuda()
        RM2 = batch['RM2'].cuda()
        
        M1 = batch['M1'].cuda()
        M2 = batch['M2'].cuda()
        
        FM1 = batch['FM1'].cuda()
        FM2 = batch['FM2'].cuda()
        
        Target1 = batch['target1'].cuda()
        Target2 = batch['target2'].cuda()

        NEG_IDX = batch['negative'].cuda()
        POS_IDX = torch.logical_not(NEG_IDX)

        optimizer.zero_grad()
        
        with torch.no_grad() : 
            T1 = F.normalize(backbone(T1), dim=1)
            T2 = F.normalize(backbone(T2), dim=1)

            MT1 = F.normalize(backbone(FM1.repeat(1, 3, 1, 1)), dim=1)
            MT2 = F.normalize(backbone(FM2.repeat(1, 3, 1, 1)), dim=1)


        X, Y =  T1, T2
        MX, MY, RS, RT, FMX, FMY, FMTX, FMTY = M1, M2, RM1, RM2, FM1, FM2, MT1, MT2

        O1, O2, O3 = netEncoder(X, Y, FMTX, RS, RT)
        
        output = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1)], dim=0)
        target = torch.cat([Target1, Target2], dim=0)
        pos_mask = torch.cat([POS_IDX, POS_IDX])[:, None, None, None]
            
        with torch.no_grad() : 
            
            target_pos_binary = torch.cat([Target1[POS_IDX], Target2[POS_IDX]], dim=0) 

            target_pos_view = target_pos_binary.view(target_pos_binary.size()[0], 1, 1, -1)
            target_pos_view = torch.max( target_pos_view, dim=3, keepdim=True)[0]
            
            target_pos_binary = (target_pos_binary == target_pos_view)
            target_pos_binary_float = target_pos_binary.type(torch.cuda.FloatTensor)
        
        try:
            assert output.shape == target.shape
            assert target.min() == 0 and target.max() == 1
            assert not torch.all(target == 0)
            assert not torch.any(torch.isnan(output))
        except:
            import pdb; pdb.set_trace()
        
        # reweighting loss for the +ve to be 10x
        loss_mask = Loss(output, target) * pos_mask
        weights = torch.ones_like(target)
        weights = torch.where(target == 1, weights * 10, weights)
        mask_loss = torch.mean(weights * loss_mask) 
        
        # dice loss
        dice_loss = losses.dice_loss(output, target) * pos_mask.squeeze()
        dice_loss = dice_loss.sum() / pos_mask.squeeze().sum()

        # classification loss
        cls_target = POS_IDX.float()[:, None]
        cls_loss = ClsLoss(O3, cls_target).mean()

        loss = mask_loss + dice_loss + cls_loss
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())
        if loss_mask is not None : 
            loss_mask_log.append(mask_loss.item())
        else : 
            loss_mask_log.append(0)
        loss_dice_log.append(dice_loss.item())
        loss_cls_log.append(cls_loss.item())
        
        with torch.no_grad() : 
            output_pos = torch.cat([O1.narrow(1, 2, 1)[POS_IDX], O2.narrow(1, 2, 1)[POS_IDX]], dim=0)
            output_neg = torch.cat([O1.narrow(1, 2, 1)[NEG_IDX], O2.narrow(1, 2, 1)[NEG_IDX]], dim=0)
            target_neg = torch.cat([Target1[NEG_IDX], Target2[NEG_IDX]], dim=0) 
            acc_pos = (((output_pos > 0.5) == target_pos_binary).type(torch.cuda.FloatTensor) * target_pos_binary_float).sum() / target_pos_binary_float.sum()
            acc_pos_log.append(acc_pos.item())

            element_neg = 1 - target_pos_binary_float
            acc_neg = ((((output_pos < 0.5) == (~target_pos_binary)).type(torch.cuda.FloatTensor) * element_neg).sum() + (output_neg < 0.5).type(torch.cuda.FloatTensor).sum()) / (element_neg.sum() + (1 - target_neg).sum())

            acc_neg_log.append(acc_neg.item())

            acc_log.append(acc_pos_log[-1] * 0.5 + acc_neg_log[-1] * 0.5)

        if warmup:
            for g in optimizer.param_groups:
                g['lr'] = lr * batch_id / iter_epoch 


        if batch_id % 100 == 99: 
            for g in optimizer.param_groups:
                lr_print = g['lr']
                break
            msg = '{} Batch id {:d}, Lr {:.6f}; \t | Loss : {:.3f}, Mask : {:.3f}, Dice : {:.3f}, Cls : {:.3f} |  Acc : {:.3f}%, Pos : {:.3f}%, Neg : {:.3f}%  \t '.format(datetime.now().time(), batch_id + 1, lr_print, np.mean(loss_log), np.mean(loss_mask_log), np.mean(loss_dice_log), np.mean(loss_cls_log), np.mean(acc_log) * 100, np.mean(acc_pos_log) * 100, np.mean(acc_neg_log) * 100)
            
            logger.info(msg)

            if not warmup:
                for i in range(len(output_pos)):
                    writer.add_image(f'out_mask_{i}', output_pos[i].repeat(3, 1, 1), epoch * iter_epoch + batch_id)

        if not warmup:
            writer.add_scalar('train_loss', loss_log[-1], epoch * iter_epoch + batch_id)
            writer.add_scalar('train_mask_loss', loss_mask_log[-1], epoch * iter_epoch + batch_id)
            writer.add_scalar('train_dice_loss', loss_dice_log[-1], epoch * iter_epoch + batch_id)
            writer.add_scalar('train_cls_loss', loss_cls_log[-1], epoch * iter_epoch + batch_id)
            writer.add_scalar('train_acc', acc_log[-1], epoch * iter_epoch + batch_id)
            writer.add_scalar('train_acc_pos', acc_pos_log[-1], epoch * iter_epoch + batch_id)
            writer.add_scalar('train_acc_neg', acc_neg_log[-1], epoch * iter_epoch + batch_id)
            
    history['trainLoss'].append(np.mean(loss_log))
    history['trainLossMask'].append(np.mean(loss_mask_log))
    history['trainLossDice'].append(np.mean(loss_dice_log))
    history['trainLossCls'].append(np.mean(loss_cls_log))
    
    history['trainAcc'].append(np.mean(acc_log))
    history['trainPosAcc'].append(np.mean(acc_pos_log))
    history['trainNegAcc'].append(np.mean(acc_neg_log))
    
    gc.collect()
    torch.cuda.empty_cache()

    return backbone, netEncoder, optimizer, history


def evalEpoch(trainLoader, backbone, netEncoder, history, Loss, ClsLoss, batch_pos, batch_neg, warp_mask, logger, eta_corr, iter_epoch, epoch, lr, writer, warmup=False) : 
    
    backbone.eval()
    netEncoder.eval()
    
    
    loss_log = []
    loss_mask_log = []
    loss_dice_log = []
    loss_cls_log = []
    
    acc_log = []
    acc_pos_log = []
    acc_neg_log = []
    
    miou_log = []

    trainLoader_iter = iter(trainLoader)
    
    for batch_id in tqdm(range(iter_epoch)):   
        
        try:
            batch = next(trainLoader_iter)
        except:
            trainLoader_iter = iter(trainLoader)
            batch = next(trainLoader_iter)

        ## put all into cuda
        T1 = batch['T1'].cuda()
        T2 = batch['T2'].cuda()
        
        
        RM1 = batch['RM1'].cuda()
        RM2 = batch['RM2'].cuda()
        
        M1 = batch['M1'].cuda()
        M2 = batch['M2'].cuda()
        
        FM1 = batch['FM1'].cuda()
        FM2 = batch['FM2'].cuda()
        
        Target1 = batch['target1'].cuda()
        Target2 = batch['target2'].cuda()

        NEG_IDX = batch['negative'].cuda()
        POS_IDX = torch.logical_not(NEG_IDX)
        
        with torch.no_grad() : 
            T1 = F.normalize(backbone(T1), dim=1)
            T2 = F.normalize(backbone(T2), dim=1)

            MT1 = F.normalize(backbone(FM1.repeat(1, 3, 1, 1)), dim=1)
            MT2 = F.normalize(backbone(FM2.repeat(1, 3, 1, 1)), dim=1)


        X, Y =  T1, T2
        MX, MY, RS, RT, FMX, FMY, FMTX, FMTY = M1, M2, RM1, RM2, FM1, FM2, MT1, MT2
        
        O1, O2, O3 = netEncoder(X, Y, FMTX, RS, RT)
        
        output = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1)], dim=0)
        target = torch.cat([Target1, Target2], dim=0)
        pos_mask = torch.cat([POS_IDX, POS_IDX])[:, None, None, None]
            
        with torch.no_grad() : 
            
            target_pos_binary = torch.cat([Target1[POS_IDX], Target2[POS_IDX]], dim=0) 
            target_pos_view = target_pos_binary.view(target_pos_binary.size()[0], 1, 1, -1)
            target_pos_view = torch.max( target_pos_view, dim=3, keepdim=True)[0]
            
            target_pos_binary = (target_pos_binary == target_pos_view)
            target_pos_binary_float = target_pos_binary.type(torch.cuda.FloatTensor)
        
        try:
            assert output.shape == target.shape
            assert target.min() == 0 and target.max() == 1
            assert not torch.all(target == 0)
            assert not torch.any(torch.isnan(output))
        except:
            import pdb; pdb.set_trace()
        
        # reweighting loss for the +ve to be 10x
        loss_mask = Loss(output, target) * pos_mask
        weights = torch.ones_like(target)
        weights = torch.where(target == 1, weights * 10, weights)
        mask_loss = torch.mean(weights * loss_mask)
        
        # dice loss
        dice_loss = losses.dice_loss(output, target) * pos_mask.squeeze()
        dice_loss = dice_loss.sum() / pos_mask.squeeze().sum()

        # classification loss
        cls_target = POS_IDX.float()[:, None]
        cls_loss = ClsLoss(O3, cls_target).mean()

        loss = mask_loss + dice_loss + cls_loss

        loss_log.append(loss.item())
        if loss_mask is not None : 
            loss_mask_log.append(mask_loss.item())
        else : 
            loss_mask_log.append(0)
        loss_dice_log.append(dice_loss.item())
        loss_cls_log.append(cls_loss.item())
        
        with torch.no_grad() : 
            output_pos = torch.cat([O1.narrow(1, 2, 1)[POS_IDX], O2.narrow(1, 2, 1)[POS_IDX]], dim=0)
            output_neg = torch.cat([O1.narrow(1, 2, 1)[NEG_IDX], O2.narrow(1, 2, 1)[NEG_IDX]], dim=0)
            target_neg = torch.cat([Target1[NEG_IDX], Target2[NEG_IDX]], dim=0) 
            acc_pos = (((output_pos > 0.5) == target_pos_binary).type(torch.cuda.FloatTensor) * target_pos_binary_float).sum() / target_pos_binary_float.sum()
            acc_pos_log.append(acc_pos.item())

            element_neg = 1 - target_pos_binary_float
            acc_neg = ((((output_pos < 0.5) == (~target_pos_binary)).type(torch.cuda.FloatTensor) * element_neg).sum() + (output_neg < 0.5).type(torch.cuda.FloatTensor).sum()) / (element_neg.sum() + (1 - target_neg).sum())

            acc_neg_log.append(acc_neg.item())

            acc_log.append(acc_pos_log[-1] * 0.5 + acc_neg_log[-1] * 0.5)

            # iou calculation
            O2_pos = O2.narrow(1, 2, 1)[POS_IDX]
            T2_pos = Target2[POS_IDX]
            miou = getIoU(T2_pos, O2_pos)
            miou_log.append(miou.item())

        if batch_id % 100 == 99: 
            msg = '{} Batch id {:d} | Loss : {:.3f}, Mask : {:.3f}, Dice : {:.3f}, Cls : {:.3f} |  Acc : {:.3f}%, Pos : {:.3f}%, Neg : {:.3f}%  \t '.format(datetime.now().time(), batch_id + 1, np.mean(loss_log), np.mean(loss_mask_log), np.mean(loss_dice_log), np.mean(loss_cls_log), np.mean(acc_log) * 100, np.mean(acc_pos_log) * 100, np.mean(acc_neg_log) * 100)
            
            logger.info(msg)

            if not warmup:
                for i in range(len(output_pos)):
                    writer.add_image(f'out_mask_{i}', output_pos[i].repeat(3, 1, 1), epoch * iter_epoch + batch_id)
            
    history['valLoss'].append(np.mean(loss_log))
    history['valLossMask'].append(np.mean(loss_mask_log))
    history['valLossDice'].append(np.mean(loss_dice_log))
    history['valLossCls'].append(np.mean(loss_cls_log))
    
    history['valAcc'].append(np.mean(acc_log))
    history['valPosAcc'].append(np.mean(acc_pos_log))
    history['valNegAcc'].append(np.mean(acc_neg_log))

    history['valMIoU'].append(np.mean(miou_log))
    
    writer.add_scalar('val_loss', history['valLoss'][-1], epoch * iter_epoch + batch_id)
    writer.add_scalar('val_mask_loss', history['valLossMask'][-1], epoch * iter_epoch + batch_id)
    writer.add_scalar('val_dice_loss', history['valLossDice'][-1], epoch * iter_epoch + batch_id)
    writer.add_scalar('val_cls_loss', history['valLossCls'][-1], epoch * iter_epoch + batch_id)
    writer.add_scalar('val_acc', acc_log[-1], epoch * iter_epoch + batch_id)
    writer.add_scalar('val_acc_pos', acc_pos_log[-1], epoch * iter_epoch + batch_id)
    writer.add_scalar('val_acc_neg', acc_neg_log[-1], epoch * iter_epoch + batch_id)
    writer.add_scalar('val_miou', miou_log[-1], epoch * iter_epoch + batch_id)

    netEncoder.train()

    return backbone, netEncoder, history
