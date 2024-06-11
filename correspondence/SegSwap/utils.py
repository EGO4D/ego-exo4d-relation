import numpy as np
import cv2

def getIoU(gt_mask, pred_mask): 
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union

def getMidDist(gt_mask, pred_mask):

    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        gt_bigc = max(gt_contours, key = cv2.contourArea)
        pred_bigc = max(pred_contours, key = cv2.contourArea)
    except:
        print('no contour')
        return -1

    gt_mid = gt_bigc.mean(axis=0)[0]
    pred_mid = pred_bigc.mean(axis=0)[0]

    return np.linalg.norm(gt_mid - pred_mid)

def getMidDistNorm(gt_mask, pred_mask):
    H, W = gt_mask.shape[:2]
    mdist = getMidDist(gt_mask, pred_mask)
    return mdist / np.sqrt(H**2 + W**2)

def getMidBinning(gt_mask, pred_mask, bin_size=5):

    H, W = gt_mask.shape
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        gt_bigc = max(gt_contours, key = cv2.contourArea)
        pred_bigc = max(pred_contours, key = cv2.contourArea)
    except:
        print('no contour')
        return -1

    gt_mid = gt_bigc.mean(axis=0)[0]
    pred_mid = pred_bigc.mean(axis=0)[0]

    # TODO: confirm x, y correspond to widht and height
    gt_x, gt_y = gt_mid.round()
    pred_x, pred_y = pred_mid.round()
    
    gt_bin_x, gt_bin_y = gt_x // bin_size, gt_y // bin_size
    pred_bin_x, pred_bin_y = pred_x // bin_size, pred_y // bin_size

    return (gt_bin_x == pred_bin_x) and (gt_bin_y == pred_bin_y)