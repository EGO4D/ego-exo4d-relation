import numpy as np
import cv2

import metrics

def reshape_img_nopad(img, max_dim=480):
    H, W = img.shape[:2]
    if H > W:
        ratio = 1. / H * max_dim
    else:
        ratio = 1. / W * max_dim
    newH, newW = int(H * ratio), int(W * ratio)
    img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST)
    return img

def remove_pad(img, orig_size):
    cur_H, cur_W = img.shape[:2]
    orig_H, orig_W = orig_size
    if orig_W > orig_H:
        ratio = 1. / orig_W * cur_W
    else:
        ratio = 1. / orig_H * cur_H
    new_H, new_W = int(orig_H * ratio), int(orig_W * ratio)
    if new_W > new_H:
        diff_H = (cur_H - new_H) // 2
        img = img[diff_H:-diff_H]
    else:
        diff_W = (cur_W - new_W) // 2
        img = img[:, diff_W:-diff_W]
    return img

def eval_mask(gt_masks: np.ndarray, fake_masks: np.ndarray):
    """TODO: Docstring for eval_mask.

    Args:
        gt_masks (np.ndarray): The
        fake_masks (np.ndarray): TODO

    Returns: TODO

    """
    iou = metrics.db_eval_iou(gt_masks, fake_masks)
    boundary = metrics.db_eval_boundary(gt_masks, fake_masks)
    return iou, boundary

def existence_accuracy(gt_mask: np.ndarray, pred_mask: np.ndarray):
    gt_zeros = np.all(gt_mask == 0)
    pred_zeros = np.all(pred_mask == 0)

    return gt_zeros == pred_zeros

def location_score(gt_mask, pred_mask, size=(480, 480)):
    H, W = size
    (gt_size, pred_size), (centroid_gt, centroid_pred), (gt_compact_mask, pred_compact_mask) = metrics.crop_mask(gt_mask, pred_mask)
    centroid_distance = np.sqrt((centroid_gt[0] - centroid_pred[0])**2 + (centroid_gt[1] - centroid_pred[1])**2)
    lscore = centroid_distance / np.sqrt(H**2 + W**2)
    return lscore