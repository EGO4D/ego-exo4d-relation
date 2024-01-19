import argparse
import os
from typing import List

import metrics
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torchvision.ops import masks_to_boxes
from tqdm import tqdm


def compute_size_boundry_centroid(binary_mask):
    is_empty = not np.any(binary_mask)
    H, W = binary_mask.shape
    if is_empty:
        return (0, 0), (H // 2, W // 2), (W // 2, W // 2, H // 2, H // 2)
    else:
        y, x = np.where(binary_mask == True)
        left_boundary = np.min(x)
        right_boundary = np.max(x)
        top_boundary = np.min(y)
        bottom_boundary = np.max(y)

        centroid_x = int(left_boundary + right_boundary) // 2
        centroid_y = int(top_boundary + bottom_boundary) // 2

        width, height = right_boundary - left_boundary + 1, bottom_boundary - top_boundary + 1

    return (
        (width, height),
        (centroid_x, centroid_y),
        (left_boundary, right_boundary, top_boundary, bottom_boundary),
    )


def crop_mask(mask1, mask2):
    """
    crop a pair of masks according to the size of the larger mask
    """
    assert (
        mask1.shape == mask2.shape
    ), f"Annotation({mask1.shape}) and segmentation:{mask2.shape} dimensions do not match."

    mask1 = np.pad(
        mask1,
        ((mask1.shape[0], mask1.shape[0]), (mask1.shape[0], mask1.shape[0])),
        mode="constant",
        constant_values=False,
    )
    mask2 = np.pad(
        mask2,
        ((mask2.shape[0], mask2.shape[0]), (mask2.shape[0], mask2.shape[0])),
        mode="constant",
        constant_values=False,
    )

    size_1, centroid_1, boundary_1 = compute_size_boundry_centroid(mask1)
    size_2, centroid_2, boundary_2 = compute_size_boundry_centroid(mask2)

    width, height = max(size_1[0], size_2[0]), max(size_1[1], size_2[1])
    # print(f"Crop Width: {width}, Crop Height: {height}")

    compact_mask_1 = mask1[
        centroid_1[1] - height // 2 : centroid_1[1] + height // 2 + 1,
        centroid_1[0] - width // 2 : centroid_1[0] + width // 2 + 1,
    ]
    compact_mask_2 = mask2[
        centroid_2[1] - height // 2 : centroid_2[1] + height // 2 + 1,
        centroid_2[0] - width // 2 : centroid_2[0] + width // 2 + 1,
    ]

    return (size_1, size_2), (centroid_1, centroid_2), (compact_mask_1, compact_mask_2)


def crop_resize_using_mask(img: torch.Tensor, mask: torch.Tensor, target_size: List[int]):
    """TODO: Docstring for crop_resize_using_mask.

    Args:
        img (torch.Tensor): shape: [C,H,W].
        mask (torch.Tensor): shape: [H,W].
        target_size (List[int])

    Returns: TODO

    """
    if mask.sum() == 0:
        return torch.ones([img.shape[0]] + list(target_size))

    bbox = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    if h > 0 and w > 0:
        img = TF.crop(img, int(y1), int(x1), int(h), int(w))  # type: ignore
    img = TF.resize(img, target_size)
    return img


def load_mask(mask_path: str, mode="L"):
    """load mask to numpy array

    Args:
        img_path (str): path to image.

    Returns: np.ndarray | None. dtype: np.bool.
        return None if file not exist or occurred some errors during loading.

    """
    try:
        if not os.path.isfile(mask_path):
            print(f"file not existed for image: {mask_path}")
            return None

        mask_pil = Image.open(mask_path).convert(mode)
        mask_pil = mask_pil.resize(IMG_SIZE, resample=Image.NEAREST)
        # foreground: >127 -> True
        return np.array(mask_pil) > 127
    except Exception as e:
        print(f"Exception while loading image: {mask_path}: {e}")
        return None


IMG_SIZE = (256, 256)  # resize image to this size for evaluation.
DIR_GT = "ground-truths"
DIR_PRED = "predictions"


def evaluate(results_dir):
    dir_gt = os.path.join(results_dir, DIR_GT)
    dir_pred = os.path.join(results_dir, DIR_PRED)

    img_names = os.listdir(dir_gt)
    print(f"number of images: {len(img_names)}")

    # metrics

    # mask existence or not
    mask_existence_gt_list = []
    mask_existence_pred_list = []

    # iou and boundary are calculated after register the prediction with the ground truth.
    riou_score_list = []
    rboundary_score_list = []
    location_error_list = []

    for _, img_name in enumerate(tqdm(img_names)):
        if not img_name.endswith(".png"):
            continue

        mask_gt = load_mask(os.path.join(dir_gt, img_name))
        if mask_gt is None:  # skip if erros with GT image.
            continue

        # mask exist if any value is not False.
        mask_existence_gt = np.any(mask_gt)

        mask_pred = load_mask(os.path.join(dir_pred, img_name))

        if mask_pred is None:
            mask_existence_pred = False
        else:
            mask_existence_pred = np.any(mask_pred)

        if not mask_existence_gt:
            mask_existence_gt_list.append(mask_existence_gt)
            mask_existence_pred_list.append(mask_existence_pred)
            # skip other metrics if gt_mask is empty (no object)
            continue

        # object exists and prediction exists
        if mask_pred is not None:
            (
                (gt_size, fake_size),
                (centroid_gt, centroid_pred),
                (gt_compact_mask, pred_compact_mask),
            ) = crop_mask(mask_gt, mask_pred)
            centroid_distance = np.sqrt(
                (centroid_gt[0] - centroid_pred[0]) ** 2
                + (centroid_gt[1] - centroid_pred[1]) ** 2
            )

            riou = metrics.db_eval_iou(gt_compact_mask, pred_compact_mask)
            rboundary = metrics.db_eval_boundary(gt_compact_mask, pred_compact_mask)
            location_error = centroid_distance / np.sqrt(IMG_SIZE[0] ** 2 + IMG_SIZE[1] ** 2)
        else:
            riou, rboundary, location_error = 0.0, 0.0, 1.0

        mask_existence_gt_list.append(mask_existence_gt)
        mask_existence_pred_list.append(mask_existence_pred)
        riou_score_list.append(riou)
        rboundary_score_list.append(rboundary)
        location_error_list.append(location_error)

    # balanced mask existence accuracy
    balanced_mask_existence_acc = balanced_accuracy_score(
        mask_existence_gt_list, mask_existence_pred_list
    )
    riou_score = np.mean(riou_score_list)
    rboundary_score = np.mean(rboundary_score_list)
    location_error = np.mean(location_error_list)

    print(f"mask_existence balanced accuracy={balanced_mask_existence_acc}")
    print(f"Registered IOU score={riou_score}")
    print(f"Registered boundary score={rboundary_score}")
    print(f"Location Error={location_error}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, help="The directory of the results")

    args = parser.parse_args()

    evaluate(args.results_dir)
