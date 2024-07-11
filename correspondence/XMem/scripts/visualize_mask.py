import os
import argparse
import json

from pycocotools import mask as mask_utils
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from natsort import natsorted
from tqdm.auto import tqdm
import pandas as pd


def getIoU(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union


def getMidDist(gt_mask, pred_mask):
    gt_contours, _ = cv2.findContours(
        gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    pred_contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    try:
        gt_bigc = max(gt_contours, key=cv2.contourArea)
        pred_bigc = max(pred_contours, key=cv2.contourArea)
    except:
        return np.nan

    gt_mid = gt_bigc.mean(axis=0)[0]
    pred_mid = pred_bigc.mean(axis=0)[0]

    return np.linalg.norm(gt_mid - pred_mid)


def getMidDistNorm(gt_mask, pred_mask):
    H, W = gt_mask.shape[:2]
    mdist = getMidDist(gt_mask, pred_mask)
    return mdist / np.sqrt(H**2 + W**2)


def getMidBinning(gt_mask, pred_mask, bin_size=5):
    H, W = gt_mask.shape
    gt_contours, _ = cv2.findContours(
        gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    pred_contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    try:
        gt_bigc = max(gt_contours, key=cv2.contourArea)
        pred_bigc = max(pred_contours, key=cv2.contourArea)
    except:
        return np.nan

    gt_mid = gt_bigc.mean(axis=0)[0]
    pred_mid = pred_bigc.mean(axis=0)[0]

    # TODO: confirm x, y correspond to widht and height
    gt_x, gt_y = gt_mid.round()
    pred_x, pred_y = pred_mid.round()

    gt_bin_x, gt_bin_y = gt_x // bin_size, gt_y // bin_size
    pred_bin_x, pred_bin_y = pred_x // bin_size, pred_y // bin_size

    return (gt_bin_x == pred_bin_x) and (gt_bin_y == pred_bin_y)


def blend_mask(input_img, binary_mask, alpha=0.5):
    if input_img.ndim == 2:
        return input_img

    mask_image = np.zeros(input_img.shape, np.uint8)
    mask_image[:, :, 1] = 255
    mask_image = mask_image * np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)

    blend_image = input_img[:, :, :]
    pos_idx = binary_mask > 0
    for ind in range(input_img.ndim):
        ch_img1 = input_img[:, :, ind]
        ch_img2 = mask_image[:, :, ind]
        ch_img3 = blend_image[:, :, ind]
        ch_img3[pos_idx] = alpha * ch_img1[pos_idx] + (1 - alpha) * ch_img2[pos_idx]
        blend_image[:, :, ind] = ch_img3
    return blend_image


def show_img(img):
    plt.figure(facecolor="white", figsize=(30, 10), dpi=100)
    plt.grid("off")
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def save_img(img, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    MAX_RES = 960

    pil_img = Image.fromarray(img)
    width, height = pil_img.size
    max_res = max(width, height)
    if max_res > MAX_RES:
        scale = MAX_RES / max_res
        pil_img = pil_img.resize(
            (int(width * scale), int(height * scale)), resample=Image.BILINEAR
        )
    pil_img.save(output)


def main(args):
    split_data = None
    splits_path = args.split_json
    with open(splits_path, "r") as fp:
        split_data = json.load(fp)

    if split_data is None:
        print("No split found")
        return
    if args.compute_stats:
        takes = [take_id for take_id in split_data[args.split]]
        df_list = []
        for take_id in tqdm(takes):
            df_i = process_take(
                take_id,
                args.input,
                args.pred,
                args.output,
                args.split,
                args.visualize,
            )
            if df_i is not None:
                df_list.append(df_i)
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(os.path.join(args.output, f"{args.split}.csv"), index=False)
    else:
        df = pd.read_csv(os.path.join(args.output, f"{args.split}.csv"))

    take_mean_iou = df.groupby(["take"])["iou"].mean()
    object_mean_iou = df.groupby(["object"])["iou"].mean()
    mean_iou = df["iou"].mean()

    print(f"Take Mean IoU: {take_mean_iou}")
    print(f"Object Mean IoU: {object_mean_iou}")
    print(f"Mean IoU: {mean_iou}")

    mean_mid_dist = df["mid_dist"].mean()
    mean_mid_dist_norm = df["mid_dist_norm"].mean()
    mean_mid_binning = df["mid_binning"].mean()

    print(f"mean_mid_dist: {mean_mid_dist}")
    print(f"mean_mid_dist_norm: {mean_mid_dist_norm}")
    print(f"mean_mid_binning: {mean_mid_binning}")


def process_take(take_id, input, pred, output, split, visualize=False):
    annotation_path = os.path.join(input, take_id, "annotation.json")
    with open(annotation_path, "r") as fp:
        annotation = json.load(fp)
    masks = annotation["masks"]

    ego_cam_name = "aria01_214-1"
    df_list = []

    for object_name, cams in masks.items():
        for cam_name, cam_data in cams.items():
            if not os.path.isdir(os.path.join(input, take_id, cam_name)):
                continue

            frames = list(cam_data.keys())
            if cam_name == ego_cam_name:
                count = 0
                for f_name in frames:
                    count += 1
                    if not (visualize and count % 5 == 0):
                        continue
                    f_str = f"{f_name}"

                    gt_data = masks[object_name][cam_name][f_name]
                    gt_mask = mask_utils.decode(gt_data)
                    gt_mask[gt_mask != 1] = 0

                    rgb_name = f_str + ".jpg"
                    rgb_path = os.path.join(input, take_id, cam_name, rgb_name)

                    rgb = np.array(Image.open(rgb_path))

                    if gt_mask.shape[:2] != rgb.shape[:2]:
                        gt_mask_img = Image.fromarray(gt_mask)
                        gt_mask_img = gt_mask_img.resize(
                            (rgb.shape[1], rgb.shape[0]), Image.NEAREST
                        )
                        gt_mask = np.array(gt_mask_img)

                    blended_img = blend_mask(rgb, gt_mask.astype(np.uint8), alpha=0.5)
                    output_path = os.path.join(
                        output, "gt", take_id, cam_name, object_name
                    )
                    os.makedirs(output_path, exist_ok=True)
                    save_img(blended_img, os.path.join(output_path, rgb_name))

                continue
            count = 0
            for f_name in frames:
                f_str = f"{f_name}"
                gt_data = masks[object_name][cam_name][f_name]
                gt_mask = mask_utils.decode(gt_data)
                gt_mask[gt_mask != 1] = 0

                if np.sum(gt_mask == 1) == 0:
                    continue

                pred_mask_path = os.path.join(
                    pred, take_id, cam_name, object_name, f_str + ".json"
                )
                if not os.path.isfile(pred_mask_path):
                    continue

                with open(pred_mask_path, "r") as fp:
                    pred_mask_data = json.load(fp)
                pred_mask = mask_utils.decode(pred_mask_data)
                pred_mask[pred_mask != 1] = 0

                if gt_mask.shape != pred_mask.shape:
                    gt_mask_img = Image.fromarray(gt_mask)
                    gt_mask_img = gt_mask_img.resize(
                        (pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST
                    )
                    gt_mask = np.array(gt_mask_img)

                inner = np.logical_and(pred_mask == 1, gt_mask == 1)
                outer = np.logical_or(pred_mask == 1, gt_mask == 1)
                iou = np.sum(inner) / np.sum(outer)

                mid_dist = getMidDist(gt_mask, pred_mask)
                mid_dist_norm = getMidDistNorm(gt_mask, pred_mask)
                mid_binning = getMidBinning(gt_mask, pred_mask)

                row = pd.DataFrame(
                    [
                        [
                            take_id,
                            cam_name,
                            object_name,
                            f_name,
                            iou,
                            mid_dist,
                            mid_dist_norm,
                            mid_binning,
                        ]
                    ],
                    columns=[
                        "take",
                        "camera",
                        "object",
                        "frame",
                        "iou",
                        "mid_dist",
                        "mid_dist_norm",
                        "mid_binning",
                    ],
                )
                df_list.append(row)
                count += 1

                if visualize and count % 5 == 0:
                    rgb_name = f_str + ".jpg"
                    rgb_path = os.path.join(input, take_id, cam_name, rgb_name)

                    rgb = np.array(Image.open(rgb_path))

                    blended_img = blend_mask(rgb, pred_mask.astype(np.uint8), alpha=0.5)
                    output_path = os.path.join(
                        output, split, take_id, cam_name, object_name
                    )
                    os.makedirs(output_path, exist_ok=True)
                    save_img(blended_img, os.path.join(output_path, rgb_name))

                    blended_img = blend_mask(rgb, gt_mask.astype(np.uint8), alpha=0.5)
                    output_path = os.path.join(
                        output, "gt", take_id, cam_name, object_name
                    )
                    os.makedirs(output_path, exist_ok=True)
                    save_img(blended_img, os.path.join(output_path, rgb_name))

    df = pd.concat(df_list, ignore_index=True) if len(df_list) > 0 else None
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="EgoExo take data root",
        default="../data/correspondence",
    )
    parser.add_argument(
        "--split_json",
        help="EgoExo take data root",
        default="../data/correspondence/split.json",
    )
    parser.add_argument("--split", help="EgoExo take data root", default="val")
    parser.add_argument(
        "--pred", help="EgoExo take data root", default="../output/E23_val/Annotations"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="EgoExo take data root"
    )
    parser.add_argument(
        "--compute_stats", action="store_true", help="EgoExo take data root"
    )
    parser.add_argument(
        "--output", help="Output data root", default="../output/E23_val/visualization"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    main(args)
