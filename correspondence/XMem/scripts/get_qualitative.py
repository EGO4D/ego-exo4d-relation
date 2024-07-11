import argparse
import json
import tqdm
import cv2
import os
import numpy as np
from pycocotools import mask as mask_utils
import random

EVALMODE = "test"


def blend_mask(input_img, binary_mask, alpha=0.5):
    if input_img.ndim == 2:
        return input_img
    mask_image = np.zeros(input_img.shape, np.uint8)
    mask_image[:, :, 1] = 255
    mask_image = mask_image * np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    blend_image = input_img[:, :, :].copy()
    pos_idx = binary_mask > 0
    for ind in range(input_img.ndim):
        ch_img1 = input_img[:, :, ind]
        ch_img2 = mask_image[:, :, ind]
        ch_img3 = blend_image[:, :, ind]
        ch_img3[pos_idx] = alpha * ch_img1[pos_idx] + (1 - alpha) * ch_img2[pos_idx]
        blend_image[:, :, ind] = ch_img3
    return blend_image


def upsample_mask(mask, frame):
    H, W = frame.shape[:2]
    mH, mW = mask.shape[:2]

    if W > H:
        ratio = mW / W
        h = H * ratio
        diff = int((mH - h) // 2)
        if diff == 0:
            mask = mask
        else:
            mask = mask[diff:-diff]
    else:
        ratio = mH / H
        w = W * ratio
        diff = int((mW - w) // 2)
        if diff == 0:
            mask = mask
        else:
            mask = mask[:, diff:-diff]

    mask = cv2.resize(mask, (W, H))
    return mask


def downsample(mask, frame):
    H, W = frame.shape[:2]
    mH, mW = mask.shape[:2]

    mask = cv2.resize(mask, (W, H))
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        help="the correspondence dataset path",
        required=True,
    )
    parser.add_argument(
        "--inference_path",
        help="The predicted path",
        required=True,
    )
    parser.add_argument(
        "--out_path", help="Output path to save the predictions", required=True
    )
    parser.add_argument(
        "--show_gt",
        help="if true, visualize anotations instead of predictions",
        action="store_true",
    )

    args = parser.parse_args()
    test_ids = os.listdir(args.datapath)

    random.seed(0)
    random.shuffle(test_ids)

    for take_id in tqdm.tqdm(test_ids[:-1]):
        print(f"Processing take {take_id}")
        # Load the GT annotations
        gt_file = f"{args.datapath}/{take_id}/annotation.json"
        if not os.path.isfile(gt_file):
            continue
        with open(gt_file, "r") as f:
            gt = json.load(f)
        # Load the predictions
        pred_file = f"{args.inference_path}/{take_id}/annotations.json"
        with open(pred_file, "r") as f:
            pred = json.load(f)

        for obj in list(pred["masks"].keys()):
            cams = list(pred["masks"][obj].keys())
            for CAM in cams:
                query_cam = CAM.split("__")[0]
                target_cam = CAM.split("__")[1]

                for frame_idx in gt["masks"][obj][query_cam].keys():
                    frame = cv2.imread(
                        f"{args.datapath}/{take_id}/{target_cam}/{frame_idx}.jpg"
                    )
                    if pred["masks"][obj][CAM].get(frame_idx) is None:
                        continue
                    mask = mask_utils.decode(pred["masks"][obj][CAM][frame_idx])
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    # breakpoint()
                    try:
                        mask = upsample_mask(mask, frame)
                        out = blend_mask(frame, mask)
                    except:
                        breakpoint()

                    os.makedirs(
                        f"{args.out_path}/{take_id}_{target_cam}_{obj}", exist_ok=True
                    )
                    cv2.imwrite(
                        f"{args.out_path}/{take_id}_{target_cam}_{obj}/{frame_idx}.jpg",
                        out,
                    )

                    # gt
                    if args.show_gt:
                        # target gt
                        if frame_idx in gt["masks"][obj][target_cam]:
                            gt_mask = mask_utils.decode(
                                gt["masks"][obj][target_cam][frame_idx]
                            )
                        else:
                            gt_mask = np.zeros_like(frame)[..., 0]

                        # gt_mask = upsample_mask(gt_mask, frame)
                        gt_mask = downsample(gt_mask, frame)
                        out = blend_mask(frame, gt_mask)

                        os.makedirs(
                            f"{args.out_path}/{take_id}_{target_cam}_{obj}_gt",
                            exist_ok=True,
                        )
                        cv2.imwrite(
                            f"{args.out_path}/{take_id}_{target_cam}_{obj}_gt/{frame_idx}.jpg",
                            out,
                        )

                        # query gt
                        frame = cv2.imread(
                            f"{args.datapath}/{take_id}/{query_cam}/{frame_idx}.jpg"
                        )
                        if frame_idx in gt["masks"][obj][query_cam]:
                            gt_mask = mask_utils.decode(
                                gt["masks"][obj][query_cam][frame_idx]
                            )
                        else:
                            gt_mask = np.zeros_like(frame)[..., 0]

                        # gt_mask = upsample_mask(gt_mask, frame)
                        gt_mask = downsample(gt_mask, frame)
                        out = blend_mask(frame, gt_mask)

                        os.makedirs(
                            f"{args.out_path}/{take_id}_{target_cam}_{obj}_gt_query",
                            exist_ok=True,
                        )
                        cv2.imwrite(
                            f"{args.out_path}/{take_id}_{target_cam}_{obj}_gt_query/{frame_idx}.jpg",
                            out,
                        )
