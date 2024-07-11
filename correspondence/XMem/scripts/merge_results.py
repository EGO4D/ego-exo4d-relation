import os
import json
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import argparse


def check_pred_format(input_dir, pred_dir, split, swap):
    from copy import deepcopy

    input_dir = os.path.join(input_dir, split)
    videos = os.listdir(input_dir)
    task_name = "ego-exo" if not swap else "exo-ego"
    annotations = {task_name: {"version": "xx", "challenge": "xx", "results": {}}}
    for vid in tqdm(videos):
        if not os.path.isdir(f"{pred_dir}/{vid}"):
            continue
        with open(f"{pred_dir}/{vid}/annotations.json", "r") as fp:
            vid_anno = json.load(fp)

        correct_anno = deepcopy(vid_anno)
        for obj in vid_anno["masks"]:
            correct_anno["masks"][obj] = {}
            for cam in vid_anno["masks"][obj]:
                for frame_idx in vid_anno["masks"][obj][cam]:
                    mask_dict = vid_anno["masks"][obj][cam][frame_idx]
                    if mask_dict:
                        mask = mask_utils.decode(mask_dict)
                    else:
                        mask = np.zeros((480, 480)).astype(np.uint8)
                    encoded_mask = mask_utils.encode(np.asfortranarray(mask))
                    encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
                    vid_anno["masks"][obj][cam][frame_idx] = {
                        "pred_mask": encoded_mask,
                        "confidence": int(
                            mask.sum() > (0.001 * mask.shape[0] * mask.shape[1])
                        ),
                    }

                correct_anno["masks"][obj][cam.replace("__", "_")] = vid_anno["masks"][
                    obj
                ][cam]

        annotations[task_name]["results"][vid] = correct_anno

    output_dir = os.path.dirname(pred_dir)
    with open(f"{output_dir}/final_results.json", "w+") as fp:
        json.dump(annotations, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="The input path",
        required=True,
    )
    parser.add_argument(
        "--split",
        help="The split name",
        required=True,
    )
    parser.add_argument(
        "--pred_dir",
        help="The predicted path",
        required=True,
    )
    parser.add_argument("--swap", action="store_true", default=False)
    args = parser.parse_args()
    check_pred_format(args.input_dir, args.pred_dir, args.split, args.swap)
