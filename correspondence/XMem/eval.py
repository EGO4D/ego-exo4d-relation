import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import json
from inference.data.test_datasets import (
    EgoExoTestDataset,
)
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from pycocotools import mask as mask_utils

from progressbar import progressbar

try:
    import hickle as hkl
except ImportError:
    print("Failed to import hickle. Fine if not using multi-scale testing.")


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument("--model", default="./saves/XMem.pth")
parser.add_argument("--swap", action="store_true", default=False)
parser.add_argument("--enable_segswap", action="store_true", default=False)

# Data options
parser.add_argument("--e23_path", default="dataset/val")
# For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
parser.add_argument("--generic_path")

parser.add_argument("--dataset", help="E23", default="E23")
parser.add_argument("--split", help="val/test", default="val")
parser.add_argument("--output", default=None)
parser.add_argument(
    "--save_all",
    action="store_true",
    help="Save all frames. Useful only in YouTubeVOS/long-time video",
)

parser.add_argument(
    "--benchmark",
    action="store_true",
    help="enable to disable amp for FPS benchmarking",
)

# Long-term memory options
parser.add_argument("--disable_long_term", action="store_true")
parser.add_argument(
    "--max_mid_term_frames",
    help="T_max in paper, decrease to save memory",
    type=int,
    default=10,
)
parser.add_argument(
    "--min_mid_term_frames",
    help="T_min in paper, decrease to save memory",
    type=int,
    default=5,
)
parser.add_argument(
    "--max_long_term_elements",
    help="LT_max in paper, increase if objects disappear for a long time",
    type=int,
    default=20000,
)
parser.add_argument("--num_prototypes", help="P in paper", type=int, default=128)

parser.add_argument("--top_k", type=int, default=20)
parser.add_argument(
    "--mem_every",
    help="r in paper. Increase to improve running speed.",
    type=int,
    default=5,
)
parser.add_argument(
    "--deep_update_every",
    help="Leave -1 normally to synchronize with mem_every",
    type=int,
    default=-1,
)

# Multi-scale options
parser.add_argument("--save_scores", action="store_true")
parser.add_argument("--flip", action="store_true")
parser.add_argument(
    "--size",
    default=480,
    type=int,
    help="Resize the shorter side to this size. -1 to use original resolution. ",
)

args = parser.parse_args()
config = vars(args)
config["enable_long_term"] = not config["disable_long_term"]

if args.output is None:
    args.output = f"../output/{args.dataset}_{args.split}"
    print(f"Output path not provided. Defaulting to {args.output}")

"""
Data preparation
"""
is_egoexo = args.dataset.startswith("E")
out_path = args.output

if is_egoexo:
    if args.dataset == "E23":
        egoexo_path = args.e23_path

        if args.split == "val":
            meta_dataset = EgoExoTestDataset(
                data_root=egoexo_path,
                split="val",
                size=args.size,
                num_frames=5,
                swap=args.swap,
            )
        elif args.split == "test":
            meta_dataset = EgoExoTestDataset(
                data_root=egoexo_path,
                split="test",
                size=args.size,
                num_frames=5,
                swap=args.swap,
            )
        else:
            raise NotImplementedError

torch.autograd.set_grad_enabled(False)

# Set up loader
meta_loader = meta_dataset.get_datasets()

# Load our checkpoint
config["segswap_model"] = args.model
config["eval"] = True
network = XMem(config, args.model, enable_segswap=args.enable_segswap).cuda().eval()
if args.model is not None:
    model_weights = torch.load(args.model)
    network.load_weights(model_weights, init_as_zero_if_needed=True)
else:
    print("No model loaded.")

total_process_time = 0
total_frames = 0

# Start eval
for vid_reader in progressbar(
    meta_loader, max_value=len(meta_dataset), redirect_stdout=False
):
    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=0)
    vid_name = vid_reader.vid_name
    vid_length = len(loader)
    config["enable_long_term_count_usage"] = True
    mapper = MaskMapper()
    processor = InferenceCore(network, config=config)
    first_mask_loaded = False
    if len(vid_reader.all_ref_keys) == 0:
        continue

    for ti, data in enumerate(loader):
        with torch.cuda.amp.autocast(enabled=not args.benchmark):
            object_name = vid_reader.object_name
            take_id = vid_reader.take_id
            ref_cam_name = vid_reader.ref_cam_name
            pred_cam_name = vid_reader.pred_cam_name

            pred_folder_name = path.join(
                take_id,
                "__".join(
                    [ref_cam_name, pred_cam_name],
                ),
                object_name,
            )
            coco_out_path = path.join(out_path, "coco", pred_folder_name)
            info = data["info"]
            frame = info["frame"][0]
            tmp = frame.split("/")
            f_name = tmp[-1]
            rgb_name = f"{f_name}.jpg"

            rgb = data["rgb"].cuda()[0]
            ref_rgb = data["ref_rgb"].cuda()[0]
            msk = data["mask"]
            info = data["info"]
            frame = info["frame"][0]
            ref_frame = info["ref_frame"][0]
            take_id = info["take_id"][0]
            shape = info["shape"]
            need_resize = info["need_resize"][0]
            has_ref = info["has_ref"][0]

            """
            For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            Seems to be very similar in testing as my previous timing method 
            with two cuda sync + time.time() in STCN though 
            """
            if not has_ref:
                continue
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if args.flip:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            # Map possibly non-continuous labels to continuous ones
            msk, labels = mapper.convert_mask(msk[0].numpy(), True)
            msk = torch.Tensor(msk).cuda()
            if need_resize:
                msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
            processor.set_all_labels(list(mapper.remappings.values()))

            # Run the model on this frame
            prob, out_cls = processor.step(
                rgb, ref_rgb, msk, labels, end=(ti == vid_length - 1)
            )

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(
                    prob.unsqueeze(1), shape, mode="bilinear", align_corners=False
                )[:, 0]

            end.record()
            torch.cuda.synchronize()
            total_process_time += start.elapsed_time(end) / 1000
            total_frames += 1

            if args.flip:
                prob = torch.flip(prob, dims=[-1])

            # Probability mask -> index mask
            out_mask = torch.max(prob, dim=0).indices
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            if args.save_scores:
                prob = (prob.detach().cpu().numpy() * 255).astype(np.uint8)

            # Save the mask
            if args.save_all or info["save"][0]:
                out_mask = mapper.remap_index_mask(out_mask)

                tmp = frame.split("/")
                f_name = tmp[-1]
                rgb_name = f"{f_name}.jpg"

                out_img_coco = mask_utils.encode(
                    np.asfortranarray((out_mask // 255).astype(np.uint8))
                )
                out_img_coco["counts"] = out_img_coco["counts"].decode("utf-8")
                coco_out_path = path.join(out_path, "coco", pred_folder_name)
                os.makedirs(coco_out_path, exist_ok=True)
                with open(
                    path.join(coco_out_path, rgb_name[:-4] + ".json"), "w+"
                ) as fp:
                    json.dump(out_img_coco, fp)

print(f"Total processing time: {total_process_time}")
print(f"Total processed frames: {total_frames}")
print(f"FPS: {total_frames / total_process_time}")
print(f"Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}")
