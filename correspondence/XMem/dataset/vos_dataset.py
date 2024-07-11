import os
from os import path, replace
import json
from pycocotools import mask as mask_utils

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from natsort import natsorted
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(
        self,
        egoexo_root,
        ego_cam_name,
        max_jump,
        is_bl,
        subset=None,
        num_frames=3,
        max_num_obj=1,
        finetune=False,
        augmentation=False,
        swap=False,
    ):
        self.egoexo_root = os.path.join(egoexo_root, "train")
        self.frame_folder = "rgb"
        self.mask_file = "annotation.json"
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj
        self.augmentation = augmentation
        self.swap = swap

        self.videos = []
        self.frames = {}

        self.takes = sorted(os.listdir(self.egoexo_root))

        for take_id in self.takes:
            annotation_path = os.path.join(self.egoexo_root, take_id, "annotation.json")
            if not os.path.exists(annotation_path):
                continue
            with open(annotation_path, "r") as fp:
                annotation = json.load(fp)
            masks = annotation["masks"]

            for object_name, cams in masks.items():
                ego_cams = [x for x in masks[object_name].keys() if "aria" in x]
                if len(ego_cams) < 1:
                    continue
                ego_cam_name = ego_cams[0]
                ego_frames = list(cams[ego_cam_name].keys())
                for cam_name, cam_data in cams.items():
                    if not os.path.isdir(
                        os.path.join(self.egoexo_root, take_id, cam_name)
                    ):
                        continue
                    exo_frames = list(cam_data.keys())
                    if cam_name == ego_cam_name:
                        continue

                    frames = np.intersect1d(ego_frames, exo_frames)
                    if len(frames) < num_frames:
                        continue

                    vid = path.join(
                        take_id, ego_cam_name, cam_name, object_name.replace("/", "-")
                    )
                    self.frames[vid] = [None] * len(frames)
                    for i, f in enumerate(frames):
                        self.frames[vid][i] = path.join(cam_name, object_name, f)
                    self.videos.append(vid)

        print(
            "%d out of %d videos accepted in %s."
            % (len(self.videos), len(self.takes), egoexo_root)
        )

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.01, 0.01, 0.01, 0),
            ]
        )

        self.pair_im_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=0 if finetune or self.is_bl else 15,
                    shear=0 if finetune or self.is_bl else 10,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=im_mean,
                ),
            ]
        )

        self.pair_gt_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=0 if finetune or self.is_bl else 15,
                    shear=0 if finetune or self.is_bl else 10,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ),
            ]
        )

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.03, 0.03, 0),
                transforms.RandomGrayscale(0.05),
            ]
        )

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (480, 480),
                        scale=(0.25, 1.00),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                ]
            )

            self.all_gt_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (480, 480),
                        scale=(0.25, 1.00),
                        interpolation=InterpolationMode.NEAREST,
                    ),
                ]
            )
        else:
            self.all_im_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (480, 480),
                        scale=(0.36, 1.00),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                ]
            )

            self.all_gt_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (480, 480),
                        scale=(0.36, 1.00),
                        interpolation=InterpolationMode.NEAREST,
                    ),
                ]
            )

        # Final transform without randomness
        self.final_gt_transform = transforms.Compose(
            [
                transforms.Resize((480, 480), interpolation=InterpolationMode.NEAREST),
            ]
        )

        self.final_im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((480, 480)),
                im_normalization,
            ]
        )

    def get_images(self, frames_idx, take_root, frames, cam_name):
        info_frames = []
        images = []
        masks = []
        sequence_seed = np.random.randint(2147483647)
        for f_idx in frames_idx:
            components = frames[f_idx].split("/")
            object_name = "/".join(components[1:-1])
            f_name = components[-1]
            rgb_name = f"{f_name}.jpg"
            # rgb_path = os.path.join(self.egoexo_root, take_id, cam_name, rgb_name)
            rgb_path = os.path.join(take_root, cam_name, rgb_name)

            annotation_path = os.path.join(take_root, self.mask_file)
            with open(annotation_path, "r") as fp:
                annotation = json.load(fp)
            masks_data = annotation["masks"]

            gt_data = masks_data[object_name][cam_name][f_name]

            info_frames.append(rgb_path)

            this_im = Image.open(rgb_path).convert("RGB")
            if self.augmentation:
                reseed(sequence_seed)
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
            this_gt = mask_utils.decode(gt_data) * 255
            this_gt = Image.fromarray(this_gt)
            if self.augmentation:
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)
            else:
                this_gt = self.final_gt_transform(this_gt)
            this_im = self.final_im_transform(this_im)
            this_gt = np.array(this_gt)
            this_gt[this_gt != 255] = 0

            images.append(this_im)
            masks.append(this_gt)
        return images, masks, info_frames

    def one_hot_gt(self, masks, target_objects):
        cls_gt = np.zeros((self.num_frames, 480, 480), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 480, 480), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = masks == l
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = this_mask[0]
        cls_gt = np.expand_dims(cls_gt, 1)
        return cls_gt, first_frame_gt

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info["name"] = video

        take_id, ego_cam_name, cam_name, object_name = video.split("/")[-4:]

        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info["frames"] = []  # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(
                range(
                    max(0, frames_idx[-1] - this_max_jump),
                    min(length, frames_idx[-1] + this_max_jump + 1),
                )
            ).difference(set(frames_idx))
            while len(frames_idx) < num_frames:
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(
                    range(
                        max(0, frames_idx[-1] - this_max_jump),
                        min(length, frames_idx[-1] + this_max_jump + 1),
                    )
                )
                acceptable_set = acceptable_set.union(new_set).difference(
                    set(frames_idx)
                )

            frames_idx = natsorted(frames_idx)

            images, masks, info_frames = self.get_images(
                frames_idx, os.path.join(self.egoexo_root, take_id), frames, cam_name
            )
            ego_images, ego_masks, _ = self.get_images(
                frames_idx,
                os.path.join(self.egoexo_root, take_id),
                frames,
                ego_cam_name,
            )
            info["frames"] = info_frames

            images = torch.stack(images, 0)
            ego_images = torch.stack(ego_images, 0)

            labels = np.unique(ego_masks[0])
            # Remove background
            labels = labels[labels != 0]

            target_objects = []
            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (ego_masks[0] == l).sum()
                    if pixel_sum > 10 * 10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30 * 30:
                            good_lables.append(l)
                        elif (
                            max((ego_masks[1] == l).sum(), (ego_masks[2] == l).sum())
                            < 20 * 20
                        ):
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)

            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(
                target_objects, size=self.max_num_obj, replace=False
            )

        info["num_objects"] = max(1, len(target_objects))

        masks = np.stack(masks, 0)
        ego_masks = np.stack(ego_masks, 0)
        # Generate one-hot ground-truth
        cls_gt, first_frame_gt = self.one_hot_gt(masks, target_objects)
        ego_cls_gt, ego_first_frame_gt = self.one_hot_gt(ego_masks, target_objects)

        # 1 if object exist, 0 otherwise
        selector = [
            1 if i < info["num_objects"] else 0 for i in range(self.max_num_obj)
        ]
        selector = torch.FloatTensor(selector)

        if not self.swap:
            data = {
                "rgb": images,
                "first_frame_gt": first_frame_gt,
                "cls_gt": cls_gt,
                "ego_rgb": ego_images,
                "ego_first_frame_gt": ego_first_frame_gt,
                "ego_cls_gt": ego_cls_gt,
                "selector": selector,
                "info": info,
            }
        else:
            data = {
                "ego_rgb": images,
                "ego_first_frame_gt": first_frame_gt,
                "ego_cls_gt": cls_gt,
                "rgb": ego_images,
                "first_frame_gt": ego_first_frame_gt,
                "cls_gt": ego_cls_gt,
                "selector": selector,
                "info": info,
            }

        return data

    def __len__(self):
        return len(self.videos)
