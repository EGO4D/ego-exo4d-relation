import os
import json
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from PIL import Image
import numpy as np
from natsort import natsorted

from dataset.range_transform import im_normalization


class EgoExoVideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """

    def __init__(
        self,
        data_root,
        vid_name,
        ref_image_dir,
        pred_image_dir,
        size=-1,
        to_save=None,
        use_all_mask=False,
    ):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        self.vid_name = vid_name
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        self.frames = to_save
        self.data_root = data_root

        tmp = pred_image_dir.split("/")
        self.take_id = tmp[0]
        self.pred_cam_name = tmp[1]
        self.object_name = "/".join(tmp[2:])
        self.ref_cam_name = ref_image_dir.split("/")[-2]

        annotation_path = os.path.join(self.data_root, self.take_id, "annotation.json")
        with open(annotation_path, "r") as fp:
            annnotation = json.load(fp)
        self.masks_data = annnotation["masks"]

        self.all_ref_keys = np.asarray(
            natsorted(self.masks_data[self.object_name][self.ref_cam_name])
        ).astype(np.int64)
        if len(self.all_ref_keys) == 0:
            return
        first_anno_key = str(self.all_ref_keys[0])

        rgb_name = f"{first_anno_key}.jpg"
        rgb_name = os.path.join(
            self.data_root, self.take_id, self.ref_cam_name, rgb_name
        )
        gt_data = self.masks_data[self.object_name][self.ref_cam_name][first_anno_key]
        this_gt = mask_utils.decode(gt_data) * 255
        self.palette = Image.fromarray(this_gt).getpalette()

        if size < 0:
            self.im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                ]
            )
        else:
            self.im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                    transforms.Resize(
                        (size, size), interpolation=InterpolationMode.BILINEAR
                    ),
                ]
            )
        self.size = size

    def get_mask_by_key(self, ref_key):
        gt_data = self.masks_data[self.object_name][self.ref_cam_name][ref_key]
        this_gt = mask_utils.decode(gt_data) * 255
        mask = Image.fromarray(this_gt).convert("P")
        mask = np.array(mask, dtype=np.uint8)
        mask[mask != 255] = 0
        return mask

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}

        tmp = frame.split("/")
        cam_name = tmp[0]
        object_name = "/".join(tmp[1:-1])
        f_name = tmp[-1]
        assert self.object_name == object_name, AssertionError("Object name mismatch")

        info["frame"] = frame
        info["take_id"] = self.take_id
        info["save"] = (self.to_save is None) or (frame in self.to_save)
        info["has_ref"] = True

        ref_key = f_name

        rgb_name = f"{f_name}.jpg"
        im_path = os.path.join(self.data_root, self.take_id, cam_name, rgb_name)
        img = Image.open(im_path).convert("RGB")
        shape = [480, 480]
        img = self.im_transform(img)

        mask = self.get_mask_by_key(ref_key)
        ref_rgb_name = f"{ref_key}.jpg"
        ref_im_path = os.path.join(
            self.data_root, self.take_id, self.ref_cam_name, ref_rgb_name
        )
        if np.sum(mask) > 0:
            ref_img = Image.open(ref_im_path).convert("RGB")
            ref_img = ref_img.resize(shape[::-1])
            ref_img = self.im_transform(ref_img)
            mask = Image.fromarray(mask).resize(shape[::-1], Image.Resampling.NEAREST)
            mask = np.array(mask)
            info["has_ref"] = np.sum(mask) > 0
        else:
            ref_img = img
            mask = np.zeros(shape, dtype=np.uint8)
            info["has_ref"] = False

        data["mask"] = mask
        info["ref_frame"] = ref_im_path
        info["shape"] = shape
        info["need_resize"] = not (self.size < 0)
        data["rgb"] = img
        data["ref_rgb"] = ref_img
        data["info"] = info

        return data

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(
            mask,
            (int(h / min_hw * self.size), int(w / min_hw * self.size)),
            mode="nearest",
        )

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)
