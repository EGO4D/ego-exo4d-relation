import os
from os import path
import json
import numpy as np
from natsort import natsorted
from inference.data.video_reader import EgoExoVideoReader


class EgoExoTestDataset:
    def __init__(
        self,
        data_root,
        split,
        size=480,
        num_frames=8,
        swap=False,
    ):
        self.data_root = os.path.join(data_root, split)
        self.req_frame_list = {}
        self.vid_list = []
        self.takes = natsorted(os.listdir(self.data_root))

        for take_id in self.takes:
            annotation_path = os.path.join(self.data_root, take_id, "annotation.json")
            with open(annotation_path, "r") as fp:
                annotation = json.load(fp)
            masks = annotation["masks"]
            subsample_idx = annotation["subsample_idx"]

            cams = os.listdir(os.path.join(self.data_root, take_id))
            cams = [c for c in cams if os.path.isdir(os.path.join(self.data_root, take_id, c))]
            ego_cam_name = None
            for cam_name in cams:
                if "aria" in cam_name:
                    ego_cam_name = cam_name
            if ego_cam_name is None:
                continue

            for object_name, annotated_cams in masks.items():
                exo_cam_names = [c for c in cams if c != ego_cam_name]
                for cam_name in exo_cam_names:
                    ego_frames = natsorted(
                        os.listdir(path.join(self.data_root, take_id, ego_cam_name))
                    )
                    ego_frames = [int(f.split(".")[0]) for f in ego_frames]
                    exo_frames = natsorted(
                        os.listdir(path.join(self.data_root, take_id, cam_name))
                    )
                    exo_frames = [int(f.split(".")[0]) for f in exo_frames]
                    if swap:
                        if annotated_cams.get(cam_name) is None:
                            continue
                        exo_frames = list(annotated_cams[cam_name].keys())
                        frames = np.intersect1d(ego_frames, exo_frames)
                        vid = path.join(take_id, cam_name, ego_cam_name, object_name)
                        self.req_frame_list[vid] = [None] * len(frames)
                        for i, f in enumerate(frames):
                            self.req_frame_list[vid][i] = path.join(
                                ego_cam_name, object_name, str(f)
                            )
                    else:
                        if annotated_cams.get(ego_cam_name) is None:
                            continue
                        ego_frames = list(annotated_cams[ego_cam_name].keys())
                        frames = np.intersect1d(ego_frames, exo_frames)
                        vid = path.join(take_id, ego_cam_name, cam_name, object_name)
                        self.req_frame_list[vid] = [None] * len(frames)
                        for i, f in enumerate(frames):
                            self.req_frame_list[vid][i] = path.join(
                                cam_name, object_name, str(f)
                            )
                    self.vid_list.append(vid)
        self.size = size

    def get_datasets(self):
        for video in self.vid_list:
            tmp = video.split("/")
            take = tmp[0]
            ref_cam_name = tmp[1]
            pred_cam_name = tmp[2]
            obj = "/".join(tmp[3:])
            yield EgoExoVideoReader(
                self.data_root,
                path.join(take, pred_cam_name, obj),
                path.join(take, ref_cam_name, obj),
                path.join(take, pred_cam_name, obj),
                size=self.size,
                to_save=self.req_frame_list[video],
                use_all_mask=False,
            )

    def __len__(self):
        return len(self.vid_list)
