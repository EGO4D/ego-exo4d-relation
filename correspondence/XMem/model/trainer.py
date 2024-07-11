"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import XMem
from model.losses import LossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs


class XMemTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config["num_frames"]
        self.num_ref_frames = config["num_ref_frames"]
        self.deep_update_prob = config["deep_update_prob"]
        self.enable_segswap = config["enable_segswap"]
        self.local_rank = local_rank

        self.XMem = nn.parallel.DistributedDataParallel(
            XMem(config, enable_segswap=self.enable_segswap).cuda(),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string(
                "model_size",
                str(sum([param.nelement() for param in self.XMem.parameters()])),
            )
        self.train_integrator = Integrator(
            self.logger, distributed=True, local_rank=local_rank, world_size=world_size
        )
        self.loss_computer = LossComputer(config)

        self.train()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.XMem.parameters()),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, config["steps"], config["gamma"]
        )
        if config["amp"]:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config["log_text_interval"]
        self.log_image_interval = config["log_image_interval"]
        self.save_network_interval = config["save_network_interval"]
        self.save_checkpoint_interval = config["save_checkpoint_interval"]
        if config["debug"]:
            self.log_text_interval = self.log_image_interval = 1

    def do_pass(self, data, it=0, do_logging=False):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        frames = data["rgb"].float()
        ego_frames = data["ego_rgb"].float()
        cls_gt = data["cls_gt"].float()
        ego_cls_gt = data["ego_cls_gt"].float()
        first_frame_gt = data["first_frame_gt"].float()
        ego_first_frame_gt = data["ego_first_frame_gt"].float()
        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data["info"]["num_objects"]]
        num_objects = first_frame_gt.shape[2]
        selector = data["selector"].unsqueeze(2).unsqueeze(2)
        iou_mean = 0
        iou_count = 0
        if torch.sum(ego_first_frame_gt) < 5 and torch.sum(first_frame_gt) < 5:
            return

        with torch.cuda.amp.autocast(enabled=self.config["amp"]):
            # image features never change, compute once
            encoded_results, mx, my = self.XMem(
                "encode_key", ego_frames, ego_cls_gt, frames
            )

            key = encoded_results[0]["key"]
            shrinkage = encoded_results[0]["shrinkage"]
            selection = encoded_results[0]["selection"]
            f16 = encoded_results[0]["f16"]
            f8 = encoded_results[0]["f8"]
            f4 = encoded_results[0]["f4"]

            ego_key = encoded_results[1]["key"]
            ego_shrinkage = encoded_results[1]["shrinkage"]
            ego_selection = encoded_results[1]["selection"]
            ego_f16 = encoded_results[1]["f16"]
            ego_f8 = encoded_results[1]["f8"]
            ego_f4 = encoded_results[1]["f4"]

            filler_one = torch.zeros(1, dtype=torch.int64)
            hidden = torch.zeros(
                (b, num_objects, self.config["hidden_dim"], *key.shape[-2:])
            )
            v16, hidden = self.XMem(
                "encode_value",
                ego_frames[:, 0],
                ego_f16[:, 0],
                hidden,
                ego_cls_gt[:, 0],
            )
            values = v16.unsqueeze(3)  # add the time dimension
            v16, hidden = self.XMem(
                "encode_value",
                frames[:, 0],
                f16[:, 0],
                hidden,
                first_frame_gt[:, 0],
            )
            values = torch.cat([values, v16.unsqueeze(3)], 3)
            if self.enable_segswap:
                out["segswap_mx"] = mx
                out["segswap_my"] = my
            ref_keys = torch.cat([ego_key[:, :, 0], key[:, :, 0]], 3)
            ref_shrinkage = (
                torch.cat([ego_shrinkage[:, :, 0], shrinkage[:, :, 0]], 3)
                if shrinkage is not None
                else None
            )
            for ti in range(1, self.num_frames):
                v16, hidden = self.XMem(
                    "encode_value",
                    ego_frames[:, ti],
                    ego_f16[:, ti],
                    hidden,
                    ego_cls_gt[:, ti],
                )
                values = torch.cat([values, v16.unsqueeze(3)], 3)
                ref_keys = torch.cat([ref_keys, ego_key[:, :, ti]], 3)
                ref_shrinkage = (
                    torch.cat([ref_shrinkage, ego_shrinkage[:, :, ti]], 3)
                    if shrinkage is not None
                    else None
                )

                ref_values = values
                # Segment frame ti
                memory_readout = self.XMem(
                    "read_memory",
                    key[:, :, ti],
                    selection[:, :, ti] if selection is not None else None,
                    ref_keys,
                    ref_shrinkage,
                    ref_values,
                )
                hidden, logits, masks = self.XMem(
                    "segment",
                    (f16[:, ti], f8[:, ti], f4[:, ti]),
                    memory_readout,
                    hidden,
                    selector,
                    h_out=(ti < (self.num_frames - 1)),
                )

                # No need to encode the last frame
                if ti < (self.num_frames - 1):
                    is_deep_update = np.random.rand() < self.deep_update_prob
                    v16, hidden = self.XMem(
                        "encode_value",
                        frames[:, ti],
                        f16[:, ti],
                        hidden,
                        masks,
                        is_deep_update=is_deep_update,
                    )
                    values = torch.cat([values, v16.unsqueeze(3)], 3)
                    ref_keys = torch.cat([ref_keys, key[:, :, ti]], 3)
                    ref_shrinkage = (
                        torch.cat([ref_shrinkage, shrinkage[:, :, ti]], 3)
                        if shrinkage is not None
                        else None
                    )

                out[f"masks_{ti}"] = masks
                out[f"logits_{ti}"] = logits

            if self._do_log or self._is_train:
                losses, iou = self.loss_computer.compute(
                    {**data, **out}, num_filled_objects, it
                )
                if iou is not None:
                    iou_mean = (iou_mean * iou_count + iou) / (iou_count + 1)
                    iou_count += 1

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if iou_count > 0:
                        self.integrator.add_dict({"iou": iou_mean})
                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2(
                                    "train/pairs",
                                    pool_pairs(images, size, num_filled_objects),
                                    it,
                                )

            if self._is_train:
                if do_logging and it != 0:
                    # if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar(
                            "train/lr", self.scheduler.get_last_lr()[0], it
                        )
                        self.logger.log_metrics(
                            "train",
                            "time",
                            (time.time() - self.last_time) / self.log_text_interval,
                            it,
                        )
                    self.last_time = time.time()
                    self.train_integrator.finalize("train", it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config["amp"]:
            self.scaler.scale(losses["total_loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses["total_loss"].backward()
            self.optimizer.step()

        self.scheduler.step()

    def save_network(self, it):
        if self.save_path is None:
            print("Saving has been disabled.")
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f"{self.save_path}_{it}.pth"
        torch.save(self.XMem.module.state_dict(), model_path)
        print(f"Network saved to {model_path}.")

    def save_checkpoint(self, it):
        if self.save_path is None:
            print("Saving has been disabled.")
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f"{self.save_path}_checkpoint_{it}.pth"
        checkpoint = {
            "it": it,
            "network": self.XMem.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}.")

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = "cuda:%d" % self.local_rank
        checkpoint = torch.load(path, map_location={"cuda:0": map_location})

        it = checkpoint["it"]
        network = checkpoint["network"]
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]

        map_location = "cuda:%d" % self.local_rank
        self.XMem.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print("Network weights, optimizer states, and scheduler states loaded.")

        return it

    def load_network_in_memory(self, src_dict):
        self.XMem.module.load_weights(src_dict)
        print("Network weight loaded from memory.")

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = "cuda:%d" % self.local_rank
        src_dict = torch.load(path, map_location={"cuda:0": map_location})

        self.load_network_in_memory(src_dict)
        print(f"Network weight loaded from {path}")

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.XMem.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.XMem.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.XMem.eval()
        return self
