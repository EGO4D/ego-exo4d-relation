import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import numpy as np


def dice_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:, i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt == (i + 1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction="none").view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * (
                (self.end_warm - it) / (self.end_warm - self.start_warm)
            )
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config["start_warm"], config["end_warm"])
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.bce_logits_reduct = torch.nn.BCEWithLogitsLoss()
        self.enable_segswap = config["enable_segswap"]

    def compute(self, data, num_objects, it):
        losses = defaultdict(int)

        b, t = data["rgb"].shape[:2]
        ious = []

        losses["total_loss"] = 0
        for ti in range(1, t):
            for bi in range(b):
                loss, p = self.bce(
                    data[f"logits_{ti}"][bi : bi + 1, : num_objects[bi] + 1],
                    data["cls_gt"][bi : bi + 1, ti, 0],
                    it,
                )
                losses["p"] += p / b / (t - 1)
                losses[f"ce_loss_{ti}"] += loss / b
                pred_mask = torch.argmax(
                    data[f"logits_{ti}"][bi : bi + 1, : num_objects[bi] + 1], dim=1
                )
                inner = torch.sum(
                    torch.logical_and(pred_mask, data["cls_gt"][bi : bi + 1, ti, 0])
                )
                outer = torch.sum(
                    torch.logical_or(pred_mask, data["cls_gt"][bi : bi + 1, ti, 0])
                )
                iou = inner / outer if outer > 1e-5 else 0.0
                if outer <= 1e-5:
                    continue
                ious.append(iou.cpu())

            losses["total_loss"] += losses["ce_loss_%d" % ti]
            losses[f"dice_loss_{ti}"] = dice_loss(
                data[f"masks_{ti}"], data["cls_gt"][:, ti, 0]
            )
            losses["total_loss"] += losses[f"dice_loss_{ti}"]

        if self.enable_segswap:
            loss_ego = self.bce_logits(data["segswap_mx"], data["ego_cls_gt"].float())
            loss_exo = self.bce_logits(data["segswap_my"], data["cls_gt"].float())
            weights = torch.ones_like(data["cls_gt"])
            weights = torch.where(data["cls_gt"] == 1, weights * 10, weights)
            losses["segswap_bce"] = torch.mean(weights * (loss_exo + loss_ego))
            losses["total_loss"] += losses["segswap_bce"]

            loss_ego = dice_loss(
                torch.flatten(data["segswap_mx"], 0, 1),
                torch.flatten(data["ego_cls_gt"], 0, 1)[:, 0].float(),
            )
            loss_exo = dice_loss(
                torch.flatten(data["segswap_my"], 0, 1),
                torch.flatten(data["cls_gt"], 0, 1)[:, 0].float(),
            )
            losses["segswap_dice"] = loss_exo + loss_ego
            losses["total_loss"] += losses["segswap_dice"]

        return losses, np.sum(ious) / len(ious) if len(ious) > 0 else None
