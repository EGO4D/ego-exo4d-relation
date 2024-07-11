import json
import argparse
from pycocotools import mask as mask_utils
import numpy as np
import cv2
import tqdm

from sklearn.metrics import balanced_accuracy_score

import metrics

EVALMODE = "test"  # it can also be test

# Threshold for the Iou Matching
IOUTHRES = 0.1
CONF_THRESH = 0.5


def reshape_img_war(img, size=(480, 480)):
    C = 1
    if len(img.shape) == 2:
        H, W = img.shape
        img = img[..., None]
    else:
        H, W, C = img.shape
    temp = np.zeros((max(H, W), max(H, W), C), dtype=np.uint8)
    if H > W:
        L = (H - W) // 2
        temp[:, L:-L] = img
    elif W > H:
        L = (W - H) // 2
        temp[L:-L] = img
    else:
        temp = img
    temp = cv2.resize(temp, size, interpolation=cv2.INTER_NEAREST)
    return temp


def reshape_img_nopad(img, max_dim=480):
    H, W = img.shape[:2]
    if H > W:
        ratio = 1.0 / H * max_dim
    else:
        ratio = 1.0 / W * max_dim
    newH, newW = int(H * ratio), int(W * ratio)
    img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST)
    return img


def reshape_img_nopad_square(img, max_dim=480):
    img = cv2.resize(img, (max_dim, max_dim), interpolation=cv2.INTER_NEAREST)
    return img


def remove_pad(img, orig_size):
    cur_H, cur_W = img.shape[:2]
    orig_H, orig_W = orig_size
    if orig_W > orig_H:
        ratio = 1.0 / orig_W * cur_W
    else:
        ratio = 1.0 / orig_H * cur_H
    new_H, new_W = int(orig_H * ratio), int(orig_W * ratio)
    if new_W > new_H:
        diff_H = (cur_H - new_H) // 2
        img = img[diff_H:-diff_H]
    else:
        diff_W = (cur_W - new_W) // 2
        img = img[:, diff_W:-diff_W]
    return img


def getIoU(gt_mask, pred_mask):
    gt_mask = mask_utils.decode(gt_mask)
    gt_mask = reshape_img_war(gt_mask)
    pred_mask = mask_utils.decode(pred_mask)
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union


def eval_mask(gt_masks: np.ndarray, fake_masks: np.ndarray):
    """TODO: Docstring for eval_mask.

    Args:
        gt_masks (np.ndarray): The
        fake_masks (np.ndarray): TODO

    Returns: TODO

    """
    iou = metrics.db_eval_iou(gt_masks, fake_masks)
    boundary = metrics.db_eval_boundary(gt_masks, fake_masks)
    return iou, boundary


def existence_accuracy(gt_mask: np.ndarray, pred_mask: np.ndarray):
    gt_zeros = np.all(gt_mask == 0)
    pred_zeros = np.all(pred_mask == 0)

    return gt_zeros == pred_zeros


def processGTPred_EGOEXO(datapath, take_id, take_annotation, gt, pred, object_ids):
    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []
    MDistNorm = []
    MDistBinning = []

    ObjExist_GT = []
    ObjExist_Pred = []

    ObjSizeGT = []
    ObjSizePred = []
    IMSize = []

    H, W = 480, 480  # resolution for evalution

    for object_id in object_ids:
        ego_cams = [x for x in gt["masks"][object_id].keys() if "aria" in x]
        if len(ego_cams) < 1:
            continue
        assert len(ego_cams) == 1
        EGOCAM = ego_cams[0]

        EXOCAMS = [x for x in gt["masks"][object_id].keys() if "aria" not in x]
        for exo_cam in EXOCAMS:
            gt_masks_ego = {}
            gt_masks_exo = {}
            pred_masks_exo = {}

            if EGOCAM in gt["masks"][object_id].keys():
                gt_masks_ego = gt["masks"][object_id][EGOCAM]
            if exo_cam in gt["masks"][object_id].keys():
                gt_masks_exo = gt["masks"][object_id][exo_cam]

            if pred.get("masks") is None:
                continue

            if (
                object_id in pred["masks"].keys()
                and f"{exo_cam}" in pred["masks"][object_id].keys()
            ):
                pred_masks_exo = pred["masks"][object_id][f"{exo_cam}"]

            frame = cv2.imread(f"{datapath}/{take_id}/{exo_cam}/000001.jpg")
            orig_mask_shape = frame.shape[:2]

            for frame_idx in gt_masks_ego.keys():
                if (
                    frame_idx
                    not in take_annotation["object_masks"][object_id][exo_cam][
                        # "annotated_frames"
                        "annotation"
                    ].keys()
                ):
                    continue

                if not frame_idx in gt_masks_exo:
                    if orig_mask_shape[0] < orig_mask_shape[1]:
                        gt_mask = np.zeros((480, 480), dtype=np.uint8)
                    else:
                        gt_mask = np.zeros((480, 480), dtype=np.uint8)

                    gt_obj_exists = 0
                else:
                    gt_mask = mask_utils.decode(gt_masks_exo[frame_idx])
                    # reshaping without padding for evaluation
                    gt_mask = reshape_img_nopad_square(gt_mask, 480)

                    gt_obj_exists = 1
                gt_mask = reshape_img_nopad_square(gt_mask, 480)

                try:
                    if frame_idx in pred_masks_exo:
                        pred_mask = mask_utils.decode(pred_masks_exo[frame_idx])
                    else:
                        pred_mask = np.zeros_like(gt_mask)
                    # remove padding from the predictions
                    # pred_mask = remove_pad(pred_mask, orig_size=orig_mask_shape)
                except:
                    breakpoint()

                pred_mask = reshape_img_nopad_square(pred_mask, 480)

                pred_obj_exists = int(np.any(pred_mask > 0))

                # iou and shape accuracy
                try:
                    iou, shape_acc = eval_mask(gt_mask, pred_mask)
                except:
                    breakpoint()

                # compute existence acc i.e. if gt == pred == ALL ZEROS or gt == pred == SOME MASK
                ex_acc = existence_accuracy(gt_mask, pred_mask)

                # location accuracy as defined by Feng
                (
                    (gt_size, pred_size),
                    (centroid_gt, centroid_pred),
                    (gt_compact_mask, pred_compact_mask),
                ) = metrics.crop_mask(gt_mask, pred_mask)
                centroid_distance = np.sqrt(
                    (centroid_gt[0] - centroid_pred[0]) ** 2
                    + (centroid_gt[1] - centroid_pred[1]) ** 2
                )
                location_score = centroid_distance / np.sqrt(H**2 + W**2)

                # MdistNorm - distance b/w biggest contour
                mdist = metrics.getMidDistNorm(gt_mask, pred_mask)
                if mdist != -1:
                    MDistNorm.append(mdist)

                mdistbin = metrics.getMidBinning(gt_mask, pred_mask)
                if mdistbin != -1:
                    MDistBinning.append(mdistbin)

                IoUs.append(iou)
                ShapeAcc.append(shape_acc)
                ExistenceAcc.append(ex_acc)
                LocationScores.append(location_score)

                ObjSizeGT.append(np.sum(gt_mask).item())
                ObjSizePred.append(np.sum(pred_mask).item())
                IMSize.append(list(gt_mask.shape[:2]))

                ObjExist_GT.append(gt_obj_exists)
                ObjExist_Pred.append(pred_obj_exists)

    IoUs = np.array(IoUs)
    ShapeAcc = np.array(ShapeAcc)
    ExistenceAcc = np.array(ExistenceAcc)
    LocationScores = np.array(LocationScores)
    MDistNorm = np.array(MDistNorm)
    MDistBinning = np.array(MDistBinning)

    # print(f"ego_exo: Total {ego_exo.shape[0]}, Correct {np.sum(ego_exo)}, Accuracy {np.mean(ego_exo)}")
    # print(f"ego_noexo: Total {ego_noexo.shape[0]}, Correct {np.sum(ego_noexo)}, Accuracy {np.mean(ego_noexo)}")

    # print("ego_exo IoU: ", np.mean(IoUs))
    # print("ShapeAcc: ", np.mean(ShapeAcc))
    # print("ExistenceAcc: ", np.mean(ExistenceAcc))
    # if np.mean(IoUs) == 0:
    #     breakpoint()

    return (
        IoUs.tolist(),
        ShapeAcc.tolist(),
        ExistenceAcc.tolist(),
        LocationScores.tolist(),
        MDistNorm.tolist(),
        MDistBinning.tolist(),
        ObjExist_GT,
        ObjExist_Pred,
        ObjSizeGT,
        ObjSizePred,
        IMSize,
    )  # list(ego_exo), list(ego_noexo)


def processGTPred_EXOEGO(take_annotation, gt, pred, object_ids):
    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []
    MDistNorm = []
    MDistBinning = []
    ObjExist_GT = []
    ObjExist_Pred = []

    ObjSizeGT = []
    ObjSizePred = []
    IMSize = []

    H, W = 480, 480  # resolution for evalution

    for object_id in object_ids:
        ego_cams = [x for x in gt["masks"][object_id].keys() if "aria" in x]
        if len(ego_cams) < 1:
            continue
        assert len(ego_cams) == 1
        EGOCAM = ego_cams[0]

        EXOCAMS = [x for x in gt["masks"][object_id].keys() if "aria" not in x]
        for exo_cam in EXOCAMS:
            gt_masks_ego = {}
            gt_masks_exo = {}
            pred_masks_ego = {}

            if EGOCAM in gt["masks"][object_id].keys():
                gt_masks_ego = gt["masks"][object_id][EGOCAM]
            if exo_cam in gt["masks"][object_id].keys():
                gt_masks_exo = gt["masks"][object_id][exo_cam]

            if pred.get("masks") is None:
                import pdb

                pdb.set_trace()
                continue

            if (
                object_id in pred["masks"].keys()
                and f"{EGOCAM}" in pred["masks"][object_id].keys()
            ):
                pred_masks_ego = pred["masks"][object_id][exo_cam]

            for frame_idx in gt_masks_exo.keys():
                if (
                    frame_idx
                    not in take_annotation["object_masks"][object_id][exo_cam][
                        # "annotated_frames"
                        "annotation"
                    ].keys()
                ):
                    continue

                if not frame_idx in gt_masks_ego:
                    gt_mask = np.zeros((H, W), dtype=np.uint8)
                    gt_obj_exists = 0
                else:
                    gt_mask = mask_utils.decode(gt_masks_ego[frame_idx])
                    gt_obj_exists = 1
                gt_mask = reshape_img_nopad_square(gt_mask, 480)

                # breakpoint()
                if frame_idx in pred_masks_ego:
                    pred_mask = mask_utils.decode(pred_masks_ego[frame_idx])
                else:
                    pred_mask = np.zeros_like(gt_mask)
                pred_mask = reshape_img_nopad_square(pred_mask, 480)

                pred_obj_exists = int(np.any(pred_mask > 0))

                # iou and shape accuracy
                iou, shape_acc = eval_mask(gt_mask, pred_mask)

                # compute existence acc i.e. if gt == pred == ALL ZEROS or gt == pred == SOME MASK
                ex_acc = existence_accuracy(gt_mask, pred_mask)

                # location accuracy as defined by Feng
                (
                    (gt_size, pred_size),
                    (centroid_gt, centroid_pred),
                    (gt_compact_mask, pred_compact_mask),
                ) = metrics.crop_mask(gt_mask, pred_mask)
                centroid_distance = np.sqrt(
                    (centroid_gt[0] - centroid_pred[0]) ** 2
                    + (centroid_gt[1] - centroid_pred[1]) ** 2
                )
                location_score = centroid_distance / np.sqrt(H**2 + W**2)

                # MdistNorm - distance b/w biggest contour
                mdist = metrics.getMidDistNorm(gt_mask, pred_mask)
                if mdist != -1:
                    MDistNorm.append(mdist)

                mdistbin = metrics.getMidBinning(gt_mask, pred_mask)
                if mdistbin != -1:
                    MDistBinning.append(mdistbin)

                IoUs.append(iou)
                ShapeAcc.append(shape_acc)
                ExistenceAcc.append(ex_acc)
                LocationScores.append(location_score)
                ObjExist_GT.append(gt_obj_exists)
                ObjExist_Pred.append(pred_obj_exists)

                ObjSizeGT.append(np.sum(gt_mask).item())
                ObjSizePred.append(np.sum(pred_mask).item())
                IMSize.append(list(gt_mask.shape[:2]))

    IoUs = np.array(IoUs)
    ShapeAcc = np.array(ShapeAcc)
    ExistenceAcc = np.array(ExistenceAcc)
    LocationScores = np.array(LocationScores)
    MDistNorm = np.array(MDistNorm)
    MDistBinning = np.array(MDistBinning)

    # print(f"ego_exo: Total {ego_exo.shape[0]}, Correct {np.sum(ego_exo)}, Accuracy {np.mean(ego_exo)}")
    # print(f"ego_noexo: Total {ego_noexo.shape[0]}, Correct {np.sum(ego_noexo)}, Accuracy {np.mean(ego_noexo)}")

    # print("ego_exo IoU: ", np.mean(IoUs))
    # print("ShapeAcc: ", np.mean(ShapeAcc))
    # print("ExistenceAcc: ", np.mean(ExistenceAcc))
    # if np.mean(IoUs) == 0:
    #     breakpoint()

    return (
        IoUs.tolist(),
        ShapeAcc.tolist(),
        ExistenceAcc.tolist(),
        LocationScores.tolist(),
        MDistNorm.tolist(),
        MDistBinning.tolist(),
        ObjExist_GT,
        ObjExist_Pred,
        ObjSizeGT,
        ObjSizePred,
        IMSize,
    )  # list(ego_exo), list(ego_noexo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        help="the correspondence dataset path",
        default="/Users/shawn/Desktop/proj-egoexo/correspondence_dataset/dataset_correspondence",
    )
    parser.add_argument(
        "--inference_path",
        help="The predicted path",
        default="/Users/shawn/Desktop/proj-egoexo/correspondence_dataset/dataset_correspondence",
    )
    parser.add_argument(
        "--reversed", help="if true: do Exo->Ego else Ego->Exo", action="store_true"
    )

    args = parser.parse_args()

    split_file = f"{args.datapath}/split.json"
    with open(split_file, "r") as f:
        split = json.load(f)
        test_ids = split[EVALMODE]

    annotations_file = f"{args.datapath}/relations_objects_latest.json"
    with open(annotations_file, "r") as fp:
        annotations = json.load(fp)

    total_iou = []
    total_shape_acc = []
    total_existence_acc = []
    total_location_scores = []
    total_mdistnorm_scores = []
    total_mdistbin_scores = []
    total_obj_sizes_gt = []
    total_obj_sizes_pred = []
    total_img_sizes = []

    total_obj_exists_gt = []
    total_obj_exists_pred = []

    take_metrics = {}
    take_raw = {}

    for take_id in tqdm.tqdm(test_ids):
        print(f"Processing take {take_id}")
        # Load the GT annotations
        gt_file = f"{args.datapath}/{take_id}/annotation.json"
        with open(gt_file, "r") as f:
            gt = json.load(f)
        # Load the predictions
        pred_file = f"{args.inference_path}/{take_id}/pred_annotations.json"
        with open(pred_file, "r") as f:
            pred = json.load(f)

        object_ids = list(gt["masks"].keys())

        if args.reversed:
            (
                ious,
                shape_accs,
                existence_accs,
                location_scores,
                mdistnorm_scores,
                mdistbin_scores,
                obj_exist_gt,
                obj_exist_pred,
                obj_size_gt,
                obj_size_pred,
                img_sizes,
            ) = processGTPred_EXOEGO(annotations[take_id], gt, pred, object_ids)
        else:
            (
                ious,
                shape_accs,
                existence_accs,
                location_scores,
                mdistnorm_scores,
                mdistbin_scores,
                obj_exist_gt,
                obj_exist_pred,
                obj_size_gt,
                obj_size_pred,
                img_sizes,
            ) = processGTPred_EGOEXO(
                args.datapath, take_id, annotations[take_id], gt, pred, object_ids
            )
        total_iou += ious
        total_shape_acc += shape_accs
        total_existence_acc += existence_accs
        total_location_scores += location_scores
        total_mdistnorm_scores += mdistnorm_scores
        total_mdistbin_scores += mdistbin_scores
        total_obj_sizes_gt += obj_size_gt
        total_obj_sizes_pred += obj_size_pred
        total_img_sizes += img_sizes
        total_obj_exists_gt += obj_exist_gt
        total_obj_exists_pred += obj_exist_pred

        take_metrics[take_id] = {
            "iou": np.mean(ious),
            "shape_accs": np.mean(shape_accs),
            "existence_acc": np.mean(existence_accs),
            "location_score": np.mean(location_scores),
            "mdistnorm_score": np.mean(mdistnorm_scores),
            "mdistbin_score": np.mean(mdistbin_scores),
            "mdistbin_score": np.mean(mdistbin_scores),
            "existence_balanced_acc": balanced_accuracy_score(
                obj_exist_gt, obj_exist_pred
            ),
        }

        take_raw[take_id] = {
            "iou": ious,
            "shape_accs": shape_accs,
            "existence_acc": existence_accs,
            "location_score": location_scores,
            "mdistnorm_score": mdistnorm_scores,
            "mdistbin_score": mdistbin_scores,
            "obj_exist_gt": obj_exist_gt,
            "obj_exist_pred": obj_exist_pred,
            "obj_size_gt": obj_size_gt,
            "obj_size_pred": obj_size_pred,
            "img_sizes": img_sizes,
        }
        print("\n")

    # print(f"Total ego_exo: Total {len(total_ego_exo)}, Correct {np.sum(total_ego_exo)}, Accuracy {np.mean(total_ego_exo)}")
    # print(f"Total ego_noexo: Total {len(total_ego_noexo)}, Correct {np.sum(total_ego_noexo)}, Accuracy {np.mean(total_ego_noexo)}")
    print("TOTAL IOU: ", len(total_iou), np.mean(total_iou))
    print("TOTAL SHAPE ACC: ", len(total_shape_acc), np.mean(total_shape_acc))
    print(
        "TOTAL EXISTENCE ACC: ", len(total_existence_acc), np.mean(total_existence_acc)
    )
    print(
        "TOTAL LOCATION SCORE: ",
        len(total_location_scores),
        np.mean(total_location_scores),
    )
    print(
        "TOTAL MDISTNORM SCORE: ",
        len(total_mdistnorm_scores),
        np.mean(total_mdistnorm_scores),
    )
    print(
        "TOTAL MDISTBIN SCORES: ",
        len(total_mdistbin_scores),
        np.mean(total_mdistbin_scores),
    )
    print(
        "TOTAL EXISTENCE BALANCED ACC: ",
        len(total_obj_exists_gt),
        balanced_accuracy_score(total_obj_exists_gt, total_obj_exists_pred),
    )

    with open(f"{args.inference_path}/take_metrics_{EVALMODE}.json", "w") as fp:
        json.dump(take_metrics, fp)

    with open(f"{args.inference_path}/take_raw_{EVALMODE}.json", "w") as fp:
        json.dump(take_raw, fp)

    total_metrics = {
        "iou": total_iou,
        "shape_accs": total_shape_acc,
        "existence_acc": total_existence_acc,
        "location_score": total_location_scores,
        "mdistnorm_score": total_mdistbin_scores,
        "mdistbin_score": total_mdistbin_scores,
        "obj_exists_gt": total_obj_exists_gt,
        "obj_exists_pred": total_obj_exists_pred,
        "obj_size_gt": total_obj_sizes_gt,
        "obj_size_pred": total_obj_sizes_pred,
        "img_sizes": total_img_sizes,
    }

    with open(f"{args.inference_path}/total_metrics_{EVALMODE}.json", "w") as fp:
        json.dump(total_metrics, fp)
