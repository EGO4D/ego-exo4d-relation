
import json
import argparse
from pycocotools import mask as mask_utils
import numpy as np
import tqdm
from sklearn.metrics import balanced_accuracy_score

import utils

CONF_THRESH = 0.5
H, W = 480, 480 # resolution for evalution

def evaluate_take(gt, pred):
    
    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []

    ObjExist_GT = []
    ObjExist_Pred = []

    ObjSizeGT = []
    ObjSizePred = []
    IMSize = []

    for object_id in gt['masks'].keys():
        ego_cams = [x for x in gt['masks'][object_id].keys() if 'aria' in x]
        # TODO: remove takes with no ego cam annotations from gt
        if len(ego_cams) < 1:
            continue
        assert len(ego_cams) == 1
        EGOCAM = ego_cams[0]

        EXOCAMS = [x for x in gt['masks'][object_id].keys() if 'aria' not in x]
        for exo_cam in EXOCAMS:
            gt_masks_ego = {}
            gt_masks_exo = {}
            pred_masks_ego = {}

            if EGOCAM in gt["masks"][object_id].keys():
                gt_masks_ego = gt["masks"][object_id][EGOCAM]
            if exo_cam in gt["masks"][object_id].keys():
                gt_masks_exo = gt["masks"][object_id][exo_cam]
            if object_id in pred["masks"].keys() and f'{exo_cam}_{EGOCAM}' in pred["masks"][object_id].keys():
                pred_masks_ego = pred["masks"][object_id][f'{exo_cam}_{EGOCAM}']

            for frame_idx in gt_masks_exo.keys():
                
                if int(frame_idx) not in gt["annotated_frames"][object_id][EGOCAM]:
                    continue
                
                if not frame_idx in gt_masks_ego:
                    gt_mask = None
                    gt_obj_exists = 0
                else:
                    gt_mask = mask_utils.decode(gt_masks_ego[frame_idx])
                    # reshaping without padding for evaluation
                    # # TODO: remove from here: move to inference script
                    # gt_mask = utils.reshape_img_nopad(gt_mask)

                    gt_obj_exists = 1

                try:
                    pred_mask = mask_utils.decode(pred_masks_ego[frame_idx]["pred_mask"])
                    # remove padding from the predictions
                    # # TODO: remove from here: move to inference script
                    # if not gt_mask is None:
                    #     pred_mask = utils.remove_pad(pred_mask, orig_size=gt_mask.shape[:2])
                except:
                    breakpoint()

                pred_obj_exists = int(pred_masks_ego[frame_idx]["confidence"] > CONF_THRESH)

                if gt_obj_exists:
                    # iou and shape accuracy
                    try:
                        iou, shape_acc = utils.eval_mask(gt_mask, pred_mask)
                    except:
                        breakpoint()

                    # compute existence acc i.e. if gt == pred == ALL ZEROS or gt == pred == SOME MASK
                    ex_acc = utils.existence_accuracy(gt_mask, pred_mask)

                    # # location accuracy
                    location_score = utils.location_score(gt_mask, pred_mask, size=(H, W))

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

    return IoUs.tolist(), ShapeAcc.tolist(), ExistenceAcc.tolist(), LocationScores.tolist(), \
            ObjExist_GT, ObjExist_Pred, ObjSizeGT, ObjSizePred, IMSize

def validate_predictions(gt, preds):

    assert "exo-ego" in preds
    preds = preds["exo-ego"]

    assert type(preds) == type({})
    for key in ["version", "challenge", "results"]:
        assert key in preds.keys()

    assert preds["version"] == gt["version"]
    assert preds["challenge"] == gt["challenge"]

    assert len(preds["results"]) == len(gt["annotations"])
    for take_id in gt["annotations"]:
        assert take_id in preds["results"]

        for key in ["masks", "subsample_idx"]:
            assert key in preds["results"][take_id]

        # check objs
        assert len(preds["results"][take_id]["masks"]) == len(gt["annotations"][take_id]["masks"])
        for obj in gt["annotations"][take_id]["masks"]:
            assert obj in preds["results"][take_id]["masks"], f"{obj} not in pred {take_id}"

            ego_cam = None
            exo_cams = []
            for cam in gt["annotations"][take_id]["masks"][obj]:
                if 'aria' in cam:
                    ego_cam = cam
                else:
                    exo_cams.append(cam)
            try:
                assert not ego_cam is None
            except:
                #TODO: post process gt to not include these objects without aria annotations
                continue
            try:
                assert len(exo_cams) > 0
            except:
                #TODO: post process gt to not include these objects with only aria annotations
                continue

            for cam in exo_cams:
                try:
                    assert f"{cam}_{ego_cam}" in preds["results"][take_id]["masks"][obj]
                except:
                    breakpoint()

                for idx in gt["annotations"][take_id]["masks"][obj][cam]:
                    assert idx in preds["results"][take_id]["masks"][obj][f"{cam}_{ego_cam}"]

                    for key in ["pred_mask", "confidence"]:
                        assert key in preds["results"][take_id]["masks"][obj][f"{cam}_{ego_cam}"][idx]

def evaluate(gt, preds):

    validate_predictions(gt, preds)
    preds = preds["exo-ego"]

    total_iou = []
    total_shape_acc = []
    total_existence_acc = []
    total_location_scores = []

    total_obj_sizes_gt = []
    total_obj_sizes_pred = []
    total_img_sizes = []

    total_obj_exists_gt = []
    total_obj_exists_pred = []

    for take_id in tqdm.tqdm(gt["annotations"]):

        ious, shape_accs, existence_accs, location_scores, obj_exist_gt, obj_exist_pred, \
            obj_size_gt, obj_size_pred, img_sizes = evaluate_take(gt["annotations"][take_id], 
                                                                  preds["results"][take_id])

        total_iou += ious
        total_shape_acc += shape_accs
        total_existence_acc += existence_accs
        total_location_scores += location_scores

        total_obj_sizes_gt += obj_size_gt
        total_obj_sizes_pred += obj_size_pred
        total_img_sizes += img_sizes

        total_obj_exists_gt += obj_exist_gt
        total_obj_exists_pred += obj_exist_pred
    
    print('TOTAL EXISTENCE BALANCED ACC: ', balanced_accuracy_score(total_obj_exists_gt, total_obj_exists_pred))
    print('TOTAL IOU: ', np.mean(total_iou))
    print('TOTAL LOCATION SCORE: ', np.mean(total_location_scores))
    print('TOTAL SHAPE ACC: ', np.mean(total_shape_acc))

def main(args):

    # load gt and pred jsons
    with open(args.gt_file, 'r') as fp:
        gt = json.load(fp)

    with open(args.pred_file, 'r') as fp:
        preds = json.load(fp)

    # evaluate
    evaluate(gt, preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-file', type=str, required=True, 
                            help="path to json with gt annotations")
    parser.add_argument('--pred-file', type=str, required=True,
                            help="")
    args = parser.parse_args()
    
    main(args)