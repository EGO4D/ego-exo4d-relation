import os
import argparse
import json

from tqdm.auto import tqdm


def main(args):
    input_path = os.path.join(args.input, args.split)
    takes = os.listdir(input_path)

    annotation_path = args.gt
    with open(annotation_path, "r") as fp:
        annotation = json.load(fp)

    for take_id in tqdm(takes):
        if not os.path.isdir(os.path.join(args.pred, take_id)):
            continue
        result = process_take(take_id, annotation, args.pred)

        with open(os.path.join(args.pred, take_id, "annotations.json"), "w+") as fp:
            json.dump(result, fp)


def get_folders(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def process_take(take_id, annotation, pred):
    pred_masks = {}
    cam_names = get_folders(os.path.join(pred, take_id))
    object_names = list(annotation["annotations"][take_id]["masks"].keys())

    count_empty = 0
    for object_name in object_names:
        pred_masks[object_name] = {}
        for cams_str in cam_names:
            pred_masks[object_name][cams_str] = {}
            if (
                cams_str.split("__")[0]
                not in annotation["annotations"][take_id]["masks"][object_name]
            ):
                continue
            f_ids = list(
                annotation["annotations"][take_id]["masks"][object_name][
                    cams_str.split("__")[0]
                ].keys()
            )
            for f_id in f_ids:
                f_str = f"{f_id}.json"

                pred_mask_path = os.path.join(
                    pred, take_id, cams_str, object_name, f_str
                )
                if not os.path.isfile(pred_mask_path):
                    pred_mask_data = {}
                    count_empty += 1
                else:
                    with open(pred_mask_path, "r") as fp:
                        pred_mask_data = json.load(fp)
                pred_masks[object_name][cams_str][f_str.split(".")[0]] = pred_mask_data
    print("Empty masks: ", count_empty)
    return {"masks": pred_masks, "subsample_idx": f_ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="EgoExo4D take data root")
    parser.add_argument("--gt", help="EgoExo4D gt annotations file path")
    parser.add_argument("--split", help="Dataset split", default="val")
    parser.add_argument("--pred", help="EgoExo4D predicted results")
    args = parser.parse_args()

    main(args)
