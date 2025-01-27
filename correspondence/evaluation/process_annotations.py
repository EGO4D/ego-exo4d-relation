import os
import json
import argparse
import tqdm

def preprocess_annotations(data_path, annotation_path, split, output_path):

    # load split
    with open(f'{data_path}/split_checking.json', 'r') as fp:
        splits = json.load(fp)
    
    with open(annotation_path, "r") as fp:
        gt_annotations = json.load(fp)['annotations']

    takes = splits[split]

    annotations = {"version": "xx",
                    "challenge": "xx",
                    "annotations": {}}
    for take in tqdm.tqdm(takes):

        with open(f'{data_path}/{take}/annotation.json', 'r') as fp:
            vid_anno = json.load(fp)

        annotations["annotations"][take] = vid_anno
        annotations["annotations"][take]["annotated_frames"] = {}
        for obj in vid_anno['masks'].keys():
            annotations["annotations"][take]["annotated_frames"][obj] = {}
            for cam in vid_anno['masks'][obj].keys():
                annotations["annotations"][take]["annotated_frames"][obj][cam] = gt_annotations[take]['object_masks'][obj][cam]['annotated_frames']

    # breakpoint()
    with open(output_path, 'w') as fp:
        json.dump(annotations, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to directory with processed dataset")
    parser.add_argument("--annotations_path", type=str, required=True,
                        help="Path to annotations file from EgoExo4D e.g. `relations_test.json`")
    parser.add_argument("--split", type=str, required=True,
                        help="Split on which to evaluate. train/test/val")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output processed annotations")
    args = parser.parse_args()

    preprocess_annotations(args.data_path, args.annotations_path, args.split, args.output_path)