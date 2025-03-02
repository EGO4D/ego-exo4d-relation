# script to generate submission file to evalai correspondence challenge
import os
import json
import argparse

def main(args):

    with open(args.ego_exo_preds, "r") as fp:
        ego_exo_preds = json.load(fp)

    with open(args.exo_ego_preds, "r") as fp:
        exo_ego_preds = json.load(fp)

    submission = {
        'ego-exo': {
            "version": "xx", # DO NOT CHANGE
            "challenge": "xx", # DO NOT CHANGE
            "results": ego_exo_preds["ego-exo"]["results"]
        }, 
        'exo-ego': {
            "version": "xx", # DO NOT CHANGE
            "challenge": "xx", # DO NOT CHANGE
            "results": exo_ego_preds["exo-ego"]["results"]
        }, 
    }

    with open(os.path.join(args.output_path, "submission.json"), "w") as fp:
        json.dump(submission, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego_exo_preds", type=str, required=True,
                        help="path to json file containing ego-exo predictions")
    parser.add_argument("--exo_ego_preds", type=str, required=True,
                        help="path to json file containing exo-ego predictions")
    parser.add_argument("--output_path", type=str, required=True,
                        help="directory for the output submission json")
    args = parser.parse_args()
    
    main(args)