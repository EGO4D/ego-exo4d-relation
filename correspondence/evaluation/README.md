To run the evaluation, first process the annotations  
```
python process_annotations.py --data_path /path/to/processed/data/ --annotations_path /path/to/relations_{train/test/val}.json --split {train/test/val} --output_path /path/to/output/correspondence-gt.json
```

then run the following command:
```
# for ego-exo
python evaluate_egoexo.py --gt-file /path/to/correspondence-gt.json --pred-file /path/to/pred.json 

#for exo-ego
python evaluate_exoego.py --gt-file /path/to/correspondence-gt.json --pred-file /path/to/pred.json 
```

The prediction gt expects the following structure.  
```json
{   
    "ego-exo": {
        "version": "xx",
        "challenge": "xx",
        "results": {
            "take-id": { //take-id as in gt. 
                "masks" : {
                    "obj_0": {
                        "{ego-cam}_{exo-cam}": {
                            "0": {
                                "pred_mask": { //coco format
                                    "size": [], 
                                    "counts": "xxx" 
                                },
                                "confidence": 0.0  //confidence that an object exists
                            },
                        },
                        .
                        .
                    },
                    .
                    .
                },
                "subsample_idx": [] //indexes annotated as in gt
            }
        }
    },
    "exo-ego" : {
        "version": "xx",
        "challenge": "xx",
        "results": {
            "take-id": { //take-id as in gt. 
                "masks" : {
                    "obj_0": {
                        "{exo-cam}_{ego-cam}": {
                            "0": {
                                "pred_mask": { //coco format
                                    "size": [], 
                                    "counts": "xxx" 
                                },
                                "confidence": 0.0  //confidence that an object exists
                            },
                        },
                        .
                        .
                    },
                    .
                    .
                },
                "subsample_idx": [] //indexes annotated as in gt
            }
        }
    }
}
```

### EvalAI Challenge Submission

Once you have prediction jsons for both ego-exo and exo-ego settings, you can use [this script](evalai/generate_evalai_submission.py) to generate a submission json for the evalai correspondence challenge,

```
python evaluation/evalai/generate_evalai_submission.py --ego_exo_preds /path/to/ego-exo-preds.json --exo_ego_preds /path/to/exo-ego-preds.json --output_path /path/to/output/dir/
```

this should generate a `submission.json` at `output_path` which is ready to be submitted to [evalai challenge](https://eval.ai/web/challenges/challenge-page/2288/overview).