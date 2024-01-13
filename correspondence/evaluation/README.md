To run the evaluation, run the following command:
```
# for ego-exo
python evaluate_egoexo.py --gt-file /path/to/correspondence-gt.json --pred-file /path/to/pred.json 

#for exo-ego
python evaluate_exoego.py --gt-file /path/to/correspondence-gt.json --pred-file /path/to/pred.json 
```

The prediction gt expects the following structure.  
```json
{   
    "version": "xx",
    "challenge": "correspondence",
    "ego-exo": {
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
