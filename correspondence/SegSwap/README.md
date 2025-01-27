# Ego-Exo4D Correspondence (xSeg-Tx) Baseline Model

Implementation of a correspondence baseline model in [Ego-Exo4D](https://ego-exo4d-data.org/) based on [SegSwap](https://github.com/XiSHEN0220/SegSwap).

## 1. Installation

```
conda env create -f environment.yaml
```

### 1.2. Pre-trained MocoV2-resnet50 + cross-transformer (~300M) from SegSwap

Quick download : 

``` Bash
cd model/pretrained
bash download_model.sh
```


## 2. Data preparation

Follow the instructions to first install the [Ego-Exo4D CLI Downloader](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/egoexo/download/README.md).

Download the takes and annotations for the correspondence benchmark.

```
egoexo -o /path/to/data/dir/ --parts annotations

egoexo -o /path/to/data/dir/ --parts takes --benchmarks correspondence -y
```

You may also use this [script](data/download_uids.sh) to download only the takes for version v2 of data used in the paper.

Once the data has been downloaded, use the following command to pre-process the data for each train, val and test splits,
```
python data/process_data.py --takepath /path/to/data/dir/takes/ --annotationpath /path/to/data/dir/annotations/relations_{split}.json --split_path data/split.json --split {split} --outputpath /path/to/output/dir/
```

Since, SegSwap based baseline (XSeg-Tx) is trained on pairs of ego->exo / exo->ego images, use the [create_pairs.py](data/create_pairs.py) script to generate the pairs.
```
python data/create_pairs.py --data_dir /path/to/output/dir/
``` 

If not already present, also copy the [split.json](data/split.json) to the data directory.

## 3. Training 

We provide the scripts to train both ego->exo and exo->ego models, 

``` Bash
cd train
bash run_ego.sh
```

``` Bash
cd train
bash run_exo.sh
```

## 4. Inference

We can then run inference on a specific split as follows,

for ego->exo,
```
python eval_segswap.py --ckpt_path /path/to/checkpoint.pth --data_path /path/to/data --splits_path /path/to/data/split.json  --split test --out_path /path/to/output/ --setting ego-exo
```

for exo->ego,
```
python eval_segswap.py --ckpt_path /path/to/checkpoint.pth --data_path /path/to/data --splits_path /path/to/data/split.json  --split test --out_path /path/to/output/ --setting exo-ego
```

the above command should produce a `ego-exo_test_results.json`/`exo-ego_test_results.json` file which can be then be used to run evaluation as described [here](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence/evaluation).

## 5. Acknowledgement

If you use the code base, consider citing
```
@article{grauman2023ego,
  title={Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives},
  author={Grauman, Kristen and Westbury, Andrew and Torresani, Lorenzo and Kitani, Kris and Malik, Jitendra and Afouras, Triantafyllos and Ashutosh, Kumar and Baiyya, Vijay and Bansal, Siddhant and Boote, Bikram and others},
  journal={arXiv preprint arXiv:2311.18259},
  year={2023}
}
```

The code in this repo is heavily based on [SegSwap](https://github.com/XiSHEN0220/SegSwap).