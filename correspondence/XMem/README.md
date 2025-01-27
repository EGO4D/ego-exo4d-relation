# Ego-Exo4D Correspondence (XView-Xmem) Baseline Model

## 1. Installation
Follow [xSeg-Tx/Installation](../SegSwap/README.md#1-installation) create the conda environment. Then install:
```bash
pip install -r requirements.txt
```

### 1.2. Pre-trained XMem checkpoint
Quick download :
```bash
cd scripts
bash download_models.sh
```

## 2. Data preparation
Follow [xSeg-Tx/Data preparation](../SegSwap/README.md##2-data-preparation) to prepare the data.

## 3. Training
For fintuning ego->exo (without xSeg-Tx):
`nproc_per_node` refers to the number of GPUs to be used.
```bash
python -m torch.distributed.launch --master_port 25763 --nproc_per_node=4 train.py --exp_id xmem_egoexo_tune --stage 3 --load_network pretrained/XMem.pth --egoexo_root /path/to/data --save_network_interval 1000 --save_checkpoint_interval 2000
```

For training with baseline XView-Xmem (+ XSegTx), attach following commands to above:
```bash
--enable_segswap --segswap_model /path/to/xSeg-Tx/checkpoint.pth
```

For training exo->ego, add `--swap`.

For more details, check out original XMem training doc [link](https://github.com/hkchengrex/XMem/blob/main/docs/TRAINING.md).

## 4. Inference
Run inference for ego->exo (without xSeg-Tx)::
```bash
python eval.py --model /path/to/checkpoint.pth --save_all --output /path/to/output/ --e23_path /path/to/data --split test
```
For inferencing with baseline XView-Xmem (+ XSegTx), attach `--enable_segswap`.
For inferencing with exo->ego, add `--swap`.

### 4.1 Merge inference results
Run:
```bash
python scripts/merge_pred.py --pred /inference/output/ --input /path/to/data --gt /path/to/correspondence-gt.json --split test
python scripts/merge_results.py --input_dir /path/to/data --split test --pred_dir /inference/output/coco # add --swap for exo->ego
```
`final_results.json` will be saved in `/inference/output/`. The file can be then be used to run evaluation as described [here](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence/evaluation).

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

The code in this repo is heavily based on [XMem](https://github.com/hkchengrex/XMem).