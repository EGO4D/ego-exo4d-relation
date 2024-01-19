## Evaluation code for Ego-exo translation task.

Translation task entails synthesizing a target ego clip from a given exo clip. This is decomposed into two separate tasks: **ego track prediction** and **ego clip generation**.

See task definition in [docs.ego-exo4d-data.org](https://docs.ego-exo4d-data.org/benchmarks/relations/translation/)

### 1. Evaluate **ego track prediction**

Evaluation code:

```bash
# suppose your results are saved in directory /path/to/your/results
python eval_subtask1.py --results_dir /path/to/your/results
```

Store your results in the following structure.

```
/path/to/your/results/
  |- ground-truths   # store ground-truth ego frames
      |- {take_name}-{exo_camera}-{object_name}-{frame_id}.png
      |- ...
  |- predictions     # store generated ego frames
      |- {take_name}-{exo_camera}-{object_name}-{frame_id}.png
      |- ...

```

Each image is either the ground-truth or predicted **ego mask**, named with `{take_name}-{exo_camera}-{object_name}-{frame_id}.png`.

For example: `sfu_cooking_012_5-cam01-Boiled pasta_0-25410.png` is the GT/predicted ego mask.

- for take `sfu_cooking_012_5`
- from exo view `cam01`
- for object `Boiled pasta_0`
- in frame id `25410` in the time-synced videos.

### 2. Evaluate **ego clip generation**

Evaluation code:

```bash
# suppose your results are saved in directory /path/to/your/results
python eval_subtask2.py --results_dir /path/to/your/results
```

Store your results in the following structure.

```
/path/to/your/results/
  |- ground-truths   # store ground-truth ego frames
      |- {take_name}-{exo_camera}-{object_name}-{frame_id}.png
      |- ...
  |- predictions     # store generated ego frames
      |- {take_name}-{exo_camera}-{object_name}-{frame_id}.png
      |- ...

```

Each image is either the ground-truth or predicted ego frame, named with `{take_name}-{exo_camera}-{object_name}-{frame_id}.png`.

For example: `sfu_cooking_012_5-cam01-Boiled pasta_0-25410.png` is the GT/predicted ego frame

- for take `sfu_cooking_012_5`
- from exo view `cam01`
- for object `Boiled pasta_0`
- in frame id `25410` in the time-synced videos.
