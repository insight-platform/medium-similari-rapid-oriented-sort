# Evaluate tracker performance

This repo allows to measure and evaluate both speed and accuracy related metrics of multi object trackers.

Currently, following trackers are implemented:

* Similari with IOU metric
* Similari with Maha metric

## Data

Measurement are performed using [WEPDTOF](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/) dataset, specifically, on `convenience_store` and `call_center` sequences.

1. Download data from the [WEPDTOF](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/) page.

2. Extract into `<repo_root>/data`.

The following file tree is expected

```
data/
    WEPDTOF/
        annotations/
            convenience_store.json
            ...
        frames/
            convenience_store/
                convenience_store_000001.jpg
                ...
```

## RAPiD weights

Download `RAPiD.ckpt` from official repo

https://github.com/ozantezcan/RAPiD-T
 

Place the model checkpoint into `./weights` directory.

## Run

Check scripts for possible arguments (trackers, dataset sequences, detections source).

### Run FPS measurement

```
docker compose up measure-fps-trackers
```

### Run RAPiD predict

```
docker compose up predict-rapid
```

### Run tracker predict

```
docker compose up predict-trackers
```

### Run GT or tracker annotations visualization

```
docker compose up draw-gt
```

Or

```
docker compose up draw-detections
```

Or

```
docker compose up draw-tracks
```

Results will be written into the `frames` directory, for example

```
data/
    WEPDTOF/
        frames/
            convenience_store_GT/
                000001.jpg
                ...
```

### Run tracking evaluation

```
docker compose up evaluate-trackers
```

Use [notebook](notebooks/read_eval_results.ipynb) to conveniently present csv results.
