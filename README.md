# Growth Coordinate Learning

Code for **Growth Coordinate Learning: A Source-Supervised Visual Representation for Cross-Domain Plant Growth Monitoring**.

Growth Coordinate Learning (GCL) uses limited lettuce trait supervision to shape frozen DINOv2 features with a lightweight MLP. Significant-subspace Alignment (SSA) adapts the representation to unlabeled target domains. The first principal component provides a developmental coordinate for monitoring relative growth and departures from expected trajectories.

## Setup

Python 3.10, PyTorch 2.9.1 and CUDA 12.8 were used for the experiments.

```bash
python -m pip install -r requirements.txt
```

## Data

The experiments use lettuce and tomato from the WUR Autonomous Greenhouse Challenge, komatsuna, LAB-Lettuce, and four GroMo25 crops (mustard, okra, radish, and wheat).

LAB-Lettuce is available at [Mendeley Data](https://doi.org/10.17632/b82wmb5yj6.1). Please obtain the other datasets from the original sources cited in the paper.

The scripts start from frozen backbone features. Configure image, feature, and annotation paths in `data/datasets.yml`. The feature root can also be set with:

```bash
export PFAL_REPRESENTATION_DATA_ROOT=/path/to/features
```

Image-to-backbone feature extraction is outside these training scripts. Set checkpoint paths and seed templates in `data/models.yml` to the models you train below.

## Training

Train the single-trait and multi-trait source MLPs:

```bash
python TrainMLP.py --dataset Lettuce --targets FreshWeightShoot \
  --seed 41 --output weights/lettuce-single.pth

python TrainMLP.py --dataset Lettuce \
  --targets FreshWeightShoot DryWeightShoot Height Diameter LeafArea \
  --seed 43 --output weights/lettuce-multi.pth
```

Adapt a source MLP to an unlabeled target dataset:

```bash
python TrainSSA.py --source Lettuce --target Tomato \
  --source-model weights/lettuce-multi.pth --output weights/tomato-ssa.pth
```

Both training scripts support the datasets configured in `data/datasets.yml`. One target column trains a single-head MLP; multiple columns train a multi-head MLP. SSA does not use target labels.

SSA defaults are K=70, rank=32, 20 epochs, batch size 256, and learning rate 0.005. Use `--before 2025-12-08` for LAB adaptation and `--shuffle` for the GroMo25 protocol. Repeated experiments use seeds 40–44. Available options, including ablations and intermediate checkpoints, are listed by `--help`.

## Figures and tables

Each computed figure or table has its own script. After configuring the trained models in `data/models.yml`, run the corresponding entry point, for example:

```bash
python Fig5.py
python Table5.py
```

| Entry points | Content |
| --- | --- |
| `Fig3.py`–`Fig6.py`, `FigS1.py`–`FigS6.py` | Representation analysis, growth trajectories, and supplementary experiments |
| `Table1.py`–`Table10.py`, `TableS1.py`, `TableS5.py`–`TableS8.py` | Quantitative evaluation and ablations |
| `src/` | Shared data loading, models, training, evaluation, and plotting |

Outputs are written to `results/`; use `--output-dir` to choose another directory. `FigS1.py --train --model-dir weights/scaling` trains the data-scaling experiment. FigS6 requires intermediate SSA checkpoints saved with `TrainSSA.py --snapshot-steps`.

Figures 1, 2, and 7 are manually composed; Supplementary Tables S2–S4 describe the methods. Fig7's computed panels can be exported with `FigS4.py --datasets okra` and `FigS5.py --datasets okra`.

## Reproducibility

A fresh run from frozen backbone features retrained 196 models and recomputed all 15 numerical tables and 10 computed figures in the stated environment. All 576 manuscript decimals matched at the reported precision, and all 20 coverage counts matched. Numerical settings are handled internally; no thread configuration is required.
