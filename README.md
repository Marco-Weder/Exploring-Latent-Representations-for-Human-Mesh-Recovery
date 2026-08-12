# Exploring Latent Representations for Human Mesh Recovery

Code for my bachelor's thesis at ETH Zurich. The thesis studies the **pose tokenizer** as
the interface between pose reconstruction and image-based prediction: how a discrete pose
representation should be built, and how an image model should be trained to predict its
tokens. It builds on [TokenHMR](https://tokenhmr.is.tue.mpg.de).

To reproduce a specific figure, table or number from the thesis, see
**[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

## What is mine and what is upstream

The modified TokenHMR lives in a fork, pinned here as a submodule:

| Branch of [Marco-Weder/TokenHMR](https://github.com/Marco-Weder/TokenHMR) | Contents |
| --- | --- |
| `main` | pristine upstream TokenHMR, commit `ededd631` |
| `thesis` | the same code plus my changes (what this repository pins) |

So the complete set of changes made for this thesis is exactly:

```bash
git -C external/tokenhmr diff main..thesis
```

`external/PHALP`, `external/MoRo` and `external/VQ-HPS` are **not** vendored here. PHALP is
installed with pip (below). MoRo and VQ-HPS were read as references while implementing the
Gumbel decode and the classification setting, but no code from either is imported.

## Setup

### 1. Clone

```bash
git clone --recurse-submodules https://github.com/Marco-Weder/Exploring-Latent-Representations-for-Human-Mesh-Recovery.git
cd Exploring-Latent-Representations-for-Human-Mesh-Recovery
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Environment

Python 3.10, for compatibility with legacy mesh-processing libraries (chumpy) on modern
hardware.

```bash
conda create -n thesis-HMR python=3.10 -y
conda activate thesis-HMR

# Build-critical dependencies first; this avoids the "No module named pip"
# error while chumpy builds.
pip install "numpy<1.24.0" setuptools wheel
pip install chumpy==0.70 --no-build-isolation

pip install -r requirements.txt

# detectron2 (set CUDA_HOME to the CUDA version your PyTorch was built against)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2

# PHALP, needed only for the video/tracking demo
pip install git+https://github.com/brjathu/PHALP
```

`stable_thesis_requirements.txt` is a frozen snapshot of the environment the reported
results were produced in, should `requirements.txt` drift.

### 3. Data and body models

Downloading requires registering at <https://tokenhmr.is.tue.mpg.de> and accepting the
licences. This fetches SMPL and SMPL-H body models and the TokenHMR / tokenization
checkpoints:

```bash
cd external/tokenhmr
bash ./fetch_demo_data.sh
```

The video demo also needs the neutral SMPL model in PHALP's cache:

```bash
mkdir -p ~/.cache/phalp/3D/models/smpl/
cp external/tokenhmr/data/body_models/smpl/SMPL_NEUTRAL.pkl ~/.cache/phalp/3D/models/smpl/
```

Tokenizer training additionally needs AMASS and MOYO, and downstream training needs the
image datasets. Both are described in the upstream TokenHMR README
(`external/tokenhmr/README.md`).

## Demos

```bash
python run_tokenhmr_demo.py     # single images
python run_tokenhmr_track.py    # video, via PHALP
```

## Stage 1: training a pose tokenizer

```bash
cd external/tokenhmr/tokenization
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo_fsq.yaml
```

`tokenizer_amass_moyo_fsq.yaml` is the tokenizer carried through the thesis: a transformer
encoder-decoder with an FSQ quantizer, 160 latent tokens, levels `[8,8,6,5]` (1920 codes,
`d = 4`). The eight tokenizers compared in the ablation are:

| Config | Quantizer | `d` | `K` |
| --- | --- | --- | --- |
| `tokenizer_amass_moyo_original.yaml` | EMA (ℓ2), convolutional | 256 | 2048 |
| `tokenizer_amass_moyo_transformer.yaml` | EMA (cosine) | 256 | 2048 |
| `tokenizer_amass_moyo_transformer_masked.yaml` | EMA (cosine), skeleton-masked | 256 | 2048 |
| `tokenizer_amass_moyo_transformer_dim4.yaml` | EMA (cosine) | 4 | 2048 |
| `tokenizer_amass_moyo_transformer_dim2.yaml` | EMA (cosine) | 2 | 2048 |
| `tokenizer_amass_moyo_fsq.yaml` | FSQ `[8,8,6,5]` | 4 | 1920 |
| `tokenizer_amass_moyo_fsq_16k.yaml` | FSQ `[8,8,8,6,5]` | 5 | 15360 |

The ℓ2 transformer row of the ablation is this same transformer config with
`DIST_METRIC: 'l2'`.

### Resuming an interrupted run

Full training state (weights, optimiser, LR scheduler, iteration counter) is written to
`latest_checkpoint.pth` at every validation, so the schedule continues smoothly:

```bash
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo_fsq.yaml \
    --resume_training --resume_pth output/<experiment_name>/latest_checkpoint.pth
```

`best_net.pth` holds only the best weights, for evaluation and inference. Resuming from it
instead restarts the schedule, which is what you want if validation diverged partway.

## Stage 2: training the downstream model

```bash
cd external/tokenhmr
python tokenhmr/train.py experiment=tokenhmr_additive
```

Experiment configs are Hydra configs under
`tokenhmr/lib/configs_hydra/experiment/`. Each names the tokenizer checkpoint it freezes as
its pose head, via `TOKENIZER_CHECKPOINT_PATH`; point that at your own stage-1 output.

## Repository layout

```
.
├── docs/REPRODUCE.md                  how to reproduce each thesis result
├── run_tokenhmr_demo.py               image demo wrapper
├── run_tokenhmr_track.py              video demo wrapper (PHALP)
├── requirements.txt                   environment
├── stable_thesis_requirements.txt     frozen snapshot of the reported environment
└── external/tokenhmr/                 submodule: Marco-Weder/TokenHMR, branch `thesis`
    ├── tokenization/                  stage 1: pose tokenizer + latent-space analysis
    ├── tokenhmr/                      stage 2: downstream model + token analysis
    └── thesis_figures/                scripts producing the thesis figures
```

## Acknowledgements

Built on [TokenHMR](https://github.com/saidwivedi/TokenHMR) (Dwivedi et al., CVPR 2024) and
[HMR2.0](https://github.com/shubham-goel/4D-Humans) (Goel et al., ICCV 2023). The
classification setting takes its idea from
[VQ-HPS](https://github.com/g-fiche/VQ-HPS) (Fiche et al., ECCV 2024) and GenHMR
(Saleem et al., AAAI 2025); the finite scalar quantizer is from
[FSQ](https://arxiv.org/abs/2309.15505) (Mentzer et al., ICLR 2024).
