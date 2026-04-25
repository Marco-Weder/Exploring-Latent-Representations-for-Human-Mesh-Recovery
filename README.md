# Exploring Latent Representations for Human Mesh Recovery

This repository contains the code for my Bachelor's thesis at ETH Zurich. The project builds upon the **TokenHMR** framework to explore latent representations for robust human mesh recovery.

## Installation & Setup

### 1. Clone the Repository
This project uses **Git submodules**. Ensure you clone the repository and its dependencies recursively:

```bash
git clone --recurse-submodules [https://github.com/Marco-Weder/Exploring-Latent-Representations-for-Human-Mesh-Recovery.git](https://github.com/Marco-Weder/Exploring-Latent-Representations-for-Human-Mesh-Recovery.git)
cd Exploring-Latent-Representations-for-Human-Mesh-Recovery
```

### 2. Environment Setup.
To ensure compatibility with legacy mesh-processing libraries (like chumpy) and modern high-performance hardware (RTX 5090 / Blackwell), we use Python 3.10.
```bash
# Create and activate the conda environment
conda create -n thesis-HMR python=3.10 -y
conda activate thesis-HMR

# Install build-critical dependencies manually 
# (This prevents the "No module named pip" error during the chumpy build)
pip install "numpy<1.24.0" setuptools wheel
pip install chumpy==0.70 --no-build-isolation

# Install remaining requirements (using cuda 12.8)
pip install -r requirements.txt

# Install detectron2 (set cuda-12.X to your CUDA version installed with PyTorch)
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2
```
## Preparing Data for Basic Setup [required for demo]
All the files are uploaded to project webpage. Downloading works only after you register and agree to the licenses. Use the script fetch_demo_data.sh to download files needed for running demo. This includes SMPL and SMPLH body models, latest TokenHMR and Tokenization checkpoints. For training and evaluation, refer to respective sections.

```bash
bash ./fetch_demo_data.sh
```
PHALP needs SMPL neutral model for running video demo. Copy the model to appropriate location.
```bash
mkdir -p ~/.cache/phalp/3D/models/smpl/
cp data/body_models/smpl/SMPL_NEUTRAL.pkl $HOME/.cache/phalp/3D/models/smpl/
```

## Running Demos
Once the environment is configured, verify the installation by running the demo script:
```bash
python run_tokenhmr_demo.py
```

## Tokenizer Training
Run tokenizer training from the tokenization directory.

```bash
cd external/tokenhmr/tokenization
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo.yaml
```

### Available tokenizer configs
- `configs/tokenizer_amass_moyo.yaml`
- `configs/tokenizer_amass_moyo_original.yaml`

### Resuming interrupted training

If your training was interrupted and you want to continue:

**Option 1: Resume from latest checkpoint (continues current training)**
```bash
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo.yaml --resume_training --resume_pth output/[experiment_name]/latest_checkpoint.pth
```

**Option 2: Resume from best checkpoint with extended training (recommended if you see overfitting)**
If validation diverged after a certain point (e.g., best at iter 25k but training continued), resume from best checkpoint:
```bash
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo.yaml --resume_training --resume_pth output/[experiment_name]/best_net.pth
```

**How resumption works:**
- Full training state is saved at each validation to `latest_checkpoint.pth`
- This includes: model weights, optimizer Adam state, LR scheduler state, and iteration counter
- `best_net.pth` still stores only the best model weights for evaluation/inference
- When resuming from `latest_checkpoint.pth`, the LR schedule continues smoothly from where it was paused

**Finding your checkpoint paths:**
- Latest: `output/tokenizer_amass_moyo_ID00_*/latest_checkpoint.pth`
- Best: `output/tokenizer_amass_moyo_ID00_*/best_net.pth`