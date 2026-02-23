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