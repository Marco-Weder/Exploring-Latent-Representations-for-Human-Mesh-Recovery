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

# Install remaining requirements
pip install -r requirements.txt
```
## Running Demos
Once the environment is configured, verify the installation by running the demo script:
```bash
python run_tokenhmr_demo.py
```