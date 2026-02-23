import sys
import os
import torch
import warnings

# 1. Apply the PyTorch 2.6+ security bypass in-memory
_original_load = torch.load
def _legacy_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _legacy_load
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Set up the paths
# Change directory so relative paths (like 'demo_sample/images/') work perfectly
submodule_root = os.path.abspath("external/tokenhmr")
os.chdir(submodule_root)

# CRITICAL FIX: Add the inner 'tokenhmr' folder to sys.path so 'import lib' works inside demo.py
inner_tokenhmr = os.path.join(submodule_root, "tokenhmr")
sys.path.insert(0, inner_tokenhmr)

# 3. Import the original submodule code directly
import demo

if __name__ == "__main__":
    print("🚀 Running TokenHMR via Thesis Wrapper (PyTorch 2.6+ Safe Mode)...")
    demo.main()