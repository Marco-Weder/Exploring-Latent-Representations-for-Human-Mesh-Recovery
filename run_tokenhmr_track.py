import sys
import os
import torch
import warnings
import urllib.request
import runpy

# 1. PyTorch 2.6+ Security Bypass
_original_load = torch.load
def _legacy_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _legacy_load
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. 403 Forbidden Bypass (Safe Class-based Spoofing)
class BrowserSafeRequest(urllib.request.Request):
    def __init__(self, *args, **kwargs):
        # Inject the User-Agent header during initialization
        headers = kwargs.get('headers', {})
        headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        kwargs['headers'] = headers
        super().__init__(*args, **kwargs)

# Replace the original class with our safe subclass
urllib.request.Request = BrowserSafeRequest

# 3. Set up the paths
submodule_root = os.path.abspath("external/tokenhmr")
os.chdir(submodule_root)
sys.path.insert(0, os.path.join(submodule_root, "tokenhmr"))

if __name__ == "__main__":
    print("🚀 Running TokenHMR Tracker (Safe Browser + PyTorch 2.6 Fix)...")
    sys.argv[0] = 'tokenhmr/track.py'
    runpy.run_path('tokenhmr/track.py', run_name='__main__')