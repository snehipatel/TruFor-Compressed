import time, torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("D:/ml_project")
sys.path.insert(0, str(PROJECT_ROOT / "test_docker" / "src"))
from config import update_config, _C as config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

class DummyArgs:
    opts = None
update_config(config, DummyArgs())

model = confcmx(cfg=config)
weights_path = PROJECT_ROOT / "pretrained_models" / "trufor.pth.tar"
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Let's test with a 1024x1024 image and a 1600x1064 image
def run_size(w, h):
    # Ensure divisible by 8
    w = (w // 8) * 8
    h = (h // 8) * 8
    dummy = torch.randn(1, 3, h, w)
    t0 = time.time()
    with torch.no_grad():
        _, _, det, _ = model(dummy)
        score = float(torch.sigmoid(det).item())
    dt = time.time() - t0
    print(f"Size {w}x{h}: score={score:.4f}, time={dt:.2f}s")

print("Testing TruFor CPU inference speed at various sizes:")
run_size(384, 384)
run_size(768, 768)
run_size(1024, 1024)
