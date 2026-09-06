"""Explicit download helper; existing checkpoints are never overwritten."""
from pathlib import Path
from urllib.request import urlretrieve

path = Path("checkpoints/sam_vit_b_01ec64.pth")
if path.exists():
    raise SystemExit(f"Already exists: {path}")
path.parent.mkdir(exist_ok=True)
temporary = path.with_suffix(".download")
urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", temporary)
temporary.replace(path)
print(path)
