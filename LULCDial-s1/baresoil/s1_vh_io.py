"""Read Sentinel-1 VH patches and convert to EarthDial-compatible float images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image # array --> image

try:
    import rasterio # geotiff library
except ImportError as exc:
    raise ImportError("Install rasterio: python -m pip install rasterio") from exc

#.tiff -->
def read_s1_vh_db(tif_path: str | Path) -> np.ndarray:
    """Load VH backscatter (dB) from a MultiSenGE S1 patch GeoTIFF."""
    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        if src.count < 2:
            vh = src.read(1).astype(np.float32)
        else:
            vh = src.read(2).astype(np.float32)

    if vh.ndim != 2:
        raise ValueError(f"Expected 2D VH array, got shape {vh.shape} from {tif_path}")

    # linear to dB
    if np.nanmax(vh) < 1.0 and np.nanmin(vh) >= 0:
        vh = 10.0 * np.log10(np.clip(vh, 1e-10, None))

    return np.clip(vh, -50.0, 10.0) # normalise

#float32 --> shard image
def vh_db_to_pil(vh_db: np.ndarray) -> Image.Image:
    """Float32 PIL image with dB values — matches EarthDial S1 shard convention."""
    arr = np.asarray(vh_db, dtype=np.float32) # check: Y?: space; earthdial convention
    if arr.shape != (256, 256):
        raise ValueError(f"Expected 256×256 patch, got {arr.shape}")
    return Image.fromarray(arr, mode="F")

#float32 --> preview image
def vh_db_to_preview_png(vh_db: np.ndarray, out_path: str | Path) -> None:
    """Human-viewable 8-bit stretch (for debugging only, not for training)."""
    vh = np.clip(vh_db, -35.0, 5.0)
    vis = ((vh + 35.0) / 40.0 * 255.0).astype(np.uint8)
    Image.fromarray(vis, mode="L").convert("RGB").save(out_path)
