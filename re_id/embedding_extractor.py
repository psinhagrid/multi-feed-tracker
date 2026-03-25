"""
Re-ID embedding extraction using fast-reid.
Extracts feature embeddings from person crops for re-identification.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

# Load .env so FAST_REID_PATH is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import cv2
import numpy as np

# Add fast-reid to PYTHONPATH if FAST_REID_PATH is set (e.g. from .env)
_fast_reid_path = os.getenv("FAST_REID_PATH")
if _fast_reid_path and Path(_fast_reid_path).exists():
    _path = str(Path(_fast_reid_path).resolve())
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultPredictor
    FASTREID_AVAILABLE = True
except ImportError:
    FASTREID_AVAILABLE = False
    DefaultPredictor = None
    get_cfg = None


class ReIDEmbeddingExtractor:
    """
    Extracts 2048-dim person re-identification embeddings using fast-reid.
    Designed for cropped person ROIs (e.g., from user-drawn box or detector).
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        device: str = "cuda",
    ):
        """
        Initialize the Re-ID predictor.

        Args:
            config_path: Path to fast-reid config YAML (e.g., configs/Market1501/bagtricks_R50.yml)
            weights_path: Path to model weights .pth file
            device: "cuda", "cpu", or "mps"
        """
        if not FASTREID_AVAILABLE:
            raise RuntimeError(
                "fast-reid is not installed. Clone it and run: pip install -r requirements.txt && python setup.py develop"
            )

        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Re-ID config not found: {config_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Re-ID weights not found: {weights_path}")

        self.cfg = get_cfg()
        self.cfg.merge_from_file(str(self.config_path))
        self.cfg.MODEL.WEIGHTS = str(self.weights_path)
        self.cfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(self.cfg)
        self.device = device
        print(f"Re-ID model loaded on {device}")

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single image (BGR, e.g. from cv2.imread).
        Background masking is expected to be done externally (YOLO-seg in app.py)
        before calling this method.

        Args:
            image: BGR image as numpy array (H, W, 3), ideally already masked.

        Returns:
            1D L2-normalised numpy array of shape (2048,)
        """
        import torch

        # Preprocess like fast-reid demo: BGR->RGB, resize, to tensor (1,C,H,W)
        image_rgb = image[:, :, ::-1]
        size_test = getattr(self.cfg.INPUT, "SIZE_TEST", (256, 128))
        if hasattr(size_test, "__iter__") and not isinstance(size_test, str):
            size_tuple = tuple(size_test[::-1]) if len(size_test) == 2 else (256, 128)
        else:
            size_tuple = (256, 128)
        resized = cv2.resize(image_rgb, size_tuple, interpolation=cv2.INTER_CUBIC)
        tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))[None]
        outputs = self.predictor(tensor)

        # Handle different output formats from fast-reid
        if hasattr(outputs, "cpu"):
            emb = outputs.cpu().numpy()
        elif isinstance(outputs, dict):
            if "outputs" in outputs:
                emb = outputs["outputs"]
            elif "feat" in outputs:
                emb = outputs["feat"]
            else:
                emb = list(outputs.values())[0]
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            emb = np.squeeze(emb)
        else:
            emb = np.array(outputs).squeeze()

        if emb.ndim > 1:
            emb = emb.squeeze()

        emb = emb.astype(np.float32)
        # Explicit L2 normalisation so stored embeddings are unit-norm and
        # cosine similarity == dot product (no re-normalisation needed at match time)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def extract_roi(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract embedding from a cropped region of a frame (no background masking).
        For masked extraction, mask the crop externally (YOLO-seg) then call extract().

        Args:
            frame: Full BGR frame
            bbox: (x, y, width, height) in pixels

        Returns:
            1D L2-normalised embedding array (2048,)
        """
        x, y, w, h = [int(v) for v in bbox]
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid ROI: bbox={bbox} produces empty crop")

        roi = frame[y1:y2, x1:x2]
        return self.extract(roi)


def get_reid_extractor(
    config_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Optional[ReIDEmbeddingExtractor]:
    """
    Lazy factory for ReIDEmbeddingExtractor.
    Uses config.py / env vars for paths.

    Returns:
        ReIDEmbeddingExtractor or None if fast-reid not available or paths not set
    """
    if not FASTREID_AVAILABLE:
        return None

    import config as app_config
    config_path = config_path or getattr(app_config, "REID_CONFIG_PATH", None)
    weights_path = weights_path or getattr(app_config, "REID_WEIGHTS_PATH", None)
    device = device or str(getattr(app_config, "REID_DEVICE", "cpu"))

    config_path = os.getenv("REID_CONFIG_PATH") or config_path
    weights_path = os.getenv("REID_WEIGHTS_PATH") or weights_path

    if not config_path or not weights_path:
        print("Re-ID: REID_CONFIG_PATH and REID_WEIGHTS_PATH must be set")
        return None

    if not Path(config_path).exists() or not Path(weights_path).exists():
        print(f"Re-ID: Config or weights file not found. Config={config_path}, Weights={weights_path}")
        return None

    return ReIDEmbeddingExtractor(
        config_path=config_path,
        weights_path=weights_path,
        device=device,
    )
