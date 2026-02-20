"""
Palm matcher — standalone application using CCNet (IEEE TIFS 2023).

Usage as CLI:
    python matcher.py <capture_dir_A> <capture_dir_B> [--weights /path/to/tongji_weights.pth]

Usage as library:
    from palm_matching.matcher import PalmMatcher
    m = PalmMatcher()
    result = m.match("path/to/capture_A", "path/to/capture_B")
    print(result)          # {"similarity": 0.85, "match": True}

Each capture directory must contain:
    ir.raw        — raw uint8 grayscale image (width × height bytes)
    bbox.txt      — width=, height=, bbox_x=, bbox_y=, bbox_w=, bbox_h=
    skeleton.txt  — count=10, [0]=...[9]= (5 keypoints × x,y)
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from ccnet import ccnet as CCNet

_HERE    = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.path.join(_HERE, "tongji_weights.pth")

# Cosine similarity threshold: >= MATCH_THRESHOLD → same person.
# Calibrated on 5-capture dataset:
#   same-person pairs:      0.848 – 0.991
#   different-person pairs: 0.694 – 0.722
MATCH_THRESHOLD = 0.80


def _load_bbox(d: str) -> dict:
    bbox = {}
    with open(os.path.join(d, "bbox.txt")) as f:
        for line in f:
            k, v = line.strip().split("=")
            bbox[k] = int(v)
    return bbox


def _load_skeleton(d: str):
    """Return 5 keypoints as [(x0,y0) … (x4,y4)]; kp4 is the wrist."""
    vals = {}
    with open(os.path.join(d, "skeleton.txt")) as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                idx, v = line.split("=")
                vals[int(idx.strip("[]"))] = float(v)
    return [(vals[i * 2], vals[i * 2 + 1]) for i in range(5)]


def _is_upside_down(kps) -> bool:
    """True when wrist (kp4) is above the four finger-base points in image coords."""
    finger_y = np.mean([kps[i][1] for i in range(4)])
    return kps[4][1] < finger_y


def load_crop_tensor(capture_dir: str) -> torch.Tensor:
    """
    Load and preprocess one capture:
    1. Crop palm region from ir.raw using bbox.txt
    2. Rotate 180° if skeleton says palm is upside-down
    3. Resize to 128×128 grayscale
    4. Return (1, 1, 128, 128) float32 tensor in [0, 1]
    """
    bbox = _load_bbox(capture_dir)
    W, H = bbox["width"], bbox["height"]
    x, y, w, h = bbox["bbox_x"], bbox["bbox_y"], bbox["bbox_w"], bbox["bbox_h"]

    raw = np.frombuffer(
        open(os.path.join(capture_dir, "ir.raw"), "rb").read(), dtype=np.uint8
    ).reshape(H, W)

    img = Image.fromarray(raw[y : y + h, x : x + w])

    if _is_upside_down(_load_skeleton(capture_dir)):
        img = img.rotate(180)

    img = img.resize((128, 128), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,128,128)


def load_model(weights_path: str = _WEIGHTS) -> CCNet:
    """Load CCNet with Tongji pretrained weights, ready for inference."""
    model = CCNet(num_classes=600, weight=0.8)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def embed(model: CCNet, capture_dir: str) -> np.ndarray:
    """Return L2-normalised 2048-d feature vector for a capture directory."""
    tensor = load_crop_tensor(capture_dir)
    feat   = model.getFeatureCode(tensor)   # (1, 2048), already L2-normalised
    return feat.squeeze().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class PalmMatcher:
    """
    Reusable matcher — load the model once, call match() many times.

    Example:
        matcher = PalmMatcher()
        result  = matcher.match("captures/ruslan", "captures/first-capture")
        # → {"similarity": 0.848, "match": True}
    """

    def __init__(self, weights_path: str = _WEIGHTS, threshold: float = MATCH_THRESHOLD):
        self.threshold = threshold
        self.model     = load_model(weights_path)

    def embed(self, capture_dir: str) -> np.ndarray:
        return embed(self.model, capture_dir)

    def match(self, dir_a: str, dir_b: str) -> dict:
        """
        Compare two capture directories.

        Returns:
            {
                "similarity": float,   # cosine similarity in [0, 1]
                "match":      bool,    # True if similarity >= threshold
                "threshold":  float,
            }
        """
        emb_a = self.embed(dir_a)
        emb_b = self.embed(dir_b)
        sim   = cosine_similarity(emb_a, emb_b)
        return {
            "similarity": round(sim, 4),
            "match":      sim >= self.threshold,
            "threshold":  self.threshold,
        }


def _cli():
    parser = argparse.ArgumentParser(description="Match two palm captures with CCNet.")
    parser.add_argument("capture_a", help="Path to first capture directory")
    parser.add_argument("capture_b", help="Path to second capture directory")
    parser.add_argument(
        "--weights", default=_WEIGHTS,
        help=f"Path to CCNet weights (default: {_WEIGHTS})"
    )
    parser.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Match threshold (default: {MATCH_THRESHOLD})"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.weights}...")
    matcher = PalmMatcher(weights_path=args.weights, threshold=args.threshold)

    result = matcher.match(args.capture_a, args.capture_b)

    print(f"\nCapture A : {args.capture_a}")
    print(f"Capture B : {args.capture_b}")
    print(f"Similarity: {result['similarity']:.4f}  (threshold: {result['threshold']})")
    print(f"Result    : {'MATCH' if result['match'] else 'NO MATCH'}")
    sys.exit(0 if result["match"] else 1)


if __name__ == "__main__":
    _cli()
