import os
import math
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
from itertools import combinations

"""
Ruslan you can use a threshold like 0.45/0.5 and it should work correctly for matching.
"""

def load_bbox(directory: str) -> dict:
    """Read width/height from bbox.txt in the given capture directory."""
    bbox = {}
    with open(os.path.join(directory, "bbox.txt")) as f:
        for line in f:
            k, v = line.strip().split("=")
            bbox[k] = int(v)
    return bbox


def load_skeleton(directory: str) -> list[tuple[float, float]]:
    """Return 5 keypoints as list of (x, y) floats: kp0–kp3 are finger bases, kp4 is wrist."""
    vals = {}
    with open(os.path.join(directory, "skeleton.txt")) as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                idx, v = line.split("=")
                vals[int(idx.strip("[]"))] = float(v)
    return [(vals[i * 2], vals[i * 2 + 1]) for i in range(5)]


def canonical_angle(kps: list[tuple[float, float]]) -> float:
    """CCW degrees to rotate so fingers always point up, wrist down."""
    finger_cx = np.mean([kps[i][0] for i in range(4)])
    finger_cy = np.mean([kps[i][1] for i in range(4)])
    wrist_x, wrist_y = kps[4]
    dx = finger_cx - wrist_x
    dy = finger_cy - wrist_y
    return np.degrees(np.arctan2(dx, -dy))  # 0° when fingers directly above wrist


def load_crop_array(directory: str) -> np.ndarray:
    """Load IR crop, orient upright via skeleton, resize to 128×128, return (1,1,128,128) float32.

    Crop region is derived from skeleton keypoints so that the same anatomical
    palm region (wrist → finger bases) is captured consistently across sessions,
    regardless of how the bounding-box detector performed.
    """
    bbox = load_bbox(directory)
    W, H = bbox["width"], bbox["height"]

    raw = np.frombuffer(open(os.path.join(directory, "ir.raw"), "rb").read(), dtype=np.uint8).reshape(H, W)
    kps = load_skeleton(directory)

    finger_xs = [kps[i][0] for i in range(4)]
    finger_ys = [kps[i][1] for i in range(4)]
    wrist_x, wrist_y = kps[4]

    cx = (np.mean(finger_xs) + wrist_x) / 2.0
    cy = (np.mean(finger_ys) + wrist_y) / 2.0

    # Use Euclidean wrist→finger-centroid distance so the crop is correct
    # for any palm orientation (image-coordinate projections underestimate
    # the true palm length when the palm is rotated).
    finger_cx = np.mean(finger_xs)
    finger_cy = np.mean(finger_ys)
    palm_len  = np.hypot(finger_cx - wrist_x, finger_cy - wrist_y)
    side      = int(palm_len * 1.4)  # 1.4 leaves room for fingertips above centroid

    # Oversized crop so rotation never clips the palm (diagonal = side√2)
    large = int(side * math.sqrt(2)) + 4
    x1, y1 = int(cx - large / 2), int(cy - large / 2)
    x2, y2 = x1 + large, y1 + large

    crop = np.zeros((large, large), dtype=np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)
    crop[sy1 - y1 : sy1 - y1 + (sy2 - sy1),
         sx1 - x1 : sx1 - x1 + (sx2 - sx1)] = raw[sy1:sy2, sx1:sx2]

    img = Image.fromarray(crop)
    img = img.rotate(canonical_angle(kps), resample=Image.BICUBIC, expand=False)

    # Centre-crop back to side × side
    m = (large - side) // 2
    img = img.crop((m, m, m + side, m + side))
    img = img.resize((128, 128), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    mask = arr > 0
    if mask.sum() > 1:
        mu, s = arr[mask].mean(), arr[mask].std()
        arr[mask] = (arr[mask] - mu) / (s + 1e-6)

    return arr[np.newaxis, np.newaxis, :, :]


def embed(session: ort.InferenceSession, arr: np.ndarray) -> np.ndarray:
    """Run a forward pass and return the L2-normalised feature vector."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    feat = session.run([output_name], {input_name: arr})[0].squeeze()
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def run(model_path: str, capture_dirs: list[str]) -> None:
    """Embed each capture directory and print all pairwise cosine similarities."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    embeddings = {}
    for d in capture_dirs:
        name = os.path.basename(os.path.normpath(d))
        embeddings[name] = embed(session, load_crop_array(d))

    print(f"\n{'Pair':<38}  {'Cosine Sim':>10}")
    print("-" * 52)
    for n1, n2 in combinations(embeddings, 2):
        sim = cosine(embeddings[n1], embeddings[n2])
        print(f"{n1} vs {n2:<20}  {sim:>10.4f}")


def main() -> None:
    """Entry point: parse arguments and run pairwise palm comparison."""
    parser = argparse.ArgumentParser(description="Palm-net ONNX inference engine")
    parser.add_argument("--model", required=True, help="Path to palm_net ONNX model file")
    parser.add_argument("captures", nargs="+", help="Capture directories to embed and compare")
    args = parser.parse_args()
    run(args.model, args.captures)


if __name__ == "__main__":
    main()
