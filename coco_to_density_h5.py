"""COCO -> CSRNet density maps (.h5).

CSRNet (this repo) expects a per-image HDF5 file with a single dataset named
`density` that is a 2D float array. The sum of pixels should equal the object
count.

This script converts a COCO-style annotations JSON into CSRNet-style density
maps.

TenebrioVision notes:
- The provided COCO file includes `segmentation` polygons and `bbox` per larva.
- CSRNet needs point annotations; we derive one point per instance:
  - Prefer polygon centroid (largest polygon), else bbox center.

Example:
  python coco_to_density_h5.py \
    --coco-json ..\\TenebrioVision_Annotations.json \
    --images-dir ..\\TenebrioVision_Images \
    --output-dir ..\\ground_truth \
        --sigma-mode fixed --sigma 8.0

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    from scipy.spatial import KDTree
except Exception:  # pragma: no cover
    KDTree = None  # type: ignore


Point = Tuple[float, float]  # (x, y)


@dataclass(frozen=True)
class ImageInfo:
    image_id: int
    file_name: str
    width: int
    height: int


def _polygon_centroid_and_area(coords: Sequence[float]) -> Optional[Tuple[Point, float]]:
    """Return (centroid(x,y), area) for polygon coords [x1,y1,x2,y2,...]."""
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    xs = np.asarray(coords[0::2], dtype=np.float64)
    ys = np.asarray(coords[1::2], dtype=np.float64)

    x0 = xs
    y0 = ys
    x1 = np.roll(xs, -1)
    y1 = np.roll(ys, -1)

    cross = x0 * y1 - x1 * y0
    area2 = cross.sum()
    if abs(area2) < 1e-9:
        return None

    cx = ((x0 + x1) * cross).sum() / (3.0 * area2)
    cy = ((y0 + y1) * cross).sum() / (3.0 * area2)
    return (float(cx), float(cy)), float(area2 / 2.0)


def _annotation_point(ann: dict) -> Optional[Point]:
    """Derive a single point per COCO annotation."""
    seg = ann.get("segmentation")
    if isinstance(seg, list) and seg:
        best: Optional[Tuple[Point, float]] = None
        for poly in seg:
            if not isinstance(poly, list):
                continue
            out = _polygon_centroid_and_area(poly)
            if out is None:
                continue
            centroid, area = out
            if best is None or abs(area) > abs(best[1]):
                best = (centroid, area)
        if best is not None:
            return best[0]

    bbox = ann.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, w, h = bbox
        return (float(x) + float(w) / 2.0, float(y) + float(h) / 2.0)

    kpts = ann.get("keypoints")
    if isinstance(kpts, list) and len(kpts) >= 3:
        x, y, v = kpts[0], kpts[1], kpts[2]
        if float(v) > 0:
            return (float(x), float(y))

    return None


def _clamp_point_to_image(pt: Point, width: int, height: int) -> Optional[Tuple[int, int]]:
    x, y = pt
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= width or yi >= height:
        return None
    return xi, yi


def _sigma_adaptive(points: np.ndarray, height: int, width: int, min_sigma: float, max_sigma: float) -> np.ndarray:
    """Adaptive sigma per point using KNN distances (ShanghaiTech-style)."""
    n = points.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        sigma = (height + width) / 2.0 / 4.0
        return np.asarray([float(np.clip(sigma, min_sigma, max_sigma))], dtype=np.float32)

    if KDTree is None:
        raise RuntimeError("scipy is required for adaptive sigma mode (scipy.spatial.KDTree)")

    tree = KDTree(points, leafsize=2048)
    k = min(4, n)
    dists, _ = tree.query(points, k=k)

    neighbor_count = max(1, k - 1)
    scale = 0.1 * (3.0 / neighbor_count)
    sigmas = scale * dists[:, 1:k].sum(axis=1)

    sigmas = np.clip(sigmas, min_sigma, max_sigma)
    return sigmas.astype(np.float32)


def _gaussian_kernel2d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    size = radius * 2 + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma)))
    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel


def _stamp_gaussian(density: np.ndarray, x: int, y: int, sigma: float, kernel_cache: Dict[float, np.ndarray]) -> None:
    sigma_key = round(float(sigma), 2)
    kernel = kernel_cache.get(sigma_key)
    if kernel is None:
        kernel = _gaussian_kernel2d(float(sigma))
        kernel_cache[sigma_key] = kernel

    kh, kw = kernel.shape
    radius_y = kh // 2
    radius_x = kw // 2

    h, w = density.shape
    y0 = max(0, y - radius_y)
    y1 = min(h, y + radius_y + 1)
    x0 = max(0, x - radius_x)
    x1 = min(w, x + radius_x + 1)

    ky0 = y0 - (y - radius_y)
    ky1 = ky0 + (y1 - y0)
    kx0 = x0 - (x - radius_x)
    kx1 = kx0 + (x1 - x0)

    patch = kernel[ky0:ky1, kx0:kx1]
    s = float(patch.sum())
    if s > 0:
        patch = patch / s
    density[y0:y1, x0:x1] += patch


def build_density_map(
    height: int,
    width: int,
    points_xy: Sequence[Tuple[int, int]],
    sigma_mode: str,
    sigma: float,
    min_sigma: float,
    max_sigma: float,
) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    if not points_xy:
        return density

    points = np.asarray(points_xy, dtype=np.float32)

    if sigma_mode == "fixed":
        kernel_cache: Dict[float, np.ndarray] = {}
        for x, y in points_xy:
            _stamp_gaussian(density, x, y, float(sigma), kernel_cache)
        s = float(density.sum())
        if s > 0:
            density *= (len(points_xy) / s)
        return density

    if sigma_mode == "adaptive":
        sigmas = _sigma_adaptive(points, height=height, width=width, min_sigma=min_sigma, max_sigma=max_sigma)
        kernel_cache = {}
        for (x, y), sgm in zip(points_xy, sigmas):
            _stamp_gaussian(density, int(x), int(y), float(sgm), kernel_cache)
        s = float(density.sum())
        if s > 0:
            density *= (len(points_xy) / s)
        return density

    raise ValueError(f"Unknown sigma_mode: {sigma_mode}")


def _load_coco(coco_json: Path) -> Tuple[List[ImageInfo], Dict[int, List[Point]]]:
    with coco_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images: List[ImageInfo] = []
    for im in data.get("images", []):
        try:
            images.append(
                ImageInfo(
                    image_id=int(im["id"]),
                    file_name=str(im["file_name"]),
                    width=int(im.get("width", 0)),
                    height=int(im.get("height", 0)),
                )
            )
        except Exception:
            continue

    points_by_image: Dict[int, List[Point]] = {}
    for ann in data.get("annotations", []):
        try:
            image_id = int(ann["image_id"])
        except Exception:
            continue
        pt = _annotation_point(ann)
        if pt is None:
            continue
        points_by_image.setdefault(image_id, []).append(pt)

    return images, points_by_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert COCO annotations to CSRNet density .h5 files")
    parser.add_argument("--coco-json", type=Path, required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing the images referenced by file_name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write .h5 density maps (default: <images-dir>/../ground_truth)",
    )
    parser.add_argument(
        "--sigma-mode",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="How to choose Gaussian spread (default: fixed)",
    )
    parser.add_argument("--sigma", type=float, default=8.0, help="Fixed sigma (only used when --sigma-mode fixed)")
    parser.add_argument("--min-sigma", type=float, default=2.0, help="Clamp minimum sigma in adaptive mode")
    parser.add_argument("--max-sigma", type=float, default=50.0, help="Clamp maximum sigma in adaptive mode")
    parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="Process only the first N images (0 = all). Useful for quick tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .h5 files (default: skip if exists)",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "none"],
        default="gzip",
        help="H5 compression (default: gzip).",
    )

    args = parser.parse_args()

    coco_json: Path = args.coco_json
    images_dir: Path = args.images_dir
    output_dir: Path = args.output_dir if args.output_dir is not None else (images_dir.parent / "ground_truth")

    if not coco_json.exists():
        raise FileNotFoundError(str(coco_json))
    if not images_dir.exists():
        raise FileNotFoundError(str(images_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    images, points_by_image = _load_coco(coco_json)
    if args.limit_images and args.limit_images > 0:
        images = images[: args.limit_images]

    missing_images = 0
    written = 0
    skipped = 0

    for idx, im in enumerate(images, start=1):
        img_path = images_dir / im.file_name
        if not img_path.exists():
            missing_images += 1
            if missing_images <= 5:
                print(f"[missing] {img_path}")
            continue

        height = int(im.height)
        width = int(im.width)
        if height <= 0 or width <= 0:
            from PIL import Image

            with Image.open(img_path) as pil:
                width, height = pil.size

        raw_points = points_by_image.get(im.image_id, [])
        points_xy: List[Tuple[int, int]] = []
        for pt in raw_points:
            clamped = _clamp_point_to_image(pt, width=width, height=height)
            if clamped is not None:
                points_xy.append(clamped)

        density = build_density_map(
            height=height,
            width=width,
            points_xy=points_xy,
            sigma_mode=args.sigma_mode,
            sigma=float(args.sigma),
            min_sigma=float(args.min_sigma),
            max_sigma=float(args.max_sigma),
        )

        out_path = output_dir / (Path(im.file_name).stem + ".h5")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with h5py.File(out_path, "w") as hf:
            if args.compression == "gzip":
                hf.create_dataset("density", data=density, dtype="float32", compression="gzip", compression_opts=4)
            else:
                hf.create_dataset("density", data=density, dtype="float32")

        written += 1

        if idx % 25 == 0 or idx == 1:
            expected = len(points_xy)
            actual = float(density.sum())
            print(f"[{idx}/{len(images)}] wrote {out_path.name}  count={expected}  sum={actual:.3f}")

    print(
        "Done. "
        f"images={len(images)} written={written} skipped={skipped} missing_images={missing_images} "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
