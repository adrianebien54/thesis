"""Verify CSRNet density maps against COCO annotation counts.

Checks that, for a random sample of images, the sum of the density map equals
(the number of COCO annotations for that image).

Usage:
  python verify_density_vs_coco.py \
    --coco-json TenebrioVision_Annotations.json \
    --ground-truth-dir ground_truth \
    --samples 15 \
    --seed 1337

"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description="Verify density map sums vs COCO counts")
    p.add_argument("--coco-json", type=Path, required=True)
    p.add_argument("--ground-truth-dir", type=Path, default=Path("ground_truth"))
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--tolerance", type=float, default=1e-2)
    args = p.parse_args()

    coco = json.load(args.coco_json.open("r", encoding="utf-8"))

    id_to_fn = {int(im["id"]): str(im["file_name"]) for im in coco.get("images", [])}

    counts: dict[int, int] = {}
    for ann in coco.get("annotations", []):
        iid = int(ann["image_id"])
        counts[iid] = counts.get(iid, 0) + 1

    image_ids = list(id_to_fn.keys())
    if not image_ids:
        raise RuntimeError("No images found in COCO JSON")

    k = min(int(args.samples), len(image_ids))
    rng = random.Random(int(args.seed))
    sample_ids = rng.sample(image_ids, k)

    missing_h5 = []
    mismatches = []
    oks = []

    for iid in sample_ids:
        fn = id_to_fn[iid]
        h5_path = args.ground_truth_dir / (Path(fn).stem + ".h5")
        if not h5_path.exists():
            missing_h5.append(fn)
            continue

        with h5py.File(h5_path, "r") as f:
            d = np.asarray(f["density"], dtype=np.float64)

        expected = int(counts.get(iid, 0))
        s = float(d.sum())
        if abs(s - expected) > float(args.tolerance):
            mismatches.append((fn, expected, s, d.shape))
        else:
            oks.append((fn, expected, s, d.shape))

    print(f"Sample size: {k}")
    print(f"Missing h5: {len(missing_h5)}")
    print(f"Mismatches: {len(mismatches)}")

    if missing_h5:
        print("Missing examples:")
        for fn in missing_h5[:5]:
            print("  ", fn)

    if mismatches:
        print("Mismatch examples:")
        for fn, expected, s, shape in mismatches[:10]:
            print("  ", fn, "expected", expected, "sum", round(s, 4), "shape", shape)

    print("OK examples:")
    for fn, expected, s, shape in oks[:5]:
        print("  ", fn, "count", expected, "sum", round(s, 4), "shape", shape)

    return 0 if (not missing_h5 and not mismatches) else 2


if __name__ == "__main__":
    raise SystemExit(main())
