#!/usr/bin/env python3
"""Convert a folder of NPZ simulation outputs into color-mapped images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Disable GUI backends for headless usage
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - matplotlib import diagnostic
    raise SystemExit(
        "matplotlib is required for npz_to_images.py. Install it via pip install matplotlib"
    ) from exc


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 2D simulation fields from NPZ files to color images",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing NPZ files (e.g., results/run_name)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where rendered images will be written",
    )
    parser.add_argument(
        "--field",
        default="density",
        help="Which field inside each NPZ to render (default: density)",
    )
    parser.add_argument(
        "--colormap",
        default="magma",
        help="Matplotlib colormap name to use (default: magma)",
    )
    parser.add_argument(
        "--extension",
        default="png",
        choices=("png", "jpg", "jpeg", "tif", "tiff"),
        help="Image file extension to emit (default: png)",
    )
    parser.add_argument(
        "--global-normalize",
        action="store_true",
        help="Derive a single min/max across all frames for consistent exposure",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Override minimum value used for normalization",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Override maximum value used for normalization",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output directory",
    )
    return parser.parse_args(argv)


def discover_npz_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input directory not found: {folder}")
    return sorted(folder.glob("*.npz"))


def validate_field_shape(field_data: np.ndarray, filename: Path) -> None:
    if field_data.ndim != 2:
        raise ValueError(
            f"Field must be 2D for image export, got shape {field_data.shape} in {filename}"
        )


def compute_global_range(paths: Iterable[Path], field: str) -> tuple[float, float]:
    min_val = np.inf
    max_val = -np.inf
    for path in paths:
        with np.load(path) as npz:
            if field not in npz:
                raise KeyError(f"Field '{field}' missing in {path}")
            data = npz[field]
        validate_field_shape(data, path)
        min_val = min(min_val, float(np.min(data)))
        max_val = max(max_val, float(np.max(data)))
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        raise ValueError("Unable to compute finite global range")
    if min_val == max_val:
        max_val = min_val + 1e-6  # avoid divide-by-zero later
    return min_val, max_val


def render_image(
    data: np.ndarray,
    output_path: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, data, cmap=cmap, vmin=vmin, vmax=vmax)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    npz_files = discover_npz_files(args.input_dir)
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {args.input_dir}")

    if args.output_dir.exists() and not args.overwrite:
        existing = list(args.output_dir.glob(f"*.{args.extension}"))
        if existing:
            raise FileExistsError(
                f"Output directory {args.output_dir} already contains images. Use --overwrite to replace."
            )

    global_min = args.vmin
    global_max = args.vmax
    if args.global_normalize and (global_min is None or global_max is None):
        global_min, global_max = compute_global_range(npz_files, args.field)

    for path in npz_files:
        with np.load(path) as npz:
            if args.field not in npz:
                raise KeyError(f"Field '{args.field}' missing in {path}")
            field_data = npz[args.field]
        validate_field_shape(field_data, path)

        field_data = np.flip(field_data, axis=0)

        frame_min = float(np.min(field_data)) if args.vmin is None else args.vmin
        frame_max = float(np.max(field_data)) if args.vmax is None else args.vmax

        if args.global_normalize:
            frame_min = global_min if global_min is not None else frame_min
            frame_max = global_max if global_max is not None else frame_max

        if frame_min == frame_max:
            frame_max = frame_min + 1e-6

        output_name = path.stem + f".{args.extension}"
        output_path = args.output_dir / output_name
        render_image(field_data, output_path, args.colormap, frame_min, frame_max)
        print(f"Saved {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
