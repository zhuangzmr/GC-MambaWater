#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare a binary water-segmentation dataset into the VOC-style layout used by this project.

Output structure:
    VOCdevkit/
        VOC2007/
            JPEGImages/
            SegmentationClass/
            ImageSets/
                Segmentation/
                    train.txt
                    val.txt

This script only prepares the dataset format and train/val split.
The geometric data augmentation described in the paper
(rotation, scaling, and random cropping) is already implemented online in
`utils/dataloader.py` during training, so it is intentionally not baked into
this preprocessing script.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image/mask dataset into VOCdevkit/VOC2007 format."
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--masks-dir", required=True, help="Directory containing source masks.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root directory. The script will create VOCdevkit/VOC2007 under this path.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio. Default: 0.1",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of converting them to RGB JPG. Masks are always saved as PNG.",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=int,
        default=0,
        help=(
            "Pixels with values greater than this threshold are treated as foreground "
            "and written as 255 in the output mask. Default: 0"
        ),
    )
    return parser.parse_args()


def list_files(root: Path, valid_exts: Iterable[str]) -> List[Path]:
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(files)


def match_image_mask_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path, str]]:
    image_map = {p.stem: p for p in list_files(images_dir, IMAGE_EXTS)}
    mask_map = {p.stem: p for p in list_files(masks_dir, MASK_EXTS)}

    common = sorted(set(image_map) & set(mask_map))
    if not common:
        raise RuntimeError(
            f"No matched image/mask pairs found between {images_dir} and {masks_dir}."
        )

    missing_images = sorted(set(mask_map) - set(image_map))
    missing_masks = sorted(set(image_map) - set(mask_map))
    if missing_images:
        print(f"[Warn] {len(missing_images)} masks have no matching image and will be skipped.")
    if missing_masks:
        print(f"[Warn] {len(missing_masks)} images have no matching mask and will be skipped.")

    return [(image_map[stem], mask_map[stem], stem) for stem in common]


def ensure_dirs(output_root: Path) -> Tuple[Path, Path, Path]:
    voc_root = output_root / "VOCdevkit" / "VOC2007"
    jpeg_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"
    split_dir = voc_root / "ImageSets" / "Segmentation"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    return jpeg_dir, mask_dir, split_dir


def save_image(src: Path, dst: Path, copy_images: bool) -> None:
    if copy_images and src.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(src, dst)
        return
    image = Image.open(src).convert("RGB")
    image.save(dst, quality=95)


def save_mask(src: Path, dst: Path, foreground_threshold: int) -> None:
    mask = Image.open(src).convert("L")
    mask_np = np.array(mask, dtype=np.uint8)

    # Keep the output simple and compatible with the project:
    # background = 0, foreground = 255.
    binary = np.where(mask_np > foreground_threshold, 255, 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(dst)


def split_names(names: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    shuffled = list(names)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_names = sorted(shuffled[:val_count])
    train_names = sorted(shuffled[val_count:])

    if not train_names:
        raise RuntimeError("Validation split consumed all samples. Please lower val_ratio.")
    return train_names, val_names


def write_split_file(path: Path, names: Sequence[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    masks_dir = Path(args.masks_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks-dir not found: {masks_dir}")

    pairs = match_image_mask_pairs(images_dir, masks_dir)
    jpeg_dir, seg_dir, split_dir = ensure_dirs(output_root)

    kept_names: List[str] = []
    for image_path, mask_path, stem in pairs:
        save_image(image_path, jpeg_dir / f"{stem}.jpg", copy_images=args.copy_images)
        save_mask(mask_path, seg_dir / f"{stem}.png", foreground_threshold=args.foreground_threshold)
        kept_names.append(stem)

    train_names, val_names = split_names(kept_names, args.val_ratio, args.seed)
    write_split_file(split_dir / "train.txt", train_names)
    write_split_file(split_dir / "val.txt", val_names)

    print(f"[Done] Prepared {len(kept_names)} samples.")
    print(f"[Train] {len(train_names)}")
    print(f"[Val]   {len(val_names)}")
    print(f"[Output] {output_root / 'VOCdevkit' / 'VOC2007'}")


if __name__ == "__main__":
    main()
