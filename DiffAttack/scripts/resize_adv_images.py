#!/usr/bin/env python3
"""Resize `*_adv_image.png` images under `./outputs` to 32x32 and save to `./outputs/results`.

Saves files as `x.png` where x is the numeric id found in the original filename
with no leading zeros (e.g. `0001_adv_image.png` -> `1.png`).

Usage:
    python scripts/resize_adv_images.py

Optional arguments:
    --src PATH       Source folder (default: ./outputs)
    --dst PATH       Destination folder (default: ./outputs/results)
    --size N         Output size (default: 32)
    --pattern STR    Filename pattern suffix (default: _adv_image.png)
    --overwrite      Overwrite existing files in destination

The script ignores any `results` subdirectory inside `--src` to avoid reprocessing.
"""
from pathlib import Path
import re
import sys
import argparse

try:
    from PIL import Image
except Exception:
    print("Pillow is required. Install with: pip install pillow")
    sys.exit(2)


def find_adv_images(src: Path, pattern: str):
    # Recursive search for files ending with the given pattern
    for p in src.rglob(f"*{pattern}"):
        # skip files under results folder to avoid reprocessing
        if 'results' in [part.lower() for part in p.parts]:
            continue
        yield p


def extract_number(name: str):
    # Find the first contiguous group of digits in the filename
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser(description="Resize adv images to smaller size and collect results")
    ap.add_argument("--src", default="./outputs", help="source folder to search")
    ap.add_argument("--dst", default="./outputs/images", help="destination folder")
    ap.add_argument("--size", type=int, default=32, help="output square size (pixels)")
    ap.add_argument("--pattern", default="_adv_image.png", help="filename suffix to look for")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing files in destination")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    size = args.size
    pattern = args.pattern

    if not src.exists() or not src.is_dir():
        print(f"Source folder not found: {src}")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0
    overwritten = 0

    for p in find_adv_images(src, pattern):
        try:
            num = extract_number(p.name)
            if num is None:
                # fallback: use stem without suffix
                out_name = p.stem + ".png"
            else:
                out_name = f"{num}.png"

            out_path = dst / out_name
            if out_path.exists() and not args.overwrite:
                skipped += 1
                print(f"Skip (exists): {out_path}")
                continue

            with Image.open(p) as im:
                # Ensure RGB for consistent PNGs
                im = im.convert("RGB")
                im_resized = im.resize((size, size), Image.LANCZOS)
                im_resized.save(out_path, format="PNG")

            if out_path.exists() and args.overwrite:
                overwritten += 1

            processed += 1
            print(f"Saved: {out_path}")

        except Exception as e:
            errors += 1
            print(f"Error processing {p}: {e}")

    print("--- Summary ---")
    print(f"Processed: {processed}")
    print(f"Skipped (exists): {skipped}")
    print(f"Overwritten: {overwritten}")
    print(f"Errors: {errors}")


if __name__ == '__main__':
    main()
