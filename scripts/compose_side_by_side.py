#!/usr/bin/env python3
"""Compose two same-length videos into a titled left/right comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = True):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--left-title", default="Left")
    p.add_argument("--right-title", default="Right")
    p.add_argument("--header", default="")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--bar-height", type=int, default=52)
    args = p.parse_args()

    left_reader = imageio.get_reader(str(args.left))
    right_reader = imageio.get_reader(str(args.right))
    left_meta = left_reader.get_meta_data()
    right_meta = right_reader.get_meta_data()
    width, height = map(int, left_meta["size"])
    rw, rh = map(int, right_meta["size"])
    if (rw, rh) != (width, height):
        raise ValueError(f"video sizes differ: left={(width, height)} right={(rw, rh)}")
    fps = float(args.fps) if args.fps > 0 else float(left_meta.get("fps", 30.0))
    bar_h = max(24, int(args.bar_height))
    out_w, out_h = width * 2, height + bar_h
    args.out.parent.mkdir(parents=True, exist_ok=True)
    title_font = font(max(16, int(bar_h * 0.48)))
    header_font = font(max(14, int(bar_h * 0.34)), bold=False)
    count = min(left_reader.count_frames(), right_reader.count_frames())
    print(f"[compose] {args.left} + {args.right} -> {args.out} frames={count} size={out_w}x{out_h} fps={fps}")

    try:
        with imageio.get_writer(str(args.out), fps=fps, macro_block_size=1) as writer:
            for idx in range(count):
                li = np.asarray(left_reader.get_data(idx), dtype=np.uint8)
                ri = np.asarray(right_reader.get_data(idx), dtype=np.uint8)
                canvas = Image.new("RGB", (out_w, out_h), (24, 28, 36))
                canvas.paste(Image.fromarray(li, mode="RGB"), (0, bar_h))
                canvas.paste(Image.fromarray(ri, mode="RGB"), (width, bar_h))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, width - 1, bar_h - 1), fill=(36, 105, 180))
                draw.rectangle((width, 0, out_w - 1, bar_h - 1), fill=(177, 91, 38))
                draw.text((18, bar_h // 2), args.left_title, fill=(255, 255, 255), font=title_font, anchor="lm")
                draw.text((width + 18, bar_h // 2), args.right_title, fill=(255, 255, 255), font=title_font, anchor="lm")
                if args.header:
                    draw.rectangle((0, bar_h, out_w, bar_h + 1), fill=(235, 235, 235))
                    draw.text((out_w // 2, 3), args.header, fill=(230, 230, 230), font=header_font, anchor="ma")
                writer.append_data(np.asarray(canvas))
                if idx == 0 or (idx + 1) % 300 == 0 or idx == count - 1:
                    print(f"[compose] frame {idx + 1}/{count}")
    finally:
        left_reader.close()
        right_reader.close()
    print(f"[compose] saved {args.out}")


if __name__ == "__main__":
    main()
