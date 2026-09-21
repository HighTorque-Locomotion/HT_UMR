#!/usr/bin/env python3
"""Compose synchronized robot videos into labelled columns and an optional clip."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from compose_side_by_side import font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--title", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip-out", type=Path)
    parser.add_argument("--clip-start", type=int, default=0, help="First encoded frame in the preview.")
    parser.add_argument("--clip-end", type=int, default=0, help="Exclusive encoded frame in the preview.")
    parser.add_argument("--source-stride", type=int, default=1)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--sequence", default="Comparison")
    args = parser.parse_args()
    if len(args.input) != len(args.title) or len(args.input) < 2:
        parser.error("Provide at least two inputs and exactly one title per input.")
    if args.source_stride < 1:
        parser.error("source-stride must be positive")
    with ExitStack() as stack:
        readers = [imageio.get_reader(str(path)) for path in args.input]
        for reader in readers:
            stack.callback(reader.close)
        metadata = [reader.get_meta_data() for reader in readers]
        counts = [reader.count_frames() for reader in readers]
        width, height = map(int, metadata[0]["size"])
        fps = float(metadata[0]["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source video FPS")
        for path, meta, count in zip(args.input, metadata, counts):
            if tuple(meta["size"]) != (width, height):
                raise ValueError(f"Video dimensions do not match: {path}")
            if not np.isclose(float(meta["fps"]), fps) or count != counts[0]:
                raise ValueError(f"Video timing does not match: {path}, fps={meta['fps']}, frames={count}")
        if args.clip_out and not 0 <= args.clip_start < args.clip_end <= counts[0]:
            parser.error("Preview frame range must be inside the synchronized videos.")
        destinations = [args.out] + ([args.clip_out] if args.clip_out else [])
        for path in destinations:
            if path.resolve() in {p.resolve() for p in args.input}:
                parser.error("Output must differ from every input")
            path.parent.mkdir(parents=True, exist_ok=True)
        if args.clip_out and args.clip_out.resolve() == args.out.resolve():
            parser.error("Full video and preview must have different output paths")
        writer = stack.enter_context(imageio.get_writer(str(args.out), fps=fps, macro_block_size=1))
        clip_writer = (stack.enter_context(imageio.get_writer(str(args.clip_out), fps=fps, macro_block_size=1))
                       if args.clip_out else None)
        bar_height = 64
        colours = [(36, 105, 180), (177, 91, 38), (39, 118, 94)]
        title_font, subtitle_font = font(24), font(16, bold=False)
        print(f"[compare] columns={len(readers)} frames={counts[0]} fps={fps} size={width * len(readers)}x{height + bar_height}", flush=True)
        for index in range(counts[0]):
            canvas = Image.new("RGB", (width * len(readers), height + bar_height), (24, 28, 36))
            draw = ImageDraw.Draw(canvas)
            source_frame = args.source_start + index * args.source_stride
            source_time = source_frame / (fps * args.source_stride)
            subtitle = f"{args.sequence} | frame {source_frame} | {source_time:.2f} s"
            for column, (reader, title) in enumerate(zip(readers, args.title)):
                left = column * width
                canvas.paste(Image.fromarray(reader.get_data(index)), (left, bar_height))
                draw.rectangle((left, 0, left + width - 1, bar_height - 1), fill=colours[column % len(colours)])
                draw.text((left + 16, 6), title, fill="white", font=title_font)
                draw.text((left + 16, 37), subtitle, fill=(230, 230, 230), font=subtitle_font)
            frame = np.asarray(canvas)
            writer.append_data(frame)
            if clip_writer is not None and args.clip_start <= index < args.clip_end:
                clip_writer.append_data(frame)
            if index == 0 or (index + 1) % 300 == 0 or index == counts[0] - 1:
                print(f"[compare] {index + 1}/{counts[0]}", flush=True)
    print(f"[compare] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
