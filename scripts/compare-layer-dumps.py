#!/usr/bin/env python3
"""Compare FP32-normalized ffn_out/l_out records from llama-eval-callback."""

import argparse
import array
import math
import re
import struct
from collections import defaultdict
from pathlib import Path


def read_dump(path: Path):
    records = defaultdict(list)
    with path.open("rb") as stream:
        if stream.read(8) != b"LLDUMP1\0":
            raise ValueError(f"bad dump header: {path}")
        while True:
            raw = stream.read(2)
            if not raw:
                break
            if len(raw) != 2:
                raise EOFError(path)
            name_len = struct.unpack("<H", raw)[0]
            name = stream.read(name_len).decode("utf-8")
            count = struct.unpack("<Q", stream.read(8))[0]
            values = array.array("f")
            values.fromfile(stream, count)
            records[name].append(values)
    return records


def metric(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("record count mismatch")
    dot = ref2 = cand2 = diff2 = 0.0
    count = 0
    for left, right in zip(reference, candidate):
        if len(left) != len(right):
            raise ValueError("tensor size mismatch")
        for a, b in zip(left, right):
            dot += a * b
            ref2 += a * a
            cand2 += b * b
            d = b - a
            diff2 += d * d
        count += len(left)
    cosine = dot / math.sqrt(ref2 * cand2) if ref2 and cand2 else float("nan")
    rel_l2 = math.sqrt(diff2 / ref2) if ref2 else float("nan")
    return count, cosine, rel_l2


def layer_number(name: str) -> int:
    match = re.search(r"-(\d+)$", name)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = read_dump(args.reference)
    candidate = read_dump(args.candidate)
    rows = []
    previous_l_out_error = 0.0
    names = sorted(set(reference) & set(candidate), key=lambda name: (layer_number(name), name))
    for name in names:
        count, cosine, rel_l2 = metric(reference[name], candidate[name])
        error = 1.0 - cosine
        amplification = ""
        if name.startswith("l_out-"):
            amplification = f"{error - previous_l_out_error:.9g}"
            previous_l_out_error = error
        rows.append(f"{layer_number(name)},{name},{count},{cosine:.9g},{rel_l2:.9g},{amplification}")

    text = "layer,tensor,values,cosine,relative_l2,cumulative_error_delta\n" + "\n".join(rows) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
