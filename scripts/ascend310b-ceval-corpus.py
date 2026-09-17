#!/usr/bin/env python3
"""Build an independent Chinese accuracy corpus from official C-Eval validation CSVs."""

import argparse
import csv
import io
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    blocks = []
    with zipfile.ZipFile(args.zip_path) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith("val/") and name.endswith(".csv"))
        for name in names:
            raw = archive.read(name).decode("utf-8-sig")
            for row in csv.DictReader(io.StringIO(raw)):
                question = (row.get("question") or "").strip()
                choices = [f"{letter}. {(row.get(letter) or '').strip()}" for letter in "ABCD"]
                if question and all(row.get(letter) for letter in "ABCD"):
                    blocks.append("题目：" + question + "\n" + "\n".join(choices) + "\n")

    if not blocks:
        raise RuntimeError("No C-Eval validation questions found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(blocks), encoding="utf-8")
    print(f"wrote {len(blocks)} validation questions to {args.output}")


if __name__ == "__main__":
    main()
