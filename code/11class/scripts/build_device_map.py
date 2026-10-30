#!/usr/bin/env python
"""Build a PLAID sample-to-device map from sub_meta.docx metadata."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-docx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def extract_items(docx_path: Path) -> list[tuple[int, dict]]:
    with ZipFile(docx_path) as archive:
        text = archive.read("word/document.xml").decode("utf-8")
    text = text.replace("</w:p>", "\n")
    text = re.sub(r"<w:tab[^>]*/>", "\t", text)
    text = re.sub(r"<w:br[^>]*/>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    decoder = json.JSONDecoder()
    items = []
    for match in re.finditer(r"Item\s+(\d+)", text):
        start = text.find("{", match.end())
        if start < 0:
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if "appliance" in payload:
            items.append((int(match.group(1)), payload["appliance"]))
    return items


def main() -> None:
    args = parse_args()
    metadata = {}
    for item_index, appliance in extract_items(args.metadata_docx):
        brand = str(appliance.get("brand", "")).strip()
        model = str(appliance.get("model_number", "")).strip()
        appliance_type = str(appliance.get("type", "")).strip()
        if brand or model:
            device_id = f"{brand}|{model}|{appliance_type}".strip("|")
        else:
            # Conservative fallback: treat the sample as its own device.
            device_id = f"unknown|{appliance_type}|item_{item_index}"
        metadata[item_index] = {
            "brand": brand,
            "model_number": model,
            "appliance_type": appliance_type,
            "device_id": device_id,
        }

    manifest = pd.read_csv(args.manifest)
    rows = []
    for source_id in manifest["source_id"].drop_duplicates():
        match = re.search(r"sample_(\d+)$", str(source_id))
        if not match:
            continue
        item_index = int(match.group(1)) + 1
        info = metadata.get(item_index)
        if info is None:
            continue
        rows.append({"source_id": source_id, **info, "item_index": item_index})
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"Mapped {result['source_id'].nunique():,} source samples to "
        f"{result['device_id'].nunique():,} device identifiers"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
