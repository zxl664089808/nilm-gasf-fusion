#!/usr/bin/env python
"""Render the keyed manuscript into a numbered Markdown and HTML preview."""

from __future__ import annotations

import re
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("manuscript_electronics_v0.2.md")
BIB = ROOT / "references" / "references_cited.bib"
NUMBERED = Path(__file__).with_name("manuscript_electronics_v0.2_numbered.md")
HTML = Path(__file__).with_name("manuscript_electronics_v0.2_preview.html")


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    entry_re = re.compile(
        r"(?ms)^@(\w+)\s*\{\s*([^,\n]+),\s*\n(.*?)(?=^@|\Z)"
    )
    field_re = re.compile(r"(?m)^\s*(\w+)\s*=\s*\{(.*)\},?\s*$")
    return {
        match.group(2).strip(): {
            key.lower(): value.strip()
            for key, value in field_re.findall(match.group(3))
        }
        for match in entry_re.finditer(text)
    }


def plain(value: str) -> str:
    value = re.sub(r"\\&", "&", value)
    return re.sub(r"[{}]", "", value).strip()


def main() -> None:
    manuscript = SOURCE.read_text(encoding="utf-8")
    entries = parse_bib(BIB.read_text(encoding="utf-8-sig"))

    order: list[str] = []
    citation_keys: list[str] = []
    for match in re.finditer(r"\[(@[^\]]+)\]", manuscript):
        citation_keys.extend(
            re.findall(r"@([A-Za-z0-9_.:-]+)", match.group(1))
        )
    for key in citation_keys:
        if key in entries and key not in order:
            order.append(key)
    missing = sorted({key for key in citation_keys if key not in entries})
    if missing:
        raise ValueError(f"Citations missing from {BIB}: {missing}")

    number = {key: index + 1 for index, key in enumerate(order)}

    def replace_citation(match: re.Match[str]) -> str:
        keys = re.findall(r"@([A-Za-z0-9_.:-]+)", match.group(1))
        return "[" + ",".join(str(number[key]) for key in keys) + "]"

    rendered = re.sub(r"\[(@[^\]]+)\]", replace_citation, manuscript)
    references: list[str] = []
    page_separator = chr(0x2013)
    for key in order:
        fields = entries[key]
        authors = plain(fields.get("author", "[Authors missing]"))
        authors = re.sub(r"\.\.+", ".", authors.replace(" and ", "; ")).rstrip(".")
        title = plain(fields.get("title", "[Title missing]")).rstrip(".")
        venue = plain(
            fields.get("journal")
            or fields.get("booktitle")
            or fields.get("publisher")
            or ""
        )
        year = fields.get("year", "[Year missing]")
        volume = fields.get("volume", "")
        issue = fields.get("number", "")
        pages = plain(fields.get("pages", ""))
        pages = re.sub(r"(?<=\d)-{1,2}(?=\d)", page_separator, pages)
        location = volume + (f"({issue})" if issue else "")
        if pages:
            location += (", " if location else "") + pages

        reference = f"[{number[key]}] {authors}. {title}. "
        if venue:
            reference += f"*{venue}*. "
        reference += str(year)
        if location:
            reference += f", {location}"
        if fields.get("doi"):
            reference += f". https://doi.org/{plain(fields['doi'])}"
        references.append(reference.rstrip() + ".")

    rendered = rendered.replace(
        "[[INSERT RENDERED REFERENCE LIST FROM references_cited.bib]]",
        "\n".join(references),
    )
    NUMBERED.write_bytes(rendered.encode("utf-8"))

    body = markdown.markdown(rendered, extensions=["tables", "sane_lists"])
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NILM manuscript draft v0.2</title>
<style>
body{font-family:Georgia,"Times New Roman",serif;max-width:980px;margin:40px auto;
padding:0 28px;line-height:1.62;color:#202124}
h1{font-size:2rem;line-height:1.25;border-bottom:2px solid #bbb;padding-bottom:.5rem}
h2{margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.25rem}
h3{margin-top:1.5rem}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.94rem}
th,td{border:1px solid #bbb;padding:.42rem .55rem;vertical-align:top}
th{background:#f3f3f3}
code{background:#f4f4f4;padding:.1rem .25rem}
img{display:block;max-width:100%;height:auto;margin:1.2rem auto}
@media print{body{max-width:none;margin:12mm}h2{break-after:avoid}
table,figure{break-inside:avoid}}
</style>
</head>
<body>
""" + body + """
</body>
</html>
"""
    HTML.write_bytes(html.encode("utf-8"))
    print(
        f"cited={len(order)} numbered_bytes={NUMBERED.stat().st_size} "
        f"html_bytes={HTML.stat().st_size}"
    )


if __name__ == "__main__":
    main()
