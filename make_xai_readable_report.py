"""
KMAR-50K XAI Readable Report Builder
====================================

The default xai_overview_contact_sheet.png is a thumbnail overview, so text can be hard to read.
This script builds readable outputs from the existing 80 Grad-CAM panels:

1. xai_outputs/readable_report/index.html
   - full-width readable HTML report
   - grouped by category
   - shows each PNG panel and metadata

2. xai_outputs/category_contact_sheets/<category>_readable_sheet.png
   - one high-resolution vertical sheet per category
   - each panel kept large enough to read text

3. xai_outputs/XAI_READABLE_REPORT.md
   - short index/instructions for README/thesis

Run:
    cd /d D:\KMAR-50K\KMAR-50K
    python make_xai_readable_report.py

Optional:
    set KMAR_XAI_SHEET_WIDTH=2600
"""

from __future__ import annotations

import csv
import html
import os
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


BASE = Path(os.environ.get("KMAR_BASE", r"D:\KMAR-50K\KMAR-50K"))
XAI_DIR = BASE / "xai_outputs"
CASE_CSV = XAI_DIR / "xai_case_table.csv"

REPORT_DIR = XAI_DIR / "readable_report"
SHEET_DIR = XAI_DIR / "category_contact_sheets"

SHEET_WIDTH = int(os.environ.get("KMAR_XAI_SHEET_WIDTH", "2600"))
PADDING = 40
HEADER_HEIGHT = 95

CATEGORY_ORDER = [
    "correct_good",
    "correct_moderate",
    "correct_bad",
    "good_to_moderate_errors",
    "good_to_bad_errors",
    "moderate_to_good_errors",
    "moderate_to_bad_errors",
    "bad_to_good_errors",
    "bad_to_moderate_errors",
    "review_required",
]


def load_rows() -> List[dict]:
    if not CASE_CSV.exists():
        raise FileNotFoundError(f"Missing XAI case table: {CASE_CSV}")

    rows = []
    with open(CASE_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {CASE_CSV}")
    return rows


def normalize_image_path(row: dict) -> Path:
    rel = row.get("image", "").replace("\\", "/")
    if rel.startswith("xai_outputs/"):
        rel = rel[len("xai_outputs/"):]
    return XAI_DIR / rel


def category_label(cat: str) -> str:
    return cat.replace("_", " ").title()


def get_font(size: int):
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_category_sheet(category: str, rows: List[dict]) -> Path:
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    font_title = get_font(34)
    font_meta = get_font(22)

    images = []
    for r in rows:
        p = normalize_image_path(r)
        if not p.exists():
            print(f"WARNING missing image: {p}")
            continue
        img = Image.open(p).convert("RGB")

        target_img_width = SHEET_WIDTH - (2 * PADDING)
        scale = target_img_width / img.width
        new_size = (target_img_width, int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        title = (
            f"Case {r.get('case_id','')} | {category_label(category)} | "
            f"True={r.get('true_name','')} | Pred={r.get('pred_name','')} | "
            f"Plane={r.get('plane_name','')} | Slice={r.get('slice','')}"
        )
        meta = (
            f"Confidence={r.get('confidence','')} | Margin={r.get('margin','')} | "
            f"Phase3(G/M/B)=({r.get('phase3_good','')}, {r.get('phase3_moderate','')}, {r.get('phase3_bad','')}) | "
            f"Phase5(G/M/B)=({r.get('phase5_good','')}, {r.get('phase5_moderate','')}, {r.get('phase5_bad','')})"
        )

        block_h = HEADER_HEIGHT + img.height + PADDING
        block = Image.new("RGB", (SHEET_WIDTH, block_h), "white")
        draw = ImageDraw.Draw(block)
        draw.text((PADDING, 18), title, fill="black", font=font_title)
        draw.text((PADDING, 58), meta[:230], fill=(40, 40, 40), font=font_meta)
        block.paste(img, (PADDING, HEADER_HEIGHT))
        images.append(block)

    if not images:
        raise RuntimeError(f"No images found for category: {category}")

    total_h = sum(i.height for i in images) + PADDING
    sheet = Image.new("RGB", (SHEET_WIDTH, total_h), "white")
    y = PADDING // 2
    for img in images:
        sheet.paste(img, (0, y))
        y += img.height

    out = SHEET_DIR / f"{category}_readable_sheet.png"
    sheet.save(out, quality=95)
    return out


def make_html_report(rows: List[dict], sheets: Dict[str, Path]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows_by_cat: Dict[str, List[dict]] = {}
    for r in rows:
        rows_by_cat.setdefault(r.get("category", "unknown"), []).append(r)

    css = """
    body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; color: #111; }
    h1 { margin-bottom: 0; }
    .sub { color: #444; margin-top: 6px; }
    .nav a { display: inline-block; margin: 4px 8px 4px 0; padding: 7px 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; text-decoration: none; color: #0645ad; }
    .category { margin-top: 42px; padding-top: 12px; border-top: 3px solid #222; }
    .case { background: white; margin: 24px 0; padding: 18px; border: 1px solid #ddd; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .meta { font-size: 14px; line-height: 1.45; margin: 10px 0 14px 0; }
    .prob { font-family: Consolas, monospace; background: #f1f1f1; padding: 8px; border-radius: 6px; display: inline-block; }
    img.panel { width: 100%; max-width: 2200px; height: auto; border: 1px solid #ccc; background: #fff; display: block; margin-top: 10px; }
    .sheet-link { margin: 12px 0; font-weight: bold; }
    code { background: #eee; padding: 2px 4px; border-radius: 3px; }
    """
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>KMAR-50K XAI Readable Report</title>")
    parts.append(f"<style>{css}</style></head><body>")
    parts.append("<h1>KMAR-50K XAI / Grad-CAM Readable Report</h1>")
    parts.append("<p class='sub'>Full-size Grad-CAM panels grouped by case category. Use browser zoom if needed.</p>")
    parts.append("<p><b>Final model:</b> Phase 3 + Phase 5 ensemble. Thresholds: Good=0.240, Moderate=0.600, Bad=0.790.</p>")

    parts.append("<div class='nav'>")
    for cat in CATEGORY_ORDER:
        if cat in rows_by_cat:
            parts.append(f"<a href='#{cat}'>{html.escape(category_label(cat))}</a>")
    parts.append("</div>")

    for cat in CATEGORY_ORDER:
        cat_rows = rows_by_cat.get(cat, [])
        if not cat_rows:
            continue
        parts.append(f"<section class='category' id='{cat}'>")
        parts.append(f"<h2>{html.escape(category_label(cat))} ({len(cat_rows)} cases)</h2>")

        if cat in sheets:
            sheet_rel = os.path.relpath(sheets[cat], REPORT_DIR).replace("\\", "/")
            parts.append(f"<p class='sheet-link'>High-resolution vertical sheet: <a href='{sheet_rel}'>{html.escape(sheets[cat].name)}</a></p>")

        for r in cat_rows:
            img_path = normalize_image_path(r)
            img_rel = os.path.relpath(img_path, REPORT_DIR).replace("\\", "/")
            parts.append("<div class='case'>")
            parts.append(
                f"<h3>Case {html.escape(str(r.get('case_id','')))} — "
                f"True: {html.escape(str(r.get('true_name','')))} | "
                f"Pred: {html.escape(str(r.get('pred_name','')))}</h3>"
            )
            parts.append("<div class='meta'>")
            parts.append(f"<b>Category:</b> {html.escape(str(r.get('category','')))}<br>")
            parts.append(f"<b>Plane:</b> {html.escape(str(r.get('plane_name','')))} | <b>Slice:</b> {html.escape(str(r.get('slice','')))}<br>")
            parts.append(f"<b>Confidence:</b> {html.escape(str(r.get('confidence','')))} | <b>Margin:</b> {html.escape(str(r.get('margin','')))}<br>")
            parts.append(f"<b>Source path:</b> <code>{html.escape(str(r.get('path','')))}</code><br>")
            parts.append("</div>")
            parts.append("<div class='prob'>")
            parts.append(f"Phase3 probs [G,M,B] = [{r.get('phase3_good','')}, {r.get('phase3_moderate','')}, {r.get('phase3_bad','')}]<br>")
            parts.append(f"Phase5 probs [G,M,B] = [{r.get('phase5_good','')}, {r.get('phase5_moderate','')}, {r.get('phase5_bad','')}]")
            parts.append("</div>")
            parts.append(f"<img class='panel' src='{img_rel}' alt='XAI panel case {html.escape(str(r.get('case_id','')))}'>")
            parts.append("</div>")
        parts.append("</section>")

    parts.append("</body></html>")
    out = REPORT_DIR / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def make_markdown_index(rows: List[dict], sheets: Dict[str, Path], html_path: Path) -> Path:
    lines = []
    lines.append("# KMAR-50K XAI Readable Report\n\n")
    lines.append("The original `xai_overview_contact_sheet.png` is only a thumbnail overview. Use this readable report for inspection and thesis screenshots.\n\n")
    lines.append(f"- HTML report: `{html_path.relative_to(BASE)}`\n")
    lines.append(f"- Category sheets folder: `{SHEET_DIR.relative_to(BASE)}`\n")
    lines.append(f"- Case table: `{CASE_CSV.relative_to(BASE)}`\n\n")
    lines.append("## Category sheets\n\n")
    for cat in CATEGORY_ORDER:
        if cat in sheets:
            lines.append(f"- {category_label(cat)}: `{sheets[cat].relative_to(BASE)}`\n")
    lines.append("\n## Commands\n\n")
    lines.append("```bat\n")
    lines.append("start xai_outputs\\readable_report\\index.html\n")
    lines.append("start xai_outputs\\category_contact_sheets\n")
    lines.append("```\n")
    out = XAI_DIR / "XAI_READABLE_REPORT.md"
    out.write_text("".join(lines), encoding="utf-8")
    return out


def main():
    print("=" * 80)
    print("KMAR-50K XAI Readable Report Builder")
    print("=" * 80)
    print(f"BASE     : {BASE}")
    print(f"XAI_DIR  : {XAI_DIR}")
    print(f"CASE CSV : {CASE_CSV}")
    print(f"WIDTH    : {SHEET_WIDTH}")

    rows = load_rows()
    print(f"Loaded rows: {len(rows)}")

    rows_by_cat: Dict[str, List[dict]] = {}
    for r in rows:
        rows_by_cat.setdefault(r.get("category", "unknown"), []).append(r)

    sheets: Dict[str, Path] = {}
    for cat in CATEGORY_ORDER:
        cat_rows = rows_by_cat.get(cat, [])
        if not cat_rows:
            continue
        print(f"Building sheet: {cat} ({len(cat_rows)} cases)")
        sheets[cat] = make_category_sheet(cat, cat_rows)

    html_path = make_html_report(rows, sheets)
    md_path = make_markdown_index(rows, sheets, html_path)

    print("\nDone.")
    print(f"HTML report : {html_path}")
    print(f"MD index    : {md_path}")
    print(f"Sheets dir  : {SHEET_DIR}")
    print("\nOpen now:")
    print(r"    start xai_outputs\readable_report\index.html")
    print(r"    start xai_outputs\category_contact_sheets")


if __name__ == "__main__":
    main()
