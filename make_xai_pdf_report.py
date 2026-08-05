"""
KMAR-50K XAI PDF Report Builder
===============================

Creates readable PDF reports from generated XAI / Grad-CAM outputs.

Inputs expected:
    xai_outputs/xai_case_table.csv
    xai_outputs/<category>/*.png

Optional input:
    xai_outputs/category_contact_sheets/*_readable_sheet.png
    created by make_xai_readable_report.py

Outputs:
    xai_outputs/KMAR50K_XAI_FULL_REPORT.pdf
    xai_outputs/KMAR50K_XAI_CATEGORY_SHEETS.pdf   if category sheets exist
    xai_outputs/KMAR50K_XAI_PDF_INDEX.md

Run:
    cd /d D:\KMAR-50K\KMAR-50K
    python make_xai_pdf_report.py

Optional:
    set KMAR_XAI_PDF_DPI=200
    set KMAR_XAI_PDF_MAX_CASES=80
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
from PIL import JpegImagePlugin, PdfImagePlugin
Image.init()


BASE = Path(os.environ.get("KMAR_BASE", r"D:\KMAR-50K\KMAR-50K"))
XAI_DIR = BASE / "xai_outputs"
CASE_CSV = XAI_DIR / "xai_case_table.csv"

CATEGORY_SHEET_DIR = XAI_DIR / "category_contact_sheets"

OUT_FULL_PDF = XAI_DIR / "KMAR50K_XAI_FULL_REPORT.pdf"
OUT_CATEGORY_PDF = XAI_DIR / "KMAR50K_XAI_CATEGORY_SHEETS.pdf"
OUT_INDEX_MD = XAI_DIR / "KMAR50K_XAI_PDF_INDEX.md"

DPI = int(os.environ.get("KMAR_XAI_PDF_DPI", "200"))
MAX_CASES = int(os.environ.get("KMAR_XAI_PDF_MAX_CASES", "80"))

# A4 landscape at 200 DPI: ~2339 x 1654
PAGE_W = int(11.69 * DPI)
PAGE_H = int(8.27 * DPI)

MARGIN = int(0.22 * DPI)
HEADER_H = int(0.62 * DPI)
FOOTER_H = int(0.22 * DPI)

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


def get_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
        ]
    candidates += [
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


FONT_TITLE = get_font(int(0.16 * DPI), bold=True)
FONT_SUBTITLE = get_font(int(0.105 * DPI), bold=False)
FONT_META = get_font(int(0.083 * DPI), bold=False)
FONT_SMALL = get_font(int(0.068 * DPI), bold=False)
FONT_BIG = get_font(int(0.22 * DPI), bold=True)


def load_rows() -> List[dict]:
    if not CASE_CSV.exists():
        raise FileNotFoundError(f"Missing XAI case table: {CASE_CSV}")

    rows = []
    with open(CASE_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise RuntimeError("xai_case_table.csv has no rows.")

    return rows[:MAX_CASES]


def normalize_image_path(row: dict) -> Path:
    rel = row.get("image", "").replace("\\", "/")
    if rel.startswith("xai_outputs/"):
        rel = rel[len("xai_outputs/"):]
    return XAI_DIR / rel


def category_label(cat: str) -> str:
    return cat.replace("_", " ").title()


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy, font, fill, max_width: int, line_gap: int = 4):
    x, y = xy
    words = str(text).split()
    line = ""
    for word in words:
        test = line + (" " if line else "") + word
        bbox = draw.textbbox((x, y), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            line = test
        else:
            if line:
                draw.text((x, y), line, font=font, fill=fill)
                y += (bbox[3] - bbox[1]) + line_gap
            line = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def make_cover_page(rows: List[dict]) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(page)

    y = MARGIN * 2
    d.text((MARGIN, y), "KMAR-50K XAI / Grad-CAM PDF Report", font=FONT_BIG, fill="black")
    y += int(0.42 * DPI)

    intro = (
        "Readable PDF report generated from final ensemble Grad-CAM evidence. "
        "Each case page includes the full XAI panel and metadata from xai_case_table.csv."
    )
    y = draw_wrapped_text(d, intro, (MARGIN, y), FONT_SUBTITLE, (40, 40, 40), PAGE_W - 2 * MARGIN)
    y += int(0.18 * DPI)

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.get("category", "unknown")] = counts.get(r.get("category", "unknown"), 0) + 1

    d.text((MARGIN, y), "Final model explained", font=FONT_TITLE, fill="black")
    y += int(0.24 * DPI)
    lines = [
        "Final full-coverage model: Phase 3 + Phase 5 probability ensemble",
        "Phase 3 weight: 0.25",
        "Phase 5 weight: 0.75",
        "TTA used for final metric: original + horizontal flip",
        "Thresholds: Good=0.240, Moderate=0.600, Bad=0.790",
        "Full-coverage locked-test accuracy: 82.69%",
        "Confidence-aware accepted-only accuracy: 86.23% at 88.91% coverage",
    ]
    for line in lines:
        d.text((MARGIN + 20, y), "- " + line, font=FONT_SUBTITLE, fill=(20, 20, 20))
        y += int(0.17 * DPI)

    y += int(0.12 * DPI)
    d.text((MARGIN, y), f"Included XAI cases: {len(rows)}", font=FONT_TITLE, fill="black")
    y += int(0.25 * DPI)

    for cat in CATEGORY_ORDER:
        if cat in counts:
            d.text((MARGIN + 20, y), f"- {category_label(cat)}: {counts[cat]}", font=FONT_SUBTITLE, fill=(20, 20, 20))
            y += int(0.15 * DPI)

    d.text((MARGIN, PAGE_H - MARGIN - 25), "Generated locally from xai_outputs. Dataset and checkpoint files are not embedded.", font=FONT_SMALL, fill=(90, 90, 90))
    return page


def make_category_page(category: str, count: int) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(page)
    title = category_label(category)
    d.text((MARGIN, PAGE_H // 2 - 80), title, font=FONT_BIG, fill="black")
    d.text((MARGIN, PAGE_H // 2 - 20), f"{count} cases", font=FONT_TITLE, fill=(60, 60, 60))
    d.text((MARGIN, PAGE_H // 2 + 45), "The following pages show full Grad-CAM panels for this category.", font=FONT_SUBTITLE, fill=(60, 60, 60))
    return page


def make_case_page(row: dict, page_num: int, total_pages: int) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(page)

    cat = row.get("category", "")
    title = (
        f"Case {row.get('case_id','')} - {category_label(cat)} | "
        f"True={row.get('true_name','')} | Pred={row.get('pred_name','')}"
    )
    d.text((MARGIN, MARGIN), title, font=FONT_TITLE, fill="black")

    meta1 = (
        f"Plane={row.get('plane_name','')} | Slice={row.get('slice','')} | "
        f"Confidence={row.get('confidence','')} | Margin={row.get('margin','')}"
    )
    d.text((MARGIN, MARGIN + int(0.22 * DPI)), meta1, font=FONT_META, fill=(45, 45, 45))

    meta2 = (
        f"Phase3 [G,M,B]=[{row.get('phase3_good','')}, {row.get('phase3_moderate','')}, {row.get('phase3_bad','')}] | "
        f"Phase5 [G,M,B]=[{row.get('phase5_good','')}, {row.get('phase5_moderate','')}, {row.get('phase5_bad','')}]"
    )
    d.text((MARGIN, MARGIN + int(0.36 * DPI)), meta2[:260], font=FONT_SMALL, fill=(70, 70, 70))

    img_path = normalize_image_path(row)
    if not img_path.exists():
        d.text((MARGIN, PAGE_H // 2), f"Missing image: {img_path}", font=FONT_TITLE, fill="red")
    else:
        img = Image.open(img_path).convert("RGB")
        max_w = PAGE_W - 2 * MARGIN
        max_h = PAGE_H - MARGIN - HEADER_H - FOOTER_H
        fitted = fit_image(img, max_w, max_h)
        x = (PAGE_W - fitted.width) // 2
        y = MARGIN + HEADER_H
        page.paste(fitted, (x, y))

    footer = f"KMAR-50K XAI PDF Report | Page {page_num}/{total_pages}"
    d.text((MARGIN, PAGE_H - MARGIN + 5), footer, font=FONT_SMALL, fill=(90, 90, 90))
    return page


def save_pdf(pages: List[Image.Image], out_path: Path):
    if not pages:
        raise RuntimeError("No pages to save.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        out_path,
        save_all=True,
        append_images=pages[1:],
        resolution=DPI,
        quality=95,
    )


def make_full_report(rows: List[dict]) -> Path:
    rows_by_cat: Dict[str, List[dict]] = {}
    for r in rows:
        rows_by_cat.setdefault(r.get("category", "unknown"), []).append(r)

    # cover + category separators + case pages
    total_pages = 1 + sum(1 for cat in CATEGORY_ORDER if cat in rows_by_cat) + len(rows)
    pages: List[Image.Image] = [make_cover_page(rows)]

    page_num = 2
    for cat in CATEGORY_ORDER:
        cat_rows = rows_by_cat.get(cat, [])
        if not cat_rows:
            continue

        pages.append(make_category_page(cat, len(cat_rows)))
        page_num += 1

        for row in cat_rows:
            pages.append(make_case_page(row, page_num, total_pages))
            page_num += 1

    save_pdf(pages, OUT_FULL_PDF)
    return OUT_FULL_PDF


def make_category_sheets_pdf() -> Path | None:
    if not CATEGORY_SHEET_DIR.exists():
        return None

    sheet_files = []
    for cat in CATEGORY_ORDER:
        p = CATEGORY_SHEET_DIR / f"{cat}_readable_sheet.png"
        if p.exists():
            sheet_files.append(p)

    if not sheet_files:
        return None

    pages = []
    for p in sheet_files:
        img = Image.open(p).convert("RGB")
        # For category sheets, use portrait page if tall; fit to a large image page.
        max_dim = 4200
        scale = min(max_dim / img.width, max_dim / img.height, 1.0)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        pages.append(img)

    save_pdf(pages, OUT_CATEGORY_PDF)
    return OUT_CATEGORY_PDF


def write_index(full_pdf: Path, category_pdf: Path | None):
    lines = []
    lines.append("# KMAR-50K XAI PDF Outputs\n\n")
    lines.append(f"- Full PDF report: `{full_pdf.relative_to(BASE)}`\n")
    if category_pdf:
        lines.append(f"- Category-sheets PDF: `{category_pdf.relative_to(BASE)}`\n")
    lines.append(f"- Source case table: `{CASE_CSV.relative_to(BASE)}`\n\n")
    lines.append("## Open commands\n\n")
    lines.append("```bat\n")
    lines.append("start xai_outputs\\KMAR50K_XAI_FULL_REPORT.pdf\n")
    if category_pdf:
        lines.append("start xai_outputs\\KMAR50K_XAI_CATEGORY_SHEETS.pdf\n")
    lines.append("```\n")
    OUT_INDEX_MD.write_text("".join(lines), encoding="utf-8")


def main():
    print("=" * 80)
    print("KMAR-50K XAI PDF Report Builder")
    print("=" * 80)
    print(f"BASE      : {BASE}")
    print(f"XAI_DIR   : {XAI_DIR}")
    print(f"CASE_CSV  : {CASE_CSV}")
    print(f"DPI       : {DPI}")
    print(f"MAX_CASES : {MAX_CASES}")

    rows = load_rows()
    print(f"Loaded cases: {len(rows)}")

    full_pdf = make_full_report(rows)
    print(f"Saved full PDF: {full_pdf}")

    category_pdf = make_category_sheets_pdf()
    if category_pdf:
        print(f"Saved category-sheets PDF: {category_pdf}")
    else:
        print("Category sheets not found. Skipped KMAR50K_XAI_CATEGORY_SHEETS.pdf")

    write_index(full_pdf, category_pdf)
    print(f"Saved index: {OUT_INDEX_MD}")

    print("\nOpen:")
    print(r"    start xai_outputs\KMAR50K_XAI_FULL_REPORT.pdf")
    if category_pdf:
        print(r"    start xai_outputs\KMAR50K_XAI_CATEGORY_SHEETS.pdf")


if __name__ == "__main__":
    main()
