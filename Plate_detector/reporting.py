"""Job report builders - CSV (summary + raw plate rows) and PDF (formatted
summary table followed by a crop image per saved plate), mirroring
restricted_zone_monitor's report builders. Detection only: there is no plate
text to report, only vehicle class, confidence/quality scores and the
enhanced crop image for each finalized track."""

from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd


def _report_rows(summary: dict) -> list[dict]:
    rows = []
    for i, p in enumerate(summary.get("plates", [])):
        rows.append({
            "#": i + 1,
            "Track ID": p.get("track_id"),
            "Vehicle": p.get("vehicle"),
            "Plate Confidence": p.get("plate_conf"),
            "Quality Score": p.get("quality"),
            "Needs Review": p.get("low_conf"),
            "Timestamp": p.get("timestamp"),
            "Image": p.get("image"),
        })
    return rows


def build_job_report_csv(summary: dict) -> Optional[bytes]:
    """CSV report: a summary block followed by one row per saved plate.
    Returns None if there is nothing to report."""
    if not summary:
        return None
    counts = summary.get("counts", {})
    overview = {
        "Source": summary.get("source", "-"),
        "Frames Processed": summary.get("frames", 0),
        "Elapsed Seconds": summary.get("elapsed_s", 0),
        "Average FPS": summary.get("fps", 0),
        "Plates Saved": counts.get("plates_saved", 0),
        "Needs Review (Low Confidence)": counts.get("lowconf_quarantined", 0),
    }
    rows = _report_rows(summary)

    buffer = io.StringIO()
    buffer.write("ALPR - Job Report\n\n")
    buffer.write("Summary\n")
    pd.DataFrame([overview]).to_csv(buffer, index=False)
    buffer.write("\nPlates\n")
    if rows:
        pd.DataFrame(rows).drop(columns=["Image"]).to_csv(buffer, index=False)
    else:
        buffer.write("No plates saved.\n")
    return buffer.getvalue().encode("utf-8")


def build_job_report_pdf(summary: dict) -> Optional[bytes]:
    """Formatted PDF report: title, a summary table, then one crop image per
    saved plate. Returns None if there is nothing to report."""
    if not summary:
        return None

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    page_width = letter[0] - 1.2 * inch

    def styled_table(data, col_widths=None):
        table = Table(data, hAlign="LEFT", repeatRows=1, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2430")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return table

    counts = summary.get("counts", {})
    overview = {
        "Source": summary.get("source", "-"),
        "Frames Processed": summary.get("frames", 0),
        "Elapsed Seconds": summary.get("elapsed_s", 0),
        "Average FPS": summary.get("fps", 0),
        "Plates Saved": counts.get("plates_saved", 0),
        "Needs Review (Low Confidence)": counts.get("lowconf_quarantined", 0),
    }
    summary_table = styled_table(
        [["Field", "Value"]] + [[str(k), str(v)] for k, v in overview.items()],
        col_widths=[page_width * 0.4, page_width * 0.6],
    )

    story = [
        Paragraph("ALPR - Job Report", styles["Title"]),
        Spacer(1, 0.15 * inch),
        Paragraph("Summary", styles["Heading2"]),
        summary_table,
        Spacer(1, 0.3 * inch),
        Paragraph("Detected Plates", styles["Heading2"]),
    ]

    rows = _report_rows(summary)
    if not rows:
        story.append(Paragraph("No plates saved.", styles["Normal"]))
    else:
        for r in rows:
            block = [Paragraph(
                f"#{r['#']}: {r['Vehicle'] or 'vehicle'} (track {r['Track ID']}) - "
                f"plate conf {r['Plate Confidence']}, quality {r['Quality Score']}"
                f"{' - needs review' if r['Needs Review'] else ''}", styles["Heading3"],
            )]
            image_path = r.get("Image")
            if image_path and os.path.exists(image_path):
                reader = ImageReader(image_path)
                iw, ih = reader.getSize()
                disp_w = page_width * 0.35
                disp_h = disp_w * ih / iw
                block.append(Image(image_path, width=disp_w, height=disp_h))
            else:
                block.append(Paragraph("(image unavailable)", styles["Normal"]))
            block.append(Spacer(1, 0.2 * inch))
            story.append(KeepTogether(block))

    doc.build(story)
    return buffer.getvalue()
