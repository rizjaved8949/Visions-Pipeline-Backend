"""
Session report builders - CSV (summary + raw event rows) and PDF (formatted
summary table followed by a breach-snapshot image per event), mirroring
Attendance's build_session_report_* functions.

Both read the just-finished session out of `monitor.current_session` (it is
kept around after stop_session() specifically so a report can still be
built) and neither one wipes anything themselves - the caller (api.py) wipes
via monitor.wipe_all_data() only after the bytes have been produced.
"""

from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd

from . import monitor


def _session_snapshot():
    """Returns (summary_dict, events_list) for the just-finished session, or
    None if there is nothing to report."""
    with monitor.session_lock:
        session = monitor.current_session
        if session is None:
            return None
        end_time = session["end_time"]
        duration = "-"
        if end_time is not None:
            secs = (end_time - session["start_time"]).total_seconds()
            duration = f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m {int(secs % 60)}s"

        summary = {
            "Session ID": session["session_id"],
            "Date": session["start_time"].date().isoformat(),
            "Start Time": session["start_time"].strftime("%H:%M:%S"),
            "End Time": end_time.strftime("%H:%M:%S") if end_time else "-",
            "Duration": duration,
            "Zones Monitored": ", ".join(z.name for z in session["zones"].zones) or "-",
            "Frames Processed": session["frame_count"],
            "Total Breaches": session["total_breaches"],
        }
        events = list(session["events"])
    return summary, events


def build_session_report_csv() -> Optional[bytes]:
    """CSV report: a summary block followed by one row per breach event.
    Returns None if there is no finished session to report."""
    snapshot = _session_snapshot()
    if snapshot is None:
        return None
    summary, events = snapshot

    buffer = io.StringIO()
    buffer.write("Restricted Zone Monitor - Session Report\n\n")
    buffer.write("Summary\n")
    pd.DataFrame([summary]).to_csv(buffer, index=False)
    buffer.write("\nBreach Events\n")
    if events:
        rows = [
            {"#": i + 1, "Track ID": e["track_id"], "Class": e["cls_name"],
             "Zone": e["zone"], "Time": e["time"]}
            for i, e in enumerate(events)
        ]
        pd.DataFrame(rows).to_csv(buffer, index=False)
    else:
        buffer.write("No breaches recorded.\n")
    return buffer.getvalue().encode("utf-8")


def build_session_report_pdf() -> Optional[bytes]:
    """Formatted PDF report: title, a summary table, an events table, then
    one snapshot image per breach (the frame at the moment it was detected).
    Returns None if there is no finished session to report."""
    snapshot = _session_snapshot()
    if snapshot is None:
        return None
    summary, events = snapshot

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

    summary_table = styled_table(
        [["Field", "Value"]] + [[str(k), str(v)] for k, v in summary.items()],
        col_widths=[page_width * 0.4, page_width * 0.6],
    )

    story = [
        Paragraph("Restricted Zone Monitor - Session Report", styles["Title"]),
        Spacer(1, 0.15 * inch),
        Paragraph("Summary", styles["Heading2"]),
        summary_table,
        Spacer(1, 0.3 * inch),
        Paragraph("Breach Events", styles["Heading2"]),
    ]

    if not events:
        story.append(Paragraph("No breaches recorded.", styles["Normal"]))
    else:
        event_rows = [["#", "Track ID", "Class", "Zone", "Time"]] + [
            [str(i + 1), str(e["track_id"]), e["cls_name"], e["zone"], e["time"]]
            for i, e in enumerate(events)
        ]
        story.append(styled_table(event_rows, col_widths=[
            page_width * 0.08, page_width * 0.17, page_width * 0.25,
            page_width * 0.25, page_width * 0.25,
        ]))
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("Breach Snapshots", styles["Heading2"]))

        for i, e in enumerate(events):
            block = [Paragraph(
                f"Event {i + 1}: {e['cls_name']} #{e['track_id']} entered "
                f"“{e['zone']}” at {e['time']}", styles["Heading3"],
            )]
            snap_path = e.get("snapshot_path")
            if snap_path and os.path.exists(snap_path):
                reader = ImageReader(snap_path)
                iw, ih = reader.getSize()
                disp_w = page_width * 0.6
                disp_h = disp_w * ih / iw
                block.append(Image(snap_path, width=disp_w, height=disp_h))
            else:
                block.append(Paragraph("(snapshot unavailable)", styles["Normal"]))
            block.append(Spacer(1, 0.2 * inch))
            story.append(KeepTogether(block))

    doc.build(story)
    return buffer.getvalue()
