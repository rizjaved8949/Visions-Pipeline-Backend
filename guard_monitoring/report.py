from __future__ import annotations

import io
from datetime import datetime, timezone

_RULE_LABELS = {
    "sleep": "Possible sleeping detected",
    "phone": "Phone use detected",
    "stationary": "Extended stationary period",
    "absence": "Absent from position",
}

_CRITICAL_RULES = {"sleep", "absence"}


def format_seconds(seconds) -> str:
    """0/None -> '0s'; otherwise 'Hh Mm' or 'Mm' or 'Ss'."""
    if not seconds or seconds <= 0:
        return "0s"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def rule_label(rule: str) -> str:
    return _RULE_LABELS.get(rule, rule)


def severity_of(rule: str) -> str:
    return "critical" if rule in _CRITICAL_RULES else "warning"


def _format_event_time(event: dict) -> str:
    wall_time = event.get("wall_time")
    if wall_time:
        return str(wall_time)
    triggered_at = event.get("triggered_at")
    if triggered_at is None:
        return ""
    return f"t+{format_seconds(triggered_at)}"


def build_report_rows(
    *,
    camera_id: str,
    source: str,
    fps: float,
    duty_seconds: float,
    stats: dict,
    events: list[dict],
) -> tuple[list[dict], list[dict]]:
    """stats: normalized dict with keys visible_frames, stationary_frames,
    sleep_candidate_frames, phone_use_frames, absence_frames (all frame counts
    at the given fps/analysis rate). Returns (summary_rows, event_rows)."""

    def seconds_for(key: str) -> float:
        return float(stats.get(key, 0) or 0) / fps if fps else 0.0

    sleep_events = [e for e in events if e.get("rule") == "sleep"]
    phone_events = [e for e in events if e.get("rule") == "phone"]
    stationary_events = [e for e in events if e.get("rule") == "stationary"]
    absence_events = [e for e in events if e.get("rule") == "absence"]

    summary_rows = [
        {
            "Camera": camera_id,
            "Source": source,
            "Generated at": datetime.now(timezone.utc).isoformat(),
            "Duty duration": format_seconds(duty_seconds),
            "Alert (present) time": format_seconds(seconds_for("visible_frames")),
            "Inactive (stationary) time": format_seconds(seconds_for("stationary_frames")),
            "Sleep time": format_seconds(seconds_for("sleep_candidate_frames")),
            "Phone-use time": format_seconds(seconds_for("phone_use_frames")),
            "Absence time": format_seconds(seconds_for("absence_frames")),
            "Sleep events": len(sleep_events),
            "Phone events": len(phone_events),
            "Stationary events": len(stationary_events),
            "Absence events": len(absence_events),
        }
    ]

    event_rows = [
        {
            "Time": _format_event_time(event),
            "Alert": rule_label(event.get("rule", "")),
            "Severity": severity_of(event.get("rule", "")).capitalize(),
            "Threshold": format_seconds(event.get("threshold_seconds")),
        }
        for event in events
    ]

    return summary_rows, event_rows


def build_xlsx(summary_rows: list[dict], event_rows: list[dict]) -> bytes:
    import pandas as pd

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(event_rows).to_excel(writer, sheet_name="Events", index=False)
    return buffer.getvalue()


def build_pdf(title: str, summary_rows: list[dict], event_rows: list[dict]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )

    page_width = letter[0] - 1.2 * inch  # minus left+right margins

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

    def as_table(rows):
        if not rows:
            return Paragraph("No events recorded.", styles["Normal"])
        columns = list(rows[0].keys())
        data = [columns] + [[str(row.get(c, "")) for c in columns] for row in rows]
        return styled_table(data, col_widths=page_width / len(columns))

    def as_vertical_table(rows):
        if not rows:
            return [Paragraph("No data.", styles["Normal"])]
        blocks = []
        for row in rows:
            data = [["Field", "Value"]] + [[str(k), str(v)] for k, v in row.items()]
            blocks.append(styled_table(data, col_widths=[page_width * 0.4, page_width * 0.6]))
        return blocks

    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 0.15 * inch),
        Paragraph("Summary", styles["Heading2"]),
        *as_vertical_table(summary_rows),
        Spacer(1, 0.3 * inch),
        Paragraph("Events", styles["Heading2"]),
        as_table(event_rows),
    ]

    doc.build(story)
    return buffer.getvalue()
