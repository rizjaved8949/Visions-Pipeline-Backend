from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .config import (
    REPORT_DIR,
)

from .storage import STORE


def _load(
    session_id,
):

    session = STORE.get_session(
        session_id
    )

    if session is None:

        raise ValueError(
            "Session not found"
        )

    violations = (
        STORE.list_violations(
            session_id
        )
    )

    return (
        session,
        violations,
    )


def create_csv(
    session_id,
):

    session, violations = (
        _load(
            session_id
        )
    )


    output = (
        REPORT_DIR /
        f"{session_id}_violations.csv"
    )


    pd.DataFrame(
        violations
    ).to_csv(
        output,
        index=False,
    )


    return output


def create_xlsx(
    session_id,
):

    session, violations = (
        _load(
            session_id
        )
    )


    output = (
        REPORT_DIR /
        f"{session_id}_report.xlsx"
    )


    summary = (
        session.get(
            "summary",
            {}
        )
    )


    summary_rows = [
        {
            "field": "session_id",
            "value": session_id,
        },
        {
            "field": "camera_id",
            "value": session.get(
                "camera_id"
            ),
        },
        {
            "field": "status",
            "value": session.get(
                "status"
            ),
        },
        {
            "field": "started_at",
            "value": session.get(
                "started_at"
            ),
        },
        {
            "field": "ended_at",
            "value": session.get(
                "ended_at"
            ),
        },
        {
            "field": "staff_detected",
            "value": summary.get(
                "staff_detected"
            ),
        },
        {
            "field": "fully_compliant",
            "value": summary.get(
                "fully_compliant"
            ),
        },
        {
            "field": "compliance_score",
            "value": summary.get(
                "compliance_score"
            ),
        },
    ]


    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        pd.DataFrame(
            summary_rows
        ).to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )


        pd.DataFrame(
            violations
        ).to_excel(
            writer,
            sheet_name="Violations",
            index=False,
        )


        persons = summary.get(
            "persons",
            []
        )


        pd.DataFrame(
            persons
        ).to_excel(
            writer,
            sheet_name="Persons",
            index=False,
        )


    return output


def create_pdf(
    session_id,
):

    session, violations = (
        _load(
            session_id
        )
    )


    output = (
        REPORT_DIR /
        f"{session_id}_report.pdf"
    )


    summary = session.get(
        "summary",
        {}
    )


    styles = (
        getSampleStyleSheet()
    )


    document = (
        SimpleDocTemplate(
            str(output),
            pagesize=A4,
        )
    )


    story = []


    story.append(
        Paragraph(
            "Kitchen PPE Compliance Report",
            styles["Title"],
        )
    )


    story.append(
        Spacer(
            1,
            14,
        )
    )


    summary_data = [
        [
            "Session",
            session_id,
        ],
        [
            "Camera",
            session.get(
                "camera_id",
                "",
            ),
        ],
        [
            "Status",
            session.get(
                "status",
                "",
            ),
        ],
        [
            "Staff detected",
            summary.get(
                "staff_detected",
                "",
            ),
        ],
        [
            "Fully compliant",
            summary.get(
                "fully_compliant",
                "",
            ),
        ],
        [
            "Compliance score",
            (
                f"{summary.get('compliance_score')}%"
                if summary.get(
                    "compliance_score"
                )
                is not None
                else "Unknown"
            ),
        ],
    ]


    table = Table(
        summary_data,
        colWidths=[
            150,
            300,
        ],
    )


    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        0,
                        -1,
                    ),
                    colors.lightgrey,
                ),
            ]
        )
    )


    story.append(
        table
    )


    story.append(
        Spacer(
            1,
            18,
        )
    )


    story.append(
        Paragraph(
            "Violation History",
            styles["Heading2"],
        )
    )


    violation_data = [
        [
            "Staff",
            "Requirement",
            "Violation",
            "Severity",
            "Confidence",
            "Started",
        ]
    ]


    for event in violations:

        violation_data.append(
            [
                event.get(
                    "staff_label",
                    "",
                ),
                event.get(
                    "requirement",
                    "",
                ),
                event.get(
                    "violation_type",
                    "",
                ),
                event.get(
                    "severity",
                    "",
                ),
                (
                    f"{event.get('confidence', 0):.2f}"
                    if event.get(
                        "confidence"
                    )
                    is not None
                    else ""
                ),
                event.get(
                    "started_at",
                    "",
                )[:19],
            ]
        )


    if len(
        violation_data
    ) == 1:

        violation_data.append(
            [
                "-",
                "-",
                "No recorded violations",
                "-",
                "-",
                "-",
            ]
        )


    violation_table = Table(
        violation_data,
        repeatRows=1,
    )


    violation_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.4,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    colors.lightgrey,
                ),
                (
                    "FONTSIZE",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7,
                ),
            ]
        )
    )


    story.append(
        violation_table
    )


    story.append(
        Spacer(
            1,
            16,
        )
    )


    story.append(
        Paragraph(
            (
                "Monitored requirements: "
                "mask, gloves and hair cover. "
                "Apron monitoring is currently "
                "unsupported by the trained model."
            ),
            styles["Normal"],
        )
    )


    document.build(
        story
    )


    return output