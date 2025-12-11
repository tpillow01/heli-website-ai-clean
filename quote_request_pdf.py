# quote_request_pdf.py

from typing import Dict
from datetime import datetime
import io

from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Order + labels for the PDF layout
FIELD_LABELS = [
    ("customer_name", "Customer Name"),
    ("address", "Address"),
    ("city_state_zip", "City / State / Zip"),
    ("contact_name", "Contact Name"),
    ("model", "Model"),
    ("fuel_voltage", "Fuel / Voltage"),
    ("mast_ohl_mfh", "Mast OHL / MFH"),
    ("mast_type", "Mast Type"),
    ("attachment", "Attachment"),
    ("aux_valve", "Auxiliary Control Valve"),
    ("aux_hose", "Auxiliary Hose Take Up"),
    ("fork_type", "Fork Type"),
    ("fork_length", "Fork Length"),
    ("tires", "Tires"),
    ("tire_compound", "Tire Compound"),
    ("seat_suspension", "Seat Suspension"),
    ("headlights", "Headlights"),
    ("back_up_alarm", "Back Up Alarm"),
    ("strobe", "Strobe"),
    ("rear_work_light", "Rear Work Light"),
    ("blue_light_front", "Blue Light Front"),
    ("blue_light_rear", "Blue Light Rear"),
    ("red_curtain_lights", "Red Curtain Lights"),
    ("battery", "Battery"),
    ("charger", "Charger"),
    ("local_options", "Local Options"),
    ("expected_delivery", "Expected Delivery"),
    ("lease_type", "Lease Type"),
    ("annual_hours", "Annual Hours"),
    ("lease_term", "Lease Term (Months)"),
    ("notes", "Notes / Special Requirements"),
    ("salesperson_name", "Salesperson Name"),
]


def build_quote_request_pdf(form_data: Dict[str, str]) -> bytes:
    """
    Build a clean, 1-page quote request PDF from the submitted form data.
    Returns raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    elements = []

    # Title + timestamp
    title = Paragraph("HELI Quote Request", styles["Title"])
    subtitle = Paragraph(
        datetime.now().strftime("Generated on %Y-%m-%d at %H:%M"),
        styles["Normal"],
    )
    elements.extend([title, subtitle, Spacer(1, 18)])

    # Table rows for each field
    rows = []
    for key, label in FIELD_LABELS:
        value = (form_data.get(key) or "").strip()
        rows.append([label, value])

    table = Table(rows, colWidths=[180, 350])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    elements.append(table)
    doc.build(elements)

    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
