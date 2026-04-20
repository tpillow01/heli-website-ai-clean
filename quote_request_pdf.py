# quote_request_pdf.py

from typing import Dict, List, Tuple
from datetime import datetime
import io
import html

from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


QUOTE_FIELD_LABELS: List[Tuple[str, str]] = [
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
    ("seat_suspension", "Seat Suspension"),
    ("headlights", "Front Work Lights"),
    ("back_up_alarm", "Back Up Alarm"),
    ("strobe", "Strobe"),
    ("rear_work_light", "Rear Work Light"),
    ("blue_light_front", "Blue Light Front"),
    ("blue_light_rear", "Blue Light Rear"),
    ("red_curtain_lights", "Red Curtain Lights"),
    ("battery", "Battery"),
    ("charger", "Charger"),
    ("local_options", "Local Options"),
    ("expected_delivery", "Customer Requested Delivery"),
    ("lease_type", "Lease Type"),
    ("annual_hours", "Annual Hours"),
    ("lease_term", "Lease Term (Months)"),
    ("notes", "Notes / Special Requirements"),
    ("salesperson_name", "Salesperson Name"),
]

DEMO_FIELD_LABELS: List[Tuple[str, str]] = [
    ("ordered_by", "Ordered By"),
    ("company_name", "Company Name"),
    ("ship_to_address", "Ship To Address"),
    ("contact_name", "Contact Name"),
    ("phone", "Phone"),
    ("bill_to_address", "Bill To Address"),
    ("company_phone_fax", "Company Phone/Fax"),
    ("cartage", "Cartage (Dock / Ground)"),
    ("po_number", "PO #"),
    ("quantity", "Quantity"),
    ("description_model", "Description / Model"),
    ("rate", "Rate (Daily/Weekly/Monthly)"),
    ("freight_charges", "Freight Charges"),
    ("fork_length", "Fork Length"),
    ("lbr", "LBR"),
    ("side_shifter", "Side Shifter"),
    ("backup_alarm", "Back-up Alarm"),
    ("headlights", "Work Lights"),
    ("tires", "Tires"),
    ("power_type", "LP or Gas / Electric"),
    ("need_lp_tank", "Need LP Tank"),
    ("mast_height", "Mast Height"),
    ("mast_type", "Mast Type"),
    ("connector", "Connector (if electric)"),
    ("need_charger", "Need Charger (if electric)"),
    ("input_volts", "Input Volts"),
    ("phase", "Phase"),
    ("special_instructions", "Special Instructions"),
]

RENTAL_FIELD_LABELS: List[Tuple[str, str]] = [
    ("ordered_by", "Ordered By"),
    ("company_name", "Company Name"),
    ("company_phone_fax", "Company Phone/Fax"),
    ("ship_to_address", "Ship To Address"),
    ("contact_name", "Contact Name"),
    ("phone", "Phone"),
    ("bill_to_address", "Bill To Address"),
    ("description_model", "Description / Model"),
    ("quantity", "Quantity"),
    ("po_number", "PO #"),
    ("cartage", "Cartage (Dock / Ground)"),
    ("rate", "Rate (Daily/Weekly/Monthly)"),
    ("freight_charges", "Freight Charges"),
    ("fork_length", "Fork Length"),
    ("lbr", "LBR"),
    ("side_shifter", "Side Shifter"),
    ("backup_alarm", "Back-up Alarm"),
    ("headlights", "Work Lights"),
    ("tires", "Tires"),
    ("power_type", "LP or Gas / Electric"),
    ("need_lp_tank", "Need LP Tank"),
    ("mast_height", "Mast Height"),
    ("mast_type", "Mast Type"),
    ("connector", "Connector (if electric)"),
    ("need_charger", "Need Charger (if electric)"),
    ("input_volts", "Input Volts"),
    ("phase", "Phase"),
    ("special_instructions", "Special Instructions"),
]

# Used Equipment: show internal info on PDF only
USED_FIELD_LABELS: List[Tuple[str, str]] = [
    ("customer_name", "Customer"),
    ("address", "Address"),
    ("city_state_zip", "City / State / Zip"),
    ("contact_name", "Contact Name"),
    ("budget_price", "Customer Budget"),
    ("model", "Model"),
    ("fuel_type", "Fuel"),
    ("battery_voltage", "Electric 36/48 Volt"),
    ("line_voltage", "Line Voltage"),
    ("mast_height", "Mast Height"),
    ("mast_type", "Mast Type"),
    ("fork_size", "Fork Size"),
    ("options_need", "Options Need"),
    ("additional_notes", "Additional Notes"),
    ("__SECTION__", "Internal Information"),
    ("quote_number", "Quote #"),
    ("asset_number", "Asset #"),
    ("serial_number", "Serial #"),
]


def _clean(v) -> str:
    return ("" if v is None else str(v)).strip()


def _to_multiline_paragraph(text: str, style: ParagraphStyle, default_text: str = "") -> Paragraph:
    cleaned = _clean(text)
    if not cleaned:
        cleaned = default_text

    escaped = html.escape(cleaned)
    escaped = escaped.replace("\r\n", "\n").replace("\r", "\n")
    escaped = escaped.replace("\n", "<br/>")

    return Paragraph(escaped, style)


def _build_request_pdf(title_text: str, field_labels: List[Tuple[str, str]], form_data: Dict[str, str]) -> bytes:
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

    title_style = styles["Title"]
    subtitle_style = styles["Normal"]

    label_style = ParagraphStyle(
        "FieldLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        spaceAfter=0,
        spaceBefore=0,
    )

    value_style = ParagraphStyle(
        "FieldValue",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        spaceAfter=0,
        spaceBefore=0,
        wordWrap="LTR",
    )

    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        spaceAfter=0,
        spaceBefore=0,
    )

    elements = []

    title = Paragraph(title_text, title_style)
    subtitle = Paragraph(
        datetime.now().strftime("Generated on %Y-%m-%d at %H:%M"),
        subtitle_style,
    )
    elements.extend([title, subtitle, Spacer(1, 18)])

    rows = []
    section_row_indexes = []

    for key, label in field_labels:
        if key == "__SECTION__":
            section_row_indexes.append(len(rows))
            rows.append([Paragraph(html.escape(label), section_style), ""])
        else:
            value_text = _clean(form_data.get(key))

            # Notes and similar long-text fields should always render as wrapping paragraphs
            if key in {"notes", "special_instructions", "additional_notes", "local_options", "attachment"}:
                value_cell = _to_multiline_paragraph(value_text, value_style, default_text="-")
            else:
                value_cell = _to_multiline_paragraph(value_text, value_style, default_text="-")

            label_cell = Paragraph(html.escape(label), label_style)
            rows.append([label_cell, value_cell])

    table = Table(rows, colWidths=[180, 350], repeatRows=0)

    base_style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    # Style section header rows (Used Equipment: Internal Information)
    for r in section_row_indexes:
        base_style += [
            ("SPAN", (0, r), (1, r)),
            ("BACKGROUND", (0, r), (1, r), colors.lightgrey),
        ]

    table.setStyle(TableStyle(base_style))
    elements.append(table)

    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def build_quote_request_pdf(form_data: Dict[str, str]) -> bytes:
    if not _clean(form_data.get("fuel_voltage")):
        fuel = _clean(form_data.get("fuel_type"))
        volts = _clean(form_data.get("battery_voltage"))
        if fuel and volts:
            form_data = dict(form_data)
            form_data["fuel_voltage"] = f"{fuel} / {volts}"
        elif fuel:
            form_data = dict(form_data)
            form_data["fuel_voltage"] = fuel

    return _build_request_pdf("HELI Quote Request", QUOTE_FIELD_LABELS, form_data)


def build_demo_request_pdf(form_data: Dict[str, str]) -> bytes:
    return _build_request_pdf("HELI Demo Request", DEMO_FIELD_LABELS, form_data)


def build_rental_request_pdf(form_data: Dict[str, str]) -> bytes:
    return _build_request_pdf("HELI Rental Request", RENTAL_FIELD_LABELS, form_data)


def build_used_equipment_request_pdf(form_data: Dict[str, str]) -> bytes:
    return _build_request_pdf("Used Equipment Request", USED_FIELD_LABELS, form_data)


def build_request_pdf(form_data: Dict[str, str], request_type: str) -> bytes:
    rt = (request_type or "").strip().lower()
    if rt == "demo":
        return build_demo_request_pdf(form_data)
    if rt == "rental":
        return build_rental_request_pdf(form_data)
    if rt == "used":
        return build_used_equipment_request_pdf(form_data)
    return build_quote_request_pdf(form_data)