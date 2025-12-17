# quote_request_pdf.py

from typing import Dict, List, Tuple
from datetime import datetime
import io

from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


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


# DEMO: no cartage, no charger_set_for, mast split, headlights label is "Work Lights"
DEMO_FIELD_LABELS: List[Tuple[str, str]] = [
    ("ordered_by", "Ordered By"),
    ("company_name", "Company Name"),
    ("ship_to_address", "Ship To Address"),
    ("contact_name", "Contact Name"),
    ("phone", "Phone"),
    ("bill_to_address", "Bill To Address"),
    ("company_phone_fax", "Company Phone/Fax"),
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


# RENTAL: keeps cartage, no charger_set_for, mast split, headlights label is "Work Lights"
RENTAL_FIELD_LABELS: List[Tuple[str, str]] = [
    ("ordered_by", "Ordered By"),
    ("company_name", "Company Name"),
    ("ship_to_address", "Ship To Address"),
    ("contact_name", "Contact Name"),
    ("phone", "Phone"),
    ("bill_to_address", "Bill To Address"),
    ("cartage", "Cartage (Dock / Ground)"),
    ("company_phone_fax", "Company Phone/Fax"),
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


def _clean(v) -> str:
    return ("" if v is None else str(v)).strip()


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
    elements = []

    title = Paragraph(title_text, styles["Title"])
    subtitle = Paragraph(
        datetime.now().strftime("Generated on %Y-%m-%d at %H:%M"),
        styles["Normal"],
    )
    elements.extend([title, subtitle, Spacer(1, 18)])

    rows = []
    for key, label in field_labels:
        value = _clean(form_data.get(key))
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


def build_request_pdf(form_data: Dict[str, str], request_type: str) -> bytes:
    rt = (request_type or "").strip().lower()
    if rt == "demo":
        return build_demo_request_pdf(form_data)
    if rt == "rental":
        return build_rental_request_pdf(form_data)
    return build_quote_request_pdf(form_data)
