import os
import re
import pandas as pd
from datetime import datetime

from app import app, db
from models import Customer, Contact, ActivityLog


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "daily_sales_activity_master.xlsx")


def normalize_header(header: str) -> str:
    return str(header).strip().lower().replace(" ", "_")


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_date(value):
    if pd.isna(value) or value == "":
        return ""

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()

    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    try:
        parsed = pd.to_datetime(text)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return text


def split_location(location_text):
    """
    If location is '123 Main St, Indianapolis' this will split it.
    If there is no city, it stores the whole value in address.
    """
    raw = clean_value(location_text)
    if not raw:
        return "", ""

    parts = [part.strip() for part in raw.split(",", 1)]
    if len(parts) == 2:
        return parts[0], parts[1]

    return raw, ""


def get_or_create_customer(company_name, location, sales_rep):
    address, city = split_location(location)

    customer = Customer.query.filter(
        Customer.company_name.ilike(company_name)
    ).first()

    if not customer:
        customer = Customer(
            company_name=company_name,
            address=address or None,
            city=city or None,
            assigned_rep=sales_rep or None,
            status="Prospect",
            priority_level="Medium",
        )
        db.session.add(customer)
        db.session.flush()
        return customer, True

    # Fill in missing values only
    if not customer.address and address:
        customer.address = address
    if not customer.city and city:
        customer.city = city
    if not customer.assigned_rep and sales_rep:
        customer.assigned_rep = sales_rep

    return customer, False


def get_or_create_contact(customer_id, contact_name, phone, email):
    if not contact_name and not email and not phone:
        return None, False

    contact = None

    if email:
        contact = Contact.query.filter_by(customer_id=customer_id, email=email).first()

    if not contact and contact_name:
        contact = Contact.query.filter_by(customer_id=customer_id, name=contact_name).first()

    if not contact:
        contact = Contact(
            customer_id=customer_id,
            name=contact_name or "Unknown Contact",
            phone=phone or None,
            email=email or None,
        )
        db.session.add(contact)
        db.session.flush()
        return contact, True

    if not contact.phone and phone:
        contact.phone = phone
    if not contact.email and email:
        contact.email = email

    return contact, False


def activity_exists(customer_id, activity_date, activity_type, summary, rep_name):
    return ActivityLog.query.filter_by(
        customer_id=customer_id,
        activity_date=activity_date or None,
        activity_type=activity_type or None,
        summary=summary or None,
        rep_name=rep_name or None,
    ).first() is not None


def create_activity(customer_id, activity_date, activity_type, summary, next_step, rep_name):
    log = ActivityLog(
        customer_id=customer_id,
        activity_type=activity_type or "General Activity",
        summary=summary or "Imported activity",
        next_step=next_step or None,
        activity_date=activity_date or None,
        rep_name=rep_name or None,
    )
    db.session.add(log)
    return log


def update_customer_dates(customer, activity_date):
    if not activity_date:
        return

    # Always update last touch/contact to the imported activity date
    # because this sheet represents rep outreach history.
    customer.last_touch_date = activity_date
    customer.last_contact_date = activity_date


def validate_columns(df):
    required = {
        "date",
        "sales_rep",
        "customer",
        "location",
        "contact_name",
        "phone",
        "email",
        "outreach_method",
        "notes",
        "next_steps",
    }

    actual = set(df.columns)
    missing = required - actual

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def main():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    df = pd.read_excel(FILE_PATH)
    df.columns = [normalize_header(col) for col in df.columns]

    validate_columns(df)

    created_customers = 0
    created_contacts = 0
    created_activities = 0
    skipped_activities = 0

    with app.app_context():
        for _, row in df.iterrows():
            activity_date = normalize_date(row.get("date", ""))
            sales_rep = clean_value(row.get("sales_rep", ""))
            company_name = clean_value(row.get("customer", ""))
            location = clean_value(row.get("location", ""))
            contact_name = clean_value(row.get("contact_name", ""))
            phone = clean_value(row.get("phone", ""))
            email = clean_value(row.get("email", "")).lower()
            outreach_method = clean_value(row.get("outreach_method", ""))
            notes = clean_value(row.get("notes", ""))
            next_steps = clean_value(row.get("next_steps", ""))

            if not company_name:
                continue

            customer, was_customer_created = get_or_create_customer(
                company_name=company_name,
                location=location,
                sales_rep=sales_rep,
            )
            if was_customer_created:
                created_customers += 1

            _, was_contact_created = get_or_create_contact(
                customer_id=customer.id,
                contact_name=contact_name,
                phone=phone,
                email=email,
            )
            if was_contact_created:
                created_contacts += 1

            if activity_exists(
                customer_id=customer.id,
                activity_date=activity_date,
                activity_type=outreach_method,
                summary=notes,
                rep_name=sales_rep,
            ):
                skipped_activities += 1
                continue

            create_activity(
                customer_id=customer.id,
                activity_date=activity_date,
                activity_type=outreach_method,
                summary=notes,
                next_step=next_steps,
                rep_name=sales_rep,
            )
            created_activities += 1

            update_customer_dates(customer, activity_date)

        db.session.commit()

    print("Import complete.")
    print(f"Customers created: {created_customers}")
    print(f"Contacts created: {created_contacts}")
    print(f"Activities created: {created_activities}")
    print(f"Duplicate activities skipped: {skipped_activities}")


if __name__ == "__main__":
    main()