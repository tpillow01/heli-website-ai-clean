from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Customer(db.Model):
    __tablename__ = "customers"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)

    company_name = db.Column(db.String(200), nullable=False)

    address = db.Column(db.String(200), nullable=True)
    city = db.Column(db.String(100), nullable=True)
    state = db.Column(db.String(50), nullable=True)
    zip_code = db.Column(db.String(20), nullable=True)
    county = db.Column(db.String(100), nullable=True)

    assigned_rep = db.Column(db.String(100), nullable=True)

    # Legacy fields kept for compatibility with existing data/forms.
    status = db.Column(db.String(50), nullable=True, default="Prospect")
    priority_level = db.Column(db.String(50), nullable=True, default="Medium")

    # Relationship-driven workflow fields
    relationship_type = db.Column(
        db.String(50),
        nullable=False,
        default="no_relationship"
    )

    opposing_company = db.Column(
        db.String(120),
        nullable=True
    )

    # Reps can flag customer profiles that need contact information.
    needs_contacts = db.Column(
        db.Boolean,
        nullable=False,
        default=False
    )

    needs_contacts_flagged_by = db.Column(
        db.Integer,
        nullable=True
    )

    needs_contacts_flagged_at = db.Column(
        db.DateTime,
        nullable=True
    )

    last_contact_date = db.Column(
        db.String(50),
        nullable=True
    )

    follow_up_date = db.Column(
        db.String(50),
        nullable=True
    )

    last_touch_date = db.Column(
        db.String(50),
        nullable=True
    )

    notes = db.Column(
        db.Text,
        nullable=True
    )

    quote_notes = db.Column(
        db.Text,
        nullable=True
    )

    service_notes = db.Column(
        db.Text,
        nullable=True
    )

    rental_notes = db.Column(
        db.Text,
        nullable=True
    )

    pm_notes = db.Column(
        db.Text,
        nullable=True
    )

    latitude = db.Column(
        db.Float,
        nullable=True
    )

    longitude = db.Column(
        db.Float,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    contacts = db.relationship(
        "Contact",
        backref="customer",
        lazy=True,
        cascade="all, delete-orphan"
    )

    activity_logs = db.relationship(
        "ActivityLog",
        backref="customer",
        lazy=True,
        cascade="all, delete-orphan"
    )

    fleet_info = db.relationship(
        "FleetInfo",
        backref="customer",
        lazy=True,
        cascade="all, delete-orphan"
    )


class Contact(db.Model):
    __tablename__ = "contacts"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        nullable=False
    )

    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customers.id"),
        nullable=False
    )

    name = db.Column(
        db.String(120),
        nullable=False
    )

    title = db.Column(
        db.String(120),
        nullable=True
    )

    # Existing primary phone field is kept for compatibility.
    phone = db.Column(
        db.String(50),
        nullable=True
    )

    phone_type = db.Column(
        db.String(50),
        nullable=True,
        default="Mobile"
    )

    email = db.Column(
        db.String(120),
        nullable=True
    )

    contact_status = db.Column(
        db.String(50),
        nullable=False,
        default="Active - Still in Position"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    phone_numbers = db.relationship(
        "ContactPhone",
        backref="contact",
        lazy=True,
        cascade="all, delete-orphan"
    )


class ContactPhone(db.Model):
    __tablename__ = "contact_phones"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    contact_id = db.Column(
        db.Integer,
        db.ForeignKey("contacts.id"),
        nullable=False
    )

    phone_type = db.Column(
        db.String(50),
        nullable=True
    )

    phone_number = db.Column(
        db.String(50),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class ActivityLog(db.Model):
    __tablename__ = "activity_logs"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        nullable=False
    )

    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customers.id"),
        nullable=False
    )

    activity_type = db.Column(
        db.String(100),
        nullable=False
    )

    summary = db.Column(
        db.Text,
        nullable=False
    )

    next_step = db.Column(
        db.String(200),
        nullable=True
    )

    activity_date = db.Column(
        db.String(50),
        nullable=True
    )

    rep_name = db.Column(
        db.String(100),
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class FleetInfo(db.Model):
    __tablename__ = "fleet_info"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        nullable=False
    )

    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customers.id"),
        nullable=False
    )

    make = db.Column(
        db.String(100),
        nullable=True
    )

    model = db.Column(
        db.String(100),
        nullable=True
    )

    capacity = db.Column(
        db.String(50),
        nullable=True
    )

    fuel_type = db.Column(
        db.String(50),
        nullable=True
    )

    quantity = db.Column(
        db.Integer,
        nullable=True
    )

    notes = db.Column(
        db.Text,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )