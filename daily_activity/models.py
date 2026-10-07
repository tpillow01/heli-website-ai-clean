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
    status = db.Column(
        db.String(50),
        nullable=True,
        default="Prospect"
    )

    priority_level = db.Column(
        db.String(50),
        nullable=True,
        default="Medium"
    )

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

    documents = db.relationship(
        "CustomerDocument",
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


class CustomerDocument(db.Model):
    __tablename__ = "customer_documents"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customers.id"),
        nullable=False
    )

    # User who uploaded the document
    user_id = db.Column(
        db.Integer,
        nullable=False
    )

    # Friendly name shown to the user
    original_filename = db.Column(
        db.String(255),
        nullable=False
    )

    # Unique filename used on the Render persistent disk
    stored_filename = db.Column(
        db.String(255),
        nullable=False,
        unique=True
    )

    document_type = db.Column(
        db.String(100),
        nullable=True,
        default="Other"
    )

    mime_type = db.Column(
        db.String(150),
        nullable=True
    )

    file_size = db.Column(
        db.Integer,
        nullable=True
    )

    uploaded_by = db.Column(
        db.String(120),
        nullable=True
    )

    uploaded_at = db.Column(
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

class SalesLead(db.Model):
    __tablename__ = "sales_leads"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # User ID of the sales rep who owns this lead.
    assigned_user_id = db.Column(
        db.Integer,
        nullable=False
    )

    assigned_rep = db.Column(
        db.String(100),
        nullable=False
    )

    lead_source = db.Column(
        db.String(100),
        nullable=True
    )

    company_name = db.Column(
        db.String(200),
        nullable=False
    )

    address = db.Column(
        db.String(200),
        nullable=True
    )

    city = db.Column(
        db.String(100),
        nullable=True
    )

    county = db.Column(
        db.String(100),
        nullable=True
    )

    status = db.Column(
        db.String(50),
        nullable=False,
        default="Active"
    )

    current_stage = db.Column(
        db.String(100),
        nullable=False,
        default="Homework"
    )

    general_notes = db.Column(
        db.Text,
        nullable=True
    )

    next_action = db.Column(
        db.String(255),
        nullable=True
    )

    next_action_date = db.Column(
        db.String(50),
        nullable=True
    )

    # Used later if the lead is converted into a customer.
    converted_customer_id = db.Column(
        db.Integer,
        nullable=True
    )

    created_by_user_id = db.Column(
        db.Integer,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    sales_process = db.relationship(
        "LeadSalesProcess",
        backref="lead",
        uselist=False,
        cascade="all, delete-orphan"
    )


class LeadSalesProcess(db.Model):
    __tablename__ = "lead_sales_process"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    lead_id = db.Column(
        db.Integer,
        db.ForeignKey("sales_leads.id"),
        nullable=False,
        unique=True
    )

    # 1. Homework
    homework = db.Column(
        db.Text,
        nullable=True
    )

    homework_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 2. Rapport
    rapport = db.Column(
        db.Text,
        nullable=True
    )

    rapport_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 3. Pain Point
    pain_point = db.Column(
        db.Text,
        nullable=True
    )

    pain_point_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 4. What is the buying process?
    buying_process = db.Column(
        db.Text,
        nullable=True
    )

    buying_process_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 5. How do we solve the pain points?
    solve_pain_points = db.Column(
        db.Text,
        nullable=True
    )

    solve_pain_points_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 6. What is our proposal?
    proposal = db.Column(
        db.Text,
        nullable=True
    )

    proposal_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 7. How do we close?
    close_plan = db.Column(
        db.Text,
        nullable=True
    )

    close_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 8. If lost, why did we lose?
    lost_reason = db.Column(
        db.Text,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )  