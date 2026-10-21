
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

    # Legacy fields kept for compatibility.
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

    # Relationship workflow
    relationship_type = db.Column(
        db.String(50),
        nullable=False,
        default="no_relationship"
    )

    opposing_company = db.Column(
        db.String(120),
        nullable=True
    )

    # Existing contact-request flags.
    # These remain for compatibility with existing routes.
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

    # Activity tracking dates
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

    # Notes
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

    # Existing relationships
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

    # NEW: Contact request history
    contact_requests = db.relationship(
        "ContactRequest",
        backref="customer",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="ContactRequest.requested_at.desc()"
    )


# ==========================================
# NEW: CONTACT REQUEST TRACKING
# ==========================================

class ContactRequest(db.Model):
    __tablename__ = "contact_requests"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # Customer associated with the request.
    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customers.id"),
        nullable=False,
        index=True
    )

    # Rep who originally requested contacts.
    requested_by = db.Column(
        db.Integer,
        nullable=False,
        index=True
    )

    # When the request was submitted.
    requested_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    # pending / completed / reviewed
    status = db.Column(
        db.String(30),
        nullable=False,
        default="pending",
        index=True
    )

    # Manager who completed the request.
    completed_by = db.Column(
        db.Integer,
        nullable=True
    )

    completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # Rep acknowledgement.
    reviewed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # Optional manager message to the rep.
    manager_notes = db.Column(
        db.Text,
        nullable=True
    )

    # Optional rep notes on the request.
    request_notes = db.Column(
        db.Text,
        nullable=True
    )

    def mark_completed(self, manager_user_id, notes=None):
        """Mark a pending request completed."""
        if self.status != "pending":
            raise ValueError(
                "Only pending requests can be completed."
            )

        self.status = "completed"
        self.completed_by = manager_user_id
        self.completed_at = datetime.utcnow()

        if notes is not None:
            self.manager_notes = notes

    def mark_reviewed(self):
        """Acknowledge a completed request."""
        if self.status != "completed":
            raise ValueError(
                "Only completed requests can be reviewed."
            )

        self.status = "reviewed"
        self.reviewed_at = datetime.utcnow()


# ==========================================
# CONTACTS
# ==========================================

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


# ==========================================
# CONTACT PHONE NUMBERS
# ==========================================

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


# ==========================================
# CUSTOMER DOCUMENTS
# ==========================================

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

    user_id = db.Column(
        db.Integer,
        nullable=False
    )

    original_filename = db.Column(
        db.String(255),
        nullable=False
    )

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


# ==========================================
# ACTIVITY LOGS
# ==========================================

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


# ==========================================
# FLEET INFORMATION
# ==========================================

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


# ==========================================
# SALES LEADS
# ==========================================

class SalesLead(db.Model):
    __tablename__ = "sales_leads"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

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


# ==========================================
# LEAD SALES PROCESS
# ==========================================

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

    # 4. Buying Process
    buying_process = db.Column(
        db.Text,
        nullable=True
    )

    buying_process_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 5. Solve Pain Points
    solve_pain_points = db.Column(
        db.Text,
        nullable=True
    )

    solve_pain_points_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 6. Proposal
    proposal = db.Column(
        db.Text,
        nullable=True
    )

    proposal_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 7. Closing Plan
    close_plan = db.Column(
        db.Text,
        nullable=True
    )

    close_completed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # 8. Lost Reason
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
