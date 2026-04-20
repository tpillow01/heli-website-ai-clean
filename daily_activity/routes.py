import os
import re
from datetime import datetime, date, timedelta
from collections import defaultdict

from flask import (
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    abort,
)

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from . import daily_activity_bp
from .models import db, Customer, Contact, ActivityLog, FleetInfo

geolocator = Nominatim(user_agent="daily_activity_app")

DA_ENDPOINTS = {
    "dashboard",
    "customers",
    "add_customer",
    "customer_detail",
    "edit_customer",
    "delete_customer",
    "activity",
    "add_activity",
    "add_activity_for_customer",
    "add_contact",
    "edit_contact",
    "add_fleet",
    "complete_followup",
    "snooze_followup",
    "reschedule_followup",
    "map_page",
    "planner",
    "calendar_page",
}


@daily_activity_bp.context_processor
def inject_daily_activity_url_for():
    def scoped_url_for(endpoint, **kwargs):
        if "." not in endpoint and endpoint in DA_ENDPOINTS:
            endpoint = f"daily_activity.{endpoint}"
        return url_for(endpoint, **kwargs)

    return {"url_for": scoped_url_for}


def da_url(endpoint, **kwargs):
    if "." not in endpoint:
        endpoint = f"daily_activity.{endpoint}"
    return url_for(endpoint, **kwargs)


def current_user_id():
    return session.get("user_id")


def current_user_role():
    return session.get("role", "rep")


def user_is_manager():
    return current_user_role() == "manager"


def scoped_customer_query():
    query = Customer.query
    if not user_is_manager():
        query = query.filter(Customer.user_id == current_user_id())
    return query


def scoped_activity_query():
    query = ActivityLog.query
    if not user_is_manager():
        query = query.filter(ActivityLog.user_id == current_user_id())
    return query


def scoped_contact_query():
    query = Contact.query
    if not user_is_manager():
        query = query.filter(Contact.user_id == current_user_id())
    return query


def scoped_fleet_query():
    query = FleetInfo.query
    if not user_is_manager():
        query = query.filter(FleetInfo.user_id == current_user_id())
    return query


def get_customer_or_403(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    if not user_is_manager() and customer.user_id != current_user_id():
        abort(403)
    return customer


# ----------------------------
# Address / geocoding helpers
# ----------------------------

def build_full_address(address, city, state, zip_code):
    parts = [address, city, state, zip_code]
    return ", ".join([part.strip() for part in parts if part and part.strip()])


def geocode_customer_address(address, city, state, zip_code):
    full_address = build_full_address(address, city, state, zip_code)

    if not full_address:
        return None, None

    try:
        location = geolocator.geocode(full_address, timeout=10)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderServiceError):
        pass

    return None, None


# ----------------------------
# Date helpers
# ----------------------------

def parse_date_safe(value):
    if not value:
        return None

    if isinstance(value, date):
        return value

    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def days_until(date_value):
    parsed = parse_date_safe(date_value)
    if not parsed:
        return None
    return (parsed - date.today()).days


def days_since(date_value):
    parsed = parse_date_safe(date_value)
    if not parsed:
        return None
    return (date.today() - parsed).days


# ----------------------------
# Relationship / planning helpers
# ----------------------------

def normalize_relationship_type(value):
    raw = (value or "").strip().lower()

    if raw in {"current_customer", "current customer", "customer", "active"}:
        return "current_customer"

    if raw in {"quoted", "quoting", "quote"}:
        return "quoted"

    if raw in {"competitor_owned", "competitor owned", "competitor", "opposing"}:
        return "competitor_owned"

    if raw in {"no_relationship", "no relationship", "none", "new"}:
        return "no_relationship"

    return "no_relationship"


def derive_relationship_from_legacy_fields(status="", pin_color=""):
    status_value = (status or "").strip().lower()
    pin_value = (pin_color or "").strip().lower()

    if pin_value == "green":
        return "current_customer"
    if pin_value == "orange":
        return "quoted"
    if pin_value == "red":
        return "competitor_owned"
    if pin_value == "blue":
        return "no_relationship"

    if status_value in {"customer", "active"}:
        return "current_customer"
    if status_value in {"quoted"}:
        return "quoted"
    if status_value in {"prospect", "target", "cold call", "inactive", "follow-up needed"}:
        return "no_relationship"

    return "no_relationship"


def relationship_label(value):
    relationship = normalize_relationship_type(value)

    if relationship == "current_customer":
        return "Current Customer"
    if relationship == "quoted":
        return "Quoting"
    if relationship == "competitor_owned":
        return "Competitor-Owned"
    return "No Relationship"


def suggested_action(customer):
    relationship = normalize_relationship_type(customer.relationship_type)
    due_in = days_until(customer.follow_up_date)
    stale_days = days_since(customer.last_touch_date)

    if due_in is not None and due_in < 0:
        return "Immediate Follow-Up"

    if due_in == 0:
        return "Call Today"

    if relationship == "competitor_owned":
        return "Competitor Attack Visit"

    if relationship == "quoted":
        return "Quote Follow-Up"

    if relationship == "current_customer":
        if stale_days is not None and stale_days >= 30:
            return "Customer Visit"
        return "Account Check-In"

    if due_in is not None and due_in <= 7:
        return "Follow Up This Week"

    return "Prospecting Visit"


def build_customer_item(customer, rank=0):
    relationship = normalize_relationship_type(customer.relationship_type)
    due_in = days_until(customer.follow_up_date)
    stale_days = days_since(customer.last_touch_date)

    reasons = []

    if relationship == "competitor_owned":
        if customer.opposing_company:
            reasons.append(f"Uses {customer.opposing_company}")
        else:
            reasons.append("Competitor-held account")

    if relationship == "quoted":
        reasons.append("Quoted account")

    if relationship == "current_customer":
        reasons.append("Current customer")

    if due_in is not None:
        if due_in < 0:
            reasons.append("Overdue follow-up")
        elif due_in == 0:
            reasons.append("Due today")
        elif due_in <= 7:
            reasons.append("Due this week")

    if stale_days is not None and stale_days >= 30:
        reasons.append(f"No touch in {stale_days} days")

    return {
        "customer": customer,
        "rank": rank,
        "score": rank,
        "reasons": reasons,
        "relationship_type": relationship,
        "relationship_label": relationship_label(relationship),
        "opposing_company": customer.opposing_company or "",
        "days_until_follow_up": due_in,
        "days_since_last_touch": stale_days,
        "county": customer.county or "Unknown",
        "rep": customer.assigned_rep or "Unassigned",
        "suggested_action": suggested_action(customer),
    }


def sort_accounts_for_planning(customers, competitor_name=None):
    competitor_filter = (competitor_name or "").strip().lower()

    def sort_key(customer):
        relationship = normalize_relationship_type(customer.relationship_type)
        due_in = days_until(customer.follow_up_date)
        stale_days = days_since(customer.last_touch_date)
        opposing = (customer.opposing_company or "").strip().lower()

        competitor_match = 1 if competitor_filter and opposing == competitor_filter else 0

        relationship_rank = {
            "competitor_owned": 4,
            "quoted": 3,
            "current_customer": 2,
            "no_relationship": 1,
        }.get(relationship, 0)

        if due_in is None:
            due_bucket = 99
        elif due_in < 0:
            due_bucket = -2
        elif due_in == 0:
            due_bucket = -1
        else:
            due_bucket = due_in

        stale_value = -(stale_days or 0)

        return (
            -competitor_match,
            -relationship_rank,
            due_bucket,
            stale_value,
            (customer.county or "").lower(),
            (customer.company_name or "").lower(),
        )

    return sorted(customers, key=sort_key)


def build_planner_data(rep_name=None, county=None, opposing_company=None, max_stops=10):
    query = Customer.query

    if not user_is_manager():
        query = query.filter(Customer.user_id == current_user_id())

    if rep_name:
        query = query.filter(Customer.assigned_rep == rep_name)

    customers = query.order_by(Customer.company_name.asc()).all()

    if county:
        customers = [
            customer for customer in customers
            if (customer.county or "").strip().lower() == county.strip().lower()
        ]

    if opposing_company:
        customers = [
            customer for customer in customers
            if (customer.opposing_company or "").strip().lower() == opposing_company.strip().lower()
        ]

    ordered_customers = sort_accounts_for_planning(customers, competitor_name=opposing_company)
    all_items = [build_customer_item(customer, rank=index + 1) for index, customer in enumerate(ordered_customers)]

    overdue_accounts = [
        item for item in all_items
        if item["days_until_follow_up"] is not None and item["days_until_follow_up"] < 0
    ]

    stale_accounts = [
        item for item in all_items
        if item["days_since_last_touch"] is not None and item["days_since_last_touch"] >= 14
    ]

    competitor_accounts = [
        item for item in all_items
        if item["relationship_type"] == "competitor_owned"
    ]

    quoted_accounts = [
        item for item in all_items
        if item["relationship_type"] == "quoted"
    ]

    current_customer_accounts = [
        item for item in all_items
        if item["relationship_type"] == "current_customer"
    ]

    county_groups = defaultdict(list)
    for item in all_items:
        county_groups[item["county"]].append(item)

    county_summary = []
    for county_name, items in county_groups.items():
        county_summary.append({
            "county": county_name,
            "count": len(items),
            "customers": items[:5],
            "top_score": len(items),
        })

    county_summary.sort(key=lambda x: x["count"], reverse=True)

    top_recommendations = all_items[:max_stops]

    return {
        "rep_name": rep_name,
        "selected_county": county,
        "selected_opposing_company": opposing_company,
        "all_accounts": all_items,
        "scored_customers": all_items,
        "top_recommendations": top_recommendations,
        "overdue_accounts": overdue_accounts,
        "stale_accounts": stale_accounts,
        "competitor_accounts": competitor_accounts,
        "quoted_accounts": quoted_accounts,
        "current_customer_accounts": current_customer_accounts,
        "county_summary": county_summary,
    }


def build_planner_summary(planner_data):
    top_recommendations = planner_data["top_recommendations"]
    rep_name = planner_data["rep_name"]
    selected_county = planner_data["selected_county"]
    selected_opposing_company = planner_data["selected_opposing_company"]
    competitor_accounts = planner_data["competitor_accounts"]
    quoted_accounts = planner_data["quoted_accounts"]
    overdue_accounts = planner_data["overdue_accounts"]

    headline = f"Plan for {rep_name}" if rep_name else "Rep day plan"

    summary_lines = []

    if selected_opposing_company:
        summary_lines.append(
            f"Planner is focused on accounts tied to {selected_opposing_company}."
        )

    if selected_county:
        summary_lines.append(
            f"Route is narrowed to {selected_county} County."
        )

    if competitor_accounts:
        summary_lines.append(
            f"There are {len(competitor_accounts)} competitor-held account"
            f"{'' if len(competitor_accounts) == 1 else 's'} in this view."
        )

    if quoted_accounts:
        summary_lines.append(
            f"There are {len(quoted_accounts)} quoted account"
            f"{'' if len(quoted_accounts) == 1 else 's'} that should be followed up."
        )

    if overdue_accounts:
        summary_lines.append(
            f"You have {len(overdue_accounts)} overdue follow-up"
            f"{'' if len(overdue_accounts) == 1 else 's'} needing attention."
        )

    if top_recommendations:
        first_three = ", ".join([
            item["customer"].company_name for item in top_recommendations[:3]
        ])
        summary_lines.append(f"Best first stops: {first_three}.")
    else:
        summary_lines.append("No accounts matched the current planner filters.")

    return {
        "headline": headline,
        "summary_lines": summary_lines,
    }


def build_reasonable_day(planner_data, max_stops=10):
    stops = planner_data["top_recommendations"][:max_stops]

    if planner_data["selected_county"]:
        chosen_area = planner_data["selected_county"]
    elif stops:
        chosen_area = stops[0]["county"]
    else:
        chosen_area = "All Areas"

    return {
        "chosen_county": chosen_area,
        "max_stops": max_stops,
        "stops": stops,
    }


def build_day_plan_summary(day_plan, selected_opposing_company=None):
    stops = day_plan["stops"]

    if not stops:
        return "No customers matched the selected workload."

    prefix = ""
    if selected_opposing_company:
        prefix = f"{selected_opposing_company} focus. "

    return (
        f"{prefix}Planned workload: {len(stops)} stop"
        f"{'' if len(stops) == 1 else 's'} in {day_plan['chosen_county']}."
    )


def build_dashboard_reminders():
    customers = scoped_customer_query().order_by(Customer.company_name.asc()).all()
    ordered_customers = sort_accounts_for_planning(customers)
    items = [build_customer_item(customer, rank=index + 1) for index, customer in enumerate(ordered_customers)]

    overdue_followups = [
        item for item in items
        if item["days_until_follow_up"] is not None and item["days_until_follow_up"] < 0
    ][:5]

    due_today = [
        item for item in items
        if item["days_until_follow_up"] == 0
    ][:5]

    due_this_week = [
        item for item in items
        if item["days_until_follow_up"] is not None and 0 < item["days_until_follow_up"] <= 7
    ][:5]

    stale_accounts = [
        item for item in items
        if item["days_since_last_touch"] is not None and item["days_since_last_touch"] >= 14
    ][:5]

    high_priority_attention = [
        item for item in items
        if item["relationship_type"] in {"competitor_owned", "quoted"}
        or (
            item["days_until_follow_up"] is not None
            and item["days_until_follow_up"] <= 7
        )
    ][:5]

    competitor_accounts = [
        item for item in items
        if item["relationship_type"] == "competitor_owned"
    ][:5]

    quoted_accounts = [
        item for item in items
        if item["relationship_type"] == "quoted"
    ][:5]

    current_customer_followups = [
        item for item in items
        if item["relationship_type"] == "current_customer"
        and item["days_until_follow_up"] is not None
        and item["days_until_follow_up"] <= 14
    ][:5]

    return {
        "overdue_followups": overdue_followups,
        "due_today": due_today,
        "due_this_week": due_this_week,
        "stale_accounts": stale_accounts,
        "high_priority_attention": high_priority_attention,
        "competitor_accounts": competitor_accounts,
        "quoted_accounts": quoted_accounts,
        "current_customer_followups": current_customer_followups,
    }


def build_calendar_followups(days_ahead=21):
    customers = scoped_customer_query().order_by(Customer.company_name.asc()).all()
    start_day = date.today()
    end_day = start_day + timedelta(days=days_ahead)

    grouped = defaultdict(list)

    for customer in customers:
        follow_up = parse_date_safe(customer.follow_up_date)
        if not follow_up:
            continue

        if start_day <= follow_up <= end_day:
            grouped[follow_up.strftime("%Y-%m-%d")].append(
                build_customer_item(customer)
            )

    grouped_sorted = []
    for follow_up_date, items in sorted(grouped.items(), key=lambda x: x[0]):
        grouped_sorted.append({
            "date": follow_up_date,
            "items": items,
            "count": len(items),
        })

    return grouped_sorted


def format_phone_number(value):
    raw = (value or "").strip()
    digits = re.sub(r"\D", "", raw)

    if not digits:
        return ""

    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]

    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 7:
        return f"{digits[:3]}-{digits[3:]}"
    return raw


# ----------------------------
# Activity update helpers
# ----------------------------

def should_update_last_contact(activity_type):
    value = (activity_type or "").strip().lower()

    customer_contact_types = [
        "call",
        "phone call",
        "site visit",
        "visit",
        "customer visit",
        "email",
        "meeting",
        "demo",
        "onsite visit",
    ]

    return value in customer_contact_types


def apply_activity_updates_to_customer(customer, activity_type, activity_date, follow_up_date=""):
    if activity_date:
        customer.last_touch_date = activity_date

        if should_update_last_contact(activity_type):
            customer.last_contact_date = activity_date

    if follow_up_date:
        customer.follow_up_date = follow_up_date


def build_activity_update_message(activity_type, activity_date, follow_up_date):
    parts = ["Activity saved successfully."]

    if activity_date:
        parts.append("Last touch was updated.")

        if should_update_last_contact(activity_type):
            parts.append("Last contact was updated.")

    if follow_up_date:
        parts.append(f"Next follow-up set for {follow_up_date}.")

    parts.append("Planner and reminders are now refreshed.")

    return " ".join(parts)


def redirect_back(default_endpoint="planner", **kwargs):
    return redirect(request.referrer or da_url(default_endpoint, **kwargs))


def shift_follow_up_date(customer, days=3):
    current = parse_date_safe(customer.follow_up_date)

    if current is None:
        current = date.today()

    customer.follow_up_date = (current + timedelta(days=days)).strftime("%Y-%m-%d")


# ----------------------------
# Routes
# ----------------------------

@daily_activity_bp.route("/")
def dashboard():
    customer_count = scoped_customer_query().count()
    contact_count = scoped_contact_query().count()
    activity_count = scoped_activity_query().count()
    fleet_count = scoped_fleet_query().count()

    recent_activity = scoped_activity_query().order_by(ActivityLog.created_at.desc()).limit(5).all()
    recent_customers = scoped_customer_query().order_by(Customer.created_at.desc()).limit(5).all()
    dashboard_reminders = build_dashboard_reminders()

    return render_template(
        "daily_activity/dashboard.html",
        customer_count=customer_count,
        contact_count=contact_count,
        activity_count=activity_count,
        fleet_count=fleet_count,
        recent_activity=recent_activity,
        recent_customers=recent_customers,
        dashboard_reminders=dashboard_reminders,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customers")
def customers():
    all_customers = scoped_customer_query().order_by(Customer.company_name.asc()).all()
    return render_template(
        "daily_activity/customers.html",
        customers=all_customers,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customers/add", methods=["GET", "POST"])
def add_customer():
    if request.method == "POST":
        company_name = request.form.get("company_name", "").strip()
        assigned_rep = request.form.get("assigned_rep", "").strip()
        address = request.form.get("address", "").strip()
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip()
        zip_code = request.form.get("zip_code", "").strip()
        county = request.form.get("county", "").strip()

        status = request.form.get("status", "").strip()
        priority_level = request.form.get("priority_level", "").strip()

        relationship_type = normalize_relationship_type(
            request.form.get("relationship_type", "").strip()
        )
        opposing_company = request.form.get("opposing_company", "").strip()

        if not relationship_type or relationship_type == "no_relationship":
            relationship_type = derive_relationship_from_legacy_fields(
                status=status,
                pin_color=request.form.get("pin_color", "").strip(),
            )

        last_contact_date = request.form.get("last_contact_date", "").strip()
        follow_up_date = request.form.get("follow_up_date", "").strip()
        last_touch_date = request.form.get("last_touch_date", "").strip()

        notes = request.form.get("notes", "").strip()
        quote_notes = request.form.get("quote_notes", "").strip()
        service_notes = request.form.get("service_notes", "").strip()
        rental_notes = request.form.get("rental_notes", "").strip()
        pm_notes = request.form.get("pm_notes", "").strip()

        if not company_name:
            flash("Company name is required.", "error")
            return redirect(da_url("add_customer"))

        latitude, longitude = geocode_customer_address(address, city, state, zip_code)

        customer = Customer(
            user_id=current_user_id(),
            company_name=company_name,
            assigned_rep=assigned_rep,
            address=address,
            city=city,
            state=state,
            zip_code=zip_code,
            county=county,
            status=status,
            priority_level=priority_level,
            relationship_type=relationship_type,
            opposing_company=opposing_company if relationship_type == "competitor_owned" else "",
            last_contact_date=last_contact_date,
            follow_up_date=follow_up_date,
            last_touch_date=last_touch_date,
            notes=notes,
            quote_notes=quote_notes,
            service_notes=service_notes,
            rental_notes=rental_notes,
            pm_notes=pm_notes,
            latitude=latitude,
            longitude=longitude,
        )

        db.session.add(customer)
        db.session.commit()

        contact_name = request.form.get("contact_name", "").strip()
        contact_phone = request.form.get("contact_phone", "").strip()
        contact_email = request.form.get("contact_email", "").strip()

        if contact_name or contact_phone or contact_email:
            contact = Contact(
                user_id=current_user_id(),
                customer_id=customer.id,
                name=contact_name or "Primary Contact",
                phone=format_phone_number(contact_phone),
                email=contact_email,
                title="Primary Contact",
            )
            db.session.add(contact)

        fleet_make = request.form.get("fleet_make", "").strip()
        fleet_model = request.form.get("fleet_model", "").strip()
        fleet_capacity = request.form.get("fleet_capacity", "").strip()
        fleet_fuel_type = request.form.get("fleet_fuel_type", "").strip()
        fleet_quantity_raw = request.form.get("fleet_quantity", "").strip()
        fleet_notes = request.form.get("fleet_notes", "").strip()

        if fleet_make or fleet_model or fleet_capacity or fleet_fuel_type or fleet_quantity_raw or fleet_notes:
            try:
                fleet_quantity = int(fleet_quantity_raw) if fleet_quantity_raw else None
            except ValueError:
                fleet_quantity = None

            fleet = FleetInfo(
                user_id=current_user_id(),
                customer_id=customer.id,
                make=fleet_make,
                model=fleet_model,
                capacity=fleet_capacity,
                fuel_type=fleet_fuel_type,
                quantity=fleet_quantity,
                notes=fleet_notes,
            )
            db.session.add(fleet)

        db.session.commit()

        flash("Customer added successfully.", "success")
        return redirect(da_url("customers"))

    return render_template("daily_activity/add_customer.html", is_manager=user_is_manager())


@daily_activity_bp.route("/customer/<int:customer_id>")
def customer_detail(customer_id):
    customer = get_customer_or_403(customer_id)
    return render_template(
        "daily_activity/customer_detail.html",
        customer=customer,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customer/<int:customer_id>/edit", methods=["GET", "POST"])
def edit_customer(customer_id):
    customer = get_customer_or_403(customer_id)

    if request.method == "POST":
        company_name = request.form.get("company_name", "").strip()
        address = request.form.get("address", "").strip()
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip()
        zip_code = request.form.get("zip_code", "").strip()

        if not company_name:
            flash("Company name is required.", "error")
            return redirect(da_url("edit_customer", customer_id=customer.id))

        latitude, longitude = geocode_customer_address(address, city, state, zip_code)

        status = request.form.get("status", "").strip()
        priority_level = request.form.get("priority_level", "").strip()

        relationship_type = normalize_relationship_type(
            request.form.get("relationship_type", "").strip()
        )
        opposing_company = request.form.get("opposing_company", "").strip()

        if not relationship_type or relationship_type == "no_relationship":
            relationship_type = derive_relationship_from_legacy_fields(status=status)

        customer.company_name = company_name
        customer.address = address
        customer.city = city
        customer.state = state
        customer.zip_code = zip_code
        customer.county = request.form.get("county", "").strip()
        customer.assigned_rep = request.form.get("assigned_rep", "").strip()
        customer.status = status
        customer.priority_level = priority_level
        customer.relationship_type = relationship_type
        customer.opposing_company = opposing_company if relationship_type == "competitor_owned" else ""
        customer.last_contact_date = request.form.get("last_contact_date", "").strip()
        customer.follow_up_date = request.form.get("follow_up_date", "").strip()
        customer.last_touch_date = request.form.get("last_touch_date", "").strip()
        customer.notes = request.form.get("notes", "").strip()
        customer.quote_notes = request.form.get("quote_notes", "").strip()
        customer.service_notes = request.form.get("service_notes", "").strip()
        customer.rental_notes = request.form.get("rental_notes", "").strip()
        customer.pm_notes = request.form.get("pm_notes", "").strip()

        if latitude is not None and longitude is not None:
            customer.latitude = latitude
            customer.longitude = longitude

        db.session.commit()

        flash("Customer updated successfully.", "success")
        return redirect(da_url("customer_detail", customer_id=customer.id))

    return render_template(
        "daily_activity/edit_customer.html",
        customer=customer,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customer/<int:customer_id>/contact/<int:contact_id>/edit", methods=["GET", "POST"])
def edit_contact(customer_id, contact_id):
    customer = get_customer_or_403(customer_id)

    if user_is_manager():
        contact = Contact.query.filter_by(
            id=contact_id,
            customer_id=customer.id,
        ).first_or_404()
    else:
        contact = Contact.query.filter_by(
            id=contact_id,
            customer_id=customer.id,
            user_id=current_user_id(),
        ).first_or_404()

    if request.method == "POST":
        contact_name = request.form.get("name", "").strip()
        contact_title = request.form.get("title", "").strip()
        contact_phone = request.form.get("phone", "").strip()
        contact_email = request.form.get("email", "").strip()

        if not contact_name:
            flash("Contact name is required.", "error")
            return redirect(da_url("edit_contact", customer_id=customer.id, contact_id=contact.id))

        contact.name = contact_name
        contact.title = contact_title
        contact.phone = format_phone_number(contact_phone)
        contact.email = contact_email

        db.session.commit()
        flash("Contact updated successfully.", "success")
        return redirect(da_url("customer_detail", customer_id=customer.id))

    return render_template(
        "daily_activity/edit_contact.html",
        customer=customer,
        contact=contact,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customer/<int:customer_id>/delete", methods=["POST"])
def delete_customer(customer_id):
    customer = get_customer_or_403(customer_id)

    if not user_is_manager():
        abort(403)

    Contact.query.filter_by(customer_id=customer.id).delete()
    ActivityLog.query.filter_by(customer_id=customer.id).delete()
    FleetInfo.query.filter_by(customer_id=customer.id).delete()

    db.session.delete(customer)
    db.session.commit()

    flash("Customer deleted successfully.", "success")
    return redirect(da_url("customers"))


@daily_activity_bp.route("/activity")
def activity():
    logs = scoped_activity_query().order_by(ActivityLog.created_at.desc()).all()
    return render_template(
        "daily_activity/activity.html",
        logs=logs,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/activity/add", methods=["GET", "POST"])
def add_activity():
    customers = scoped_customer_query().order_by(Customer.company_name.asc()).all()

    if request.method == "POST":
        customer_id = request.form.get("customer_id", "").strip()
        activity_type = request.form.get("activity_type", "").strip()
        summary = request.form.get("summary", "").strip()
        activity_date = request.form.get("activity_date", "").strip()
        next_step = request.form.get("next_step", "").strip()
        follow_up_date = request.form.get("follow_up_date", "").strip()
        rep_name = request.form.get("rep_name", "").strip()

        if not customer_id or not activity_type or not summary:
            flash("Customer, activity type, and summary are required.", "error")
            return redirect(da_url("add_activity"))

        customer = get_customer_or_403(int(customer_id))

        log = ActivityLog(
            user_id=current_user_id(),
            customer_id=customer.id,
            activity_type=activity_type,
            summary=summary,
            next_step=next_step,
            activity_date=activity_date,
            rep_name=rep_name,
        )

        db.session.add(log)

        apply_activity_updates_to_customer(
            customer=customer,
            activity_type=activity_type,
            activity_date=activity_date,
            follow_up_date=follow_up_date,
        )

        db.session.commit()

        flash(
            build_activity_update_message(activity_type, activity_date, follow_up_date),
            "success",
        )

        return redirect(da_url("planner"))

    return render_template(
        "daily_activity/add_activity.html",
        customers=customers,
        selected_customer_id=None,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customer/<int:customer_id>/add-activity", methods=["GET", "POST"])
def add_activity_for_customer(customer_id):
    customer = get_customer_or_403(customer_id)
    customers = scoped_customer_query().order_by(Customer.company_name.asc()).all()

    if request.method == "POST":
        activity_type = request.form.get("activity_type", "").strip()
        summary = request.form.get("summary", "").strip()
        activity_date = request.form.get("activity_date", "").strip()
        next_step = request.form.get("next_step", "").strip()
        follow_up_date = request.form.get("follow_up_date", "").strip()
        rep_name = request.form.get("rep_name", "").strip()

        if not activity_type or not summary:
            flash("Activity type and summary are required.", "error")
            return redirect(da_url("add_activity_for_customer", customer_id=customer.id))

        log = ActivityLog(
            user_id=current_user_id(),
            customer_id=customer.id,
            activity_type=activity_type,
            summary=summary,
            next_step=next_step,
            activity_date=activity_date,
            rep_name=rep_name,
        )

        db.session.add(log)

        apply_activity_updates_to_customer(
            customer=customer,
            activity_type=activity_type,
            activity_date=activity_date,
            follow_up_date=follow_up_date,
        )

        db.session.commit()

        flash(
            build_activity_update_message(activity_type, activity_date, follow_up_date),
            "success",
        )

        return redirect(da_url("customer_detail", customer_id=customer.id))

    return render_template(
        "daily_activity/add_activity.html",
        customers=customers,
        selected_customer_id=customer.id,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customer/<int:customer_id>/add-contact", methods=["GET", "POST"])
def add_contact(customer_id):
    customer = get_customer_or_403(customer_id)

    if request.method == "POST":
        name = request.form.get("name", "").strip()

        if not name:
            flash("Contact name is required.", "error")
            return redirect(da_url("add_contact", customer_id=customer.id))

        contact = Contact(
            user_id=current_user_id(),
            customer_id=customer.id,
            name=name,
            title=request.form.get("title", "").strip(),
            phone=format_phone_number(request.form.get("phone", "").strip()),
            email=request.form.get("email", "").strip(),
        )

        db.session.add(contact)
        db.session.commit()

        flash("Contact added successfully.", "success")
        return redirect(da_url("customer_detail", customer_id=customer.id))

    return render_template(
        "daily_activity/add_contact.html",
        customer=customer,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customer/<int:customer_id>/add-fleet", methods=["GET", "POST"])
def add_fleet(customer_id):
    customer = get_customer_or_403(customer_id)

    if request.method == "POST":
        qty_raw = request.form.get("quantity", "").strip()

        fleet = FleetInfo(
            user_id=current_user_id(),
            customer_id=customer.id,
            make=request.form.get("make", "").strip(),
            model=request.form.get("model", "").strip(),
            capacity=request.form.get("capacity", "").strip(),
            fuel_type=request.form.get("fuel_type", "").strip(),
            quantity=int(qty_raw) if qty_raw else None,
            notes=request.form.get("notes", "").strip(),
        )

        db.session.add(fleet)
        db.session.commit()

        flash("Fleet info added successfully.", "success")
        return redirect(da_url("customer_detail", customer_id=customer.id))

    return render_template(
        "daily_activity/add_fleet.html",
        customer=customer,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/customer/<int:customer_id>/complete-followup", methods=["POST"])
def complete_followup(customer_id):
    customer = get_customer_or_403(customer_id)

    customer.follow_up_date = None
    db.session.commit()

    flash(f"Follow-up marked complete for {customer.company_name}.", "success")
    return redirect_back("planner")


@daily_activity_bp.route("/customer/<int:customer_id>/snooze-followup", methods=["POST"])
def snooze_followup(customer_id):
    customer = get_customer_or_403(customer_id)
    days_raw = request.form.get("days", "3").strip()

    try:
        days = int(days_raw)
    except ValueError:
        days = 3

    shift_follow_up_date(customer, days=days)
    db.session.commit()

    flash(
        f"Follow-up snoozed for {customer.company_name} by {days} day{'s' if days != 1 else ''}.",
        "success",
    )
    return redirect_back("planner")


@daily_activity_bp.route("/customer/<int:customer_id>/reschedule-followup", methods=["POST"])
def reschedule_followup(customer_id):
    customer = get_customer_or_403(customer_id)
    new_follow_up_date = request.form.get("follow_up_date", "").strip()

    if not new_follow_up_date:
        flash("Please choose a new follow-up date.", "error")
        return redirect_back("planner")

    customer.follow_up_date = new_follow_up_date
    db.session.commit()

    flash(f"Follow-up rescheduled for {customer.company_name}.", "success")
    return redirect_back("planner")


@daily_activity_bp.route("/map")
def map_page():
    customers = scoped_customer_query().order_by(Customer.company_name.asc()).all()
    map_customers = []

    for customer in customers:
        if customer.latitude is None or customer.longitude is None:
            continue

        relationship = normalize_relationship_type(customer.relationship_type)

        map_customers.append({
            "id": customer.id,
            "company_name": customer.company_name,
            "address": customer.address or "",
            "city": customer.city or "",
            "state": customer.state or "",
            "county": customer.county or "",
            "assigned_rep": customer.assigned_rep or "",
            "status": customer.status or "",
            "priority_level": customer.priority_level or "",
            "relationship_type": relationship,
            "relationship_label": relationship_label(relationship),
            "opposing_company": customer.opposing_company or "",
            "follow_up_date": customer.follow_up_date or "",
            "last_touch_date": customer.last_touch_date or "",
            "latitude": customer.latitude,
            "longitude": customer.longitude,
        })

    return render_template(
        "daily_activity/map.html",
        customers=customers,
        map_customers=map_customers,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/planner")
def planner():
    rep_name = request.args.get("rep", "").strip() or None
    county = request.args.get("county", "").strip() or None
    opposing_company = request.args.get("opposing_company", "").strip() or None

    max_stops_raw = request.args.get("max_stops", "10").strip()
    try:
        max_stops = int(max_stops_raw)
    except ValueError:
        max_stops = 10

    if max_stops < 1:
        max_stops = 1
    if max_stops > 25:
        max_stops = 25

    planner_data = build_planner_data(
        rep_name=rep_name,
        county=county,
        opposing_company=opposing_company,
        max_stops=max_stops,
    )
    planner_summary = build_planner_summary(planner_data)

    day_plan = build_reasonable_day(
        planner_data=planner_data,
        max_stops=max_stops,
    )
    day_plan_summary = build_day_plan_summary(
        day_plan,
        selected_opposing_company=opposing_company,
    )

    visible_customers = scoped_customer_query().all()

    all_reps = sorted({
        customer.assigned_rep
        for customer in visible_customers
        if customer.assigned_rep
    })

    all_counties = sorted({
        customer.county
        for customer in visible_customers
        if customer.county
    })

    all_opposing_companies = sorted({
        customer.opposing_company
        for customer in visible_customers
        if customer.opposing_company
    })

    return render_template(
        "daily_activity/planner.html",
        planner_data=planner_data,
        planner_summary=planner_summary,
        day_plan=day_plan,
        day_plan_summary=day_plan_summary,
        reps=all_reps,
        counties=all_counties,
        opposing_companies=all_opposing_companies,
        selected_rep=rep_name,
        selected_county=county,
        selected_opposing_company=opposing_company,
        selected_max_stops=max_stops,
        is_manager=user_is_manager(),
    )


@daily_activity_bp.route("/calendar")
def calendar_page():
    followup_groups = build_calendar_followups(days_ahead=21)
    return render_template(
        "daily_activity/calendar.html",
        followup_groups=followup_groups,
        is_manager=user_is_manager(),
    )