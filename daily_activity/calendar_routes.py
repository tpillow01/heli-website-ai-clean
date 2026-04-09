from __future__ import annotations

import importlib
import json
from datetime import datetime
from functools import wraps
from typing import Any

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for


calendar_bp = Blueprint("calendar_bp", __name__)


def _load_flask_login():
    try:
        return importlib.import_module("flask_login")
    except Exception:
        return None


def _login_required_fallback(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


_flask_login = _load_flask_login()
login_required = _flask_login.login_required if _flask_login and hasattr(_flask_login, "login_required") else _login_required_fallback
current_user = getattr(_flask_login, "current_user", None) if _flask_login else None


def _get_db():
    try:
        if "sqlalchemy" in current_app.extensions:
            ext = current_app.extensions["sqlalchemy"]
            if hasattr(ext, "db"):
                return ext.db
            return ext
    except Exception:
        pass

    try:
        extensions_module = importlib.import_module("extensions")
        if hasattr(extensions_module, "db"):
            return extensions_module.db
    except Exception:
        pass

    raise RuntimeError(
        "Could not resolve SQLAlchemy db instance. "
        "Make sure your app initializes SQLAlchemy and stores it in current_app.extensions "
        "or exposes db in extensions.py."
    )


def _get_event_model():
    module_candidates = [
        "models",
        "app",
    ]

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "Event"):
                return getattr(module, "Event")
        except Exception:
            continue

    raise RuntimeError(
        "Could not resolve Event model. "
        "Make sure your Event model exists in models.py or app.py and is named Event."
    )


def _current_user_id():
    try:
        if current_user is not None and hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
            return getattr(current_user, "id", None)
    except Exception:
        pass
    return None


def _serialize_event(event) -> dict[str, Any]:
    return {
        "id": getattr(event, "id", None),
        "title": getattr(event, "title", "") or "Untitled Event",
        "start": getattr(event, "start_time", None).isoformat() if getattr(event, "start_time", None) else None,
        "end": getattr(event, "end_time", None).isoformat() if getattr(event, "end_time", None) else None,
        "event_type": getattr(event, "event_type", None) or "Other",
        "location": getattr(event, "location", None) or "",
        "assigned_rep": getattr(event, "assigned_rep", None) or "",
        "description": getattr(event, "description", None) or "",
        "allDay": False,
    }


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    value = value.strip()
    if not value:
        return None

    formats = [
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _apply_user_filter(query, event_model):
    user_id = _current_user_id()

    if user_id is None:
        return query

    if hasattr(event_model, "user_id"):
        return query.filter(event_model.user_id == user_id)

    return query


@calendar_bp.route("/calendar", methods=["GET"])
@login_required
def calendar_page():
    event_model = _get_event_model()

    query = event_model.query
    query = _apply_user_filter(query, event_model)

    if hasattr(event_model, "start_time"):
        events = query.order_by(event_model.start_time.asc()).all()
    else:
        events = query.all()

    events_json = json.dumps([_serialize_event(event) for event in events])

    return render_template("calendar.html", events_json=events_json)


@calendar_bp.route("/add_event", methods=["POST"])
@login_required
def add_event():
    event_model = _get_event_model()
    db = _get_db()

    title = (request.form.get("title") or "").strip()
    event_type = (request.form.get("event_type") or "Other").strip()
    start_raw = request.form.get("start")
    end_raw = request.form.get("end")
    location = (request.form.get("location") or "").strip()
    assigned_rep = (request.form.get("assigned_rep") or "").strip()
    description = (request.form.get("description") or "").strip()

    start_time = _parse_datetime(start_raw)
    end_time = _parse_datetime(end_raw)

    if not title:
        flash("Event title is required.", "error")
        return redirect(url_for("calendar_bp.calendar_page"))

    if not start_time or not end_time:
        flash("Start and end date/time are required.", "error")
        return redirect(url_for("calendar_bp.calendar_page"))

    if end_time <= start_time:
        flash("End time must be after start time.", "error")
        return redirect(url_for("calendar_bp.calendar_page"))

    event = event_model(
        title=title,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time,
        location=location,
        assigned_rep=assigned_rep,
        description=description,
    )

    if hasattr(event, "user_id"):
        user_id = _current_user_id()
        if user_id is not None:
            event.user_id = user_id

    db.session.add(event)
    db.session.commit()

    flash("Event added successfully.", "success")
    return redirect(url_for("calendar_bp.calendar_page"))


@calendar_bp.route("/calendar/events", methods=["GET"])
@login_required
def calendar_events_api():
    event_model = _get_event_model()

    query = event_model.query
    query = _apply_user_filter(query, event_model)

    start_arg = request.args.get("start")
    end_arg = request.args.get("end")

    start_dt = _parse_datetime(start_arg)
    end_dt = _parse_datetime(end_arg)

    if start_dt and hasattr(event_model, "end_time"):
        query = query.filter(event_model.end_time >= start_dt)

    if end_dt and hasattr(event_model, "start_time"):
        query = query.filter(event_model.start_time <= end_dt)

    if hasattr(event_model, "start_time"):
        events = query.order_by(event_model.start_time.asc()).all()
    else:
        events = query.all()

    return [_serialize_event(event) for event in events]