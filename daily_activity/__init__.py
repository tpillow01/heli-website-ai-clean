from flask import Blueprint

daily_activity_bp = Blueprint(
    "daily_activity",
    __name__,
    url_prefix="/daily-activity",
    template_folder="templates",
    static_folder="static",
)

from . import routes