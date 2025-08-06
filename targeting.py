# targeting.py
import os
import io
import math
import pandas as pd
from flask import Blueprint, render_template, request, send_file

# ── Safe imports ──
FOLIUM_AVAILABLE = True
try:
    import folium
    from folium.plugins import MarkerCluster
except ImportError:
    FOLIUM_AVAILABLE = False

USE_ZIP_GEO = True
try:
    import pgeocode
except ImportError:
    USE_ZIP_GEO = False

# Blueprint
targeting_bp = Blueprint("targeting", __name__, template_folder="templates")

# ─── Config: file names ─────────────────────────────────────────────────
CUSTOMER_CSV = os.environ.get("TARGETING_CUSTOMER_CSV", "customer_report")
BILLING_CSV  = os.environ.get("TARGETING_BILLING_CSV",  "customer_billing")

# ─── Robust readers ──────────────────────────────────────────────────────
def _resolve_path(base: str) -> str:
    for p in (base, f"{base}.csv", f"{base}.xlsx"):
        if os.path.exists(p):
            return p
    return base

def _read_loose(path: str) -> pd.DataFrame:
    path = _resolve_path(path)
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except Exception:
        pass
    try:
        return pd.read_csv(path, dtype=str, encoding="cp1252", on_bad_lines="skip")
    except Exception:
        pass
    return pd.read_excel(path, dtype=str)

# ─── Column map ──────────────────────────────────────────────────────────
C = {
    "ship_to_id":   "Ship to ID",
    "sold_to_id":   "Sold to ID",
    "rep":          "Sales Rep Name",
    "ship_to_name": "Ship to Name",
    "sold_to_name": "Sold to Name",
    "address":      "Address",
    "city":         "City",
    "zip":          "Zip Code",
    "county_state": "County State",
    "r12":          "Revenue Rolling 12 Months - Aftermarket",
    "r13_24":       "Revenue Rolling 13 - 24 Months - Aftermarket",
    "r25_36":       "Revenue Rolling 25 - 36 Months - Aftermarket",
    "new36":        "New Equip R36 Revenue",
    "used36":       "Used Equip R36 Revenue",
    "parts12":      "Parts Revenue R12",
    "service12":    "Service Revenue R12 (Includes GM)",
    "ps12":         "Parts & Service Revenue R12",
    "rental12":     "Rental Revenue R12",
}

# ─── Currency parser ─────────────────────────────────────────────────────
def _to_number(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("+", "").strip()
    if not s or s.upper() == "N/A":
        return None
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None

# ─── Load customer report ─────────────────────────────────────────────────
def _load_customer():
    path = _resolve_path(CUSTOMER_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {CUSTOMER_CSV} (csv/xlsx)")
    df = _read_loose(path)

    # ensure all expected columns exist
    for col in C.values():
        if col not in df.columns:
            df[col] = pd.NA

    # guarantee geo columns
    df["Latitude"] = pd.NA
    df["Longitude"] = pd.NA

    # numeric conversions
    for key in ["r12","r13_24","r25_36","new36","used36","parts12","service12","ps12","rental12"]:
        df[C[key]] = df[C[key]].apply(_to_number)

    # derive state code
    df["State"] = df[C["county_state"]].astype(str).str[-2:].str.upper()

    # momentum %
    eps = 1e-9
    r12 = df[C["r12"]].fillna(0)
    r13 = df[C["r13_24"]].fillna(0)
    df["Momentum %"] = (r12 - r13) / (r13.abs() + eps)

    # 3-year peak & recapture
    df["3Yr Peak ($)"] = df[[C["r12"],C["r13_24"],C["r25_36"]]].fillna(0).max(axis=1)
    df["Re-capture Potential ($)"] = (df["3Yr Peak ($)"] - r12).clip(lower=0)

    # breadth
    cats = [C["parts12"],C["service12"],C["rental12"],C["new36"],C["used36"]]
    df["Breadth"] = df[cats].gt(0).sum(axis=1)

    # flags
    df["Reactivated"] = (r12 > 0) & (r13 == 0)
    df["At Risk"]     = (r12 == 0) & (r13 > 0)

    # coerce join keys
    df[C["sold_to_name"]] = df[C["sold_to_name"]].astype(str)
    df[C["ship_to_name"]] = df[C["ship_to_name"]].astype(str)

    # optional ZIP→lat/lon
    if USE_ZIP_GEO:
        try:
            df["ZIP5"] = df[C["zip"]].astype(str).str.extract(r"(\d{5})")[0]
            nomi = pgeocode.Nominatim("us")
            zips = df["ZIP5"].dropna().unique()
            if len(zips):
                geo = nomi.query_postal_code(zips)
                geo = geo[["postal_code","latitude","longitude"]].dropna()
                geo.columns = ["ZIP5","Latitude","Longitude"]
                df = df.merge(geo, on="ZIP5", how="left")
        except Exception:
            pass

    return df

# ─── Load billing data ────────────────────────────────────────────────────
def _load_billing():
    path = _resolve_path(BILLING_CSV)
    if not os.path.exists(path):
        return None
    b = _read_loose(path)

    for col in ["Date","Type","REVENUE","CUSTOMER"]:
        if col not in b.columns:
            b[col] = pd.NA

    b["CUSTOMER"] = b["CUSTOMER"].astype(str)
    b["Date"]     = pd.to_datetime(b["Date"], errors="coerce")
    b["REVENUE"]  = b["REVENUE"].apply(_to_number)

    latest = b.groupby("CUSTOMER", dropna=False)["Date"].max().reset_index()
    latest.columns = ["CUSTOMER","Last Invoice Date"]

    today = pd.Timestamp.now().normalize()
    b["Days Ago"] = (today - b["Date"]).dt.days

    def total_in(days):
        return b[b["Days Ago"] <= days].groupby("CUSTOMER")["REVENUE"].sum()

    out = (
        latest
        .merge(total_in(90).rename("Rev 90d"),   on="CUSTOMER", how="left")
        .merge(total_in(180).rename("Rev 180d"), on="CUSTOMER", how="left")
        .merge(total_in(365).rename("Rev 365d"), on="CUSTOMER", how="left")
    )
    out["Days Since Last Invoice"] = (today - out["Last Invoice Date"]).dt.days
    return out

# ─── Segmentation & tactics ──────────────────────────────────────────────
def _segment_row(r, momentum_thresh=0.20, recapture_thresh=100000):
    r12       = r.get(C["r12"]) or 0.0
    breadth   = r.get("Breadth") or 0
    momentum  = r.get("Momentum %") or 0.0
    recapture = r.get("Re-capture Potential ($)") or 0.0
    at_risk   = bool(r.get("At Risk"))

    if at_risk or recapture >= recapture_thresh:
        return "ATTACK"
    if (r12 > 0 and breadth <= 1) or momentum >= momentum_thresh:
        return "GROW"
    if r12 > 0:
        return "MAINTAIN"
    return "TEST/EXPAND"

def _tactic(seg):
    return {
        "ATTACK":     "Win-back/displace: multi-thread, demo, sharp pricing, service rescue.",
        "GROW":       "Upsell/cross-sell: add PM contract, attachments, lithium conversion.",
        "MAINTAIN":   "Defend & delight: QBR cadence, SLA adherence, renewal focus.",
        "TEST/EXPAND":"Low-friction pilot: starter order, trial service, quick win."
    }.get(seg, "")

# ─── Map HTML generator ──────────────────────────────────────────────────
def _make_map(df, segment_colors):
    # guard missing geo cols
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        df["Latitude"] = pd.NA
        df["Longitude"] = pd.NA

    if not FOLIUM_AVAILABLE:
        return "<div>Map unavailable; install folium.</div>"

    if df["Latitude"].notna().any() and df["Longitude"].notna().any():
        lat = df["Latitude"].dropna().astype(float).mean()
        lon = df["Longitude"].dropna().astype(float).mean()
        m = folium.Map(location=[lat, lon], tiles="cartodbpositron", zoom_start=5)
    else:
        m = folium.Map(location=[39.8283, -98.5795], tiles="cartodbpositron", zoom_start=4)

    cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        lt, ln = row["Latitude"], row["Longitude"]
        if pd.isna(lt) or pd.isna(ln):
            continue
        seg    = row["Segment"]
        color  = segment_colors.get(seg, "gray")
        radius = 6
        try:
            rec = float(row.get("Re-capture Potential ($)") or 0)
            if rec > 0:
                radius = min(22, 6 + math.log10(rec + 10) * 6)
        except:
            pass
        tooltip = folium.Tooltip(
            f"<b>{row[C['sold_to_name']]}</b><br>"
            f"{row[C['city']]}, {row['State']}<br>"
            f"R12: ${row[C['r12']]:,.0f}<br>"
            f"Momentum: {row['Momentum %']*100:.1f}%<br>"
            f"Re-capture: ${row['Re-capture Potential ($)']:,.0f}"
        )
        folium.CircleMarker(
            location=[lt, ln], radius=radius,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.75, opacity=0.9
        ).add_child(tooltip).add_to(cluster)

    return m._repr_html_()

# ─── Routes ───────────────────────────────────────────────────────────────
@targeting_bp.route("/targeting")
def targeting_page():
    momentum = float(request.args.get("momentum",  0.20))
    recapture = float(request.args.get("recapture",100000))

    df      = _load_customer()
    billing = _load_billing()

    if billing is not None:
        bill_idx = billing.set_index("CUSTOMER")
        bill_idx.index = bill_idx.index.astype(str)

        # join sold-to
        df = df.join(bill_idx, on=C["sold_to_name"], rsuffix="_s")
        sold_cols = ["Last Invoice Date","Rev 90d","Rev 180d","Rev 365d","Days Since Last Invoice"]
        for c in sold_cols:
            df.rename(columns={c: f"{c}_s"}, inplace=True)
        # join ship-to
        df = df.join(bill_idx, on=C["ship_to_name"], rsuffix="_h")
        for c in sold_cols:
            df.rename(columns={c: f"{c}_h"}, inplace=True)
        # fill & cleanup
        for c in sold_cols:
            df[c] = df[f"{c}_s"].fillna(df[f"{c}_h"])
        df.drop(columns=[f"{c}_s" for c in sold_cols] + [f"{c}_h" for c in sold_cols], inplace=True)

    df["Segment"]           = df.apply(lambda r: _segment_row(r, momentum, recapture), axis=1)
    df["Recommended Tactic"] = df["Segment"].apply(_tactic)

    segment_colors = {
        "ATTACK":     "#ff3333",
        "GROW":       "#ff9933",
        "TEST/EXPAND":"#33aaff",
        "MAINTAIN":   "#66cc66",
    }

    table_cols = [
        C["sold_to_name"], C["ship_to_name"], C["rep"], C["city"],
        "State", "Segment", "Recommended Tactic"
    ]
    table_rows = df[table_cols].values.tolist()

    return render_template(
        "targeting.html",
        map_html=_make_map(df, segment_colors),
        table_columns=table_cols,
        table_rows=table_rows,
        momentum=momentum,
        recapture=recapture
    )

@targeting_bp.route("/targeting/download")
def targeting_download():
    momentum = float(request.args.get("momentum",  0.20))
    recapture = float(request.args.get("recapture",100000))

    df      = _load_customer()
    billing = _load_billing()
    if billing is not None:
        bill_idx = billing.set_index("CUSTOMER")
        bill_idx.index = bill_idx.index.astype(str)
        df = df.join(bill_idx, on=C["sold_to_name"], rsuffix="_s")
        df = df.join(bill_idx, on=C["ship_to_name"], rsuffix="_h")

    df["Segment"]           = df.apply(lambda r: _segment_row(r, momentum, recapture), axis=1)
    df["Recommended Tactic"] = df["Segment"].apply(_tactic)

    raw_cols = [
        C["sold_to_name"], C["ship_to_name"], C["rep"], C["city"], "State",
        C["r12"], C["r13_24"], C["r25_36"], "Momentum %",
        "Re-capture Potential ($)", "Breadth", "Segment", "Recommended Tactic",
        "Days Since Last Invoice", "Rev 90d", "Rev 180d", "Rev 365d"
    ]

    buf = io.BytesIO()
    df[raw_cols].to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="target_accounts.csv",
        mimetype="text/csv"
    )
