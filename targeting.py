# targeting.py
import os
import io
import math
from datetime import timezone
import pandas as pd
from flask import Blueprint, render_template, request, send_file

# ── Safe imports so the server never crashes if libs are missing ──
FOLIUM_AVAILABLE = True
try:
    import folium  # type: ignore
    from folium.plugins import MarkerCluster  # type: ignore
except Exception:
    FOLIUM_AVAILABLE = False

USE_ZIP_GEO = True
try:
    import pgeocode  # type: ignore
except Exception:
    USE_ZIP_GEO = False

# Blueprint
targeting_bp = Blueprint("targeting", __name__, template_folder="templates")

# ─── Config: file paths (override via env vars if needed) ────────────────
# Your files have no extension: "customer_report" and "customer_billing"
CUSTOMER_CSV = os.environ.get("TARGETING_CUSTOMER_CSV", "customer_report")
BILLING_CSV  = os.environ.get("TARGETING_BILLING_CSV", "customer_billing")

# ─── Minimal robust readers (fixes the encoding crash) ───────────────────
def _resolve_path(base: str) -> str:
    """Try plain, .csv, .xlsx — return the first that exists."""
    for p in (base, f"{base}.csv", f"{base}.xlsx"):
        if os.path.exists(p):
            return p
    return base  # downstream will error if missing

def _read_loose(path: str) -> pd.DataFrame:
    """
    Tolerant table loader:
    1) UTF-8 CSV
    2) Windows-1252 CSV (skip bad lines)
    3) Excel (xlsx/xls)
    """
    path = _resolve_path(path)
    # Try UTF-8 CSV
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        pass
    except Exception:
        pass
    # Try CP1252 CSV
    try:
        return pd.read_csv(path, dtype=str, encoding="cp1252", on_bad_lines="skip")
    except Exception:
        pass
    # Excel last
    return pd.read_excel(path, dtype=str)

# Your report headers (from your sample)
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
    # revenue windows (aftermarket)
    "r12":          "Revenue Rolling 12 Months - Aftermarket",
    "r13_24":       "Revenue Rolling 13 - 24 Months - Aftermarket",
    "r25_36":       "Revenue Rolling 25 - 36 Months - Aftermarket",
    # categories
    "new36":        "New Equip R36 Revenue",
    "used36":       "Used Equip R36 Revenue",
    "parts12":      "Parts Revenue R12",
    "service12":    "Service Revenue R12 (Includes GM)",
    "ps12":         "Parts & Service Revenue R12",
    "rental12":     "Rental Revenue R12",
}

# ─── Utilities ───────────────────────────────────────────────────────────
def _to_number(x):
    """Convert formats like '$81,593', '($80)', ' 1,672.06 ' to float."""
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
    except Exception:
        return None

# ─── Loaders (using the tolerant reader) ─────────────────────────────────
def _load_customer():
    if not os.path.exists(_resolve_path(CUSTOMER_CSV)):
        raise FileNotFoundError(f"Missing {CUSTOMER_CSV} (.csv/.xlsx). Put it next to app or set TARGETING_CUSTOMER_CSV.")
    df = _read_loose(CUSTOMER_CSV)

    # ensure expected columns exist
    for _, col in C.items():
        if col not in df.columns:
            df[col] = pd.NA

    # numerics
    num_cols = [C["r12"], C["r13_24"], C["r25_36"], C["new36"], C["used36"],
                C["parts12"], C["service12"], C["ps12"], C["rental12"]]
    for col in num_cols:
        df[col] = df[col].apply(_to_number)

    # State (from "County State" like "Hendricks IN")
    df["State"] = df[C["county_state"]].astype(str).str[-2:].str.upper()

    # Momentum % = (R12 - R13_24) / max(|R13_24|, eps)
    eps = 1e-9
    r12 = df[C["r12"]].fillna(0)
    r13 = df[C["r13_24"]].fillna(0)
    df["Momentum %"] = (r12 - r13) / (r13.abs() + eps)

    # 3-year peak + recapture
    df["3Yr Peak ($)"] = df[[C["r12"], C["r13_24"], C["r25_36"]]].fillna(0).max(axis=1)
    df["Re-capture Potential ($)"] = (df["3Yr Peak ($)"] - r12).clip(lower=0)

    # Breadth across categories
    cats = [C["parts12"], C["service12"], C["rental12"], C["new36"], C["used36"]]
    df["Breadth"] = df[cats].gt(0).sum(axis=1)

    # Flags
    df["Reactivated"] = (r12 > 0) & (r13 == 0)
    df["At Risk"]     = (r12 == 0) & (r13 > 0)

    # ZIP→lat/lon
    df["ZIP5"] = df[C["zip"]].astype(str).str.extract(r"(\d{5})")[0]
    df["Latitude"] = pd.NA
    df["Longitude"] = pd.NA
    if USE_ZIP_GEO:
        try:
            nomi = pgeocode.Nominatim("us")
            zips = df["ZIP5"].dropna().unique().tolist()
            if zips:
                geo = nomi.query_postal_code(zips)
                geo = geo[["postal_code", "latitude", "longitude"]].dropna()
                geo.columns = ["ZIP5", "Latitude", "Longitude"]
                df = df.merge(geo, on="ZIP5", how="left")
        except Exception:
            pass

    return df

def _load_billing():
    """
    Expect columns: Date, Type, REVENUE, CUSTOMER
    Handles CSV (utf-8, cp1252) and Excel workbooks.
    """
    if not os.path.exists(_resolve_path(BILLING_CSV)):
        return None

    b = _read_loose(BILLING_CSV)

    # normalize expected columns
    for need in ["Date", "Type", "REVENUE", "CUSTOMER"]:
        if need not in b.columns:
            b[need] = pd.NA

    # types
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce", infer_datetime_format=True)
    b["REVENUE"] = b["REVENUE"].apply(_to_number)

    # recency + rolling sums
    latest = b.groupby("CUSTOMER", dropna=False)["Date"].max().reset_index()
    latest.columns = ["CUSTOMER", "Last Invoice Date"]

    today = pd.Timestamp.now(tz=timezone.utc).normalize()
    b["Days Ago"] = (today - b["Date"]).dt.days

    def total_in(days):
        return b.loc[b["Days Ago"] <= days].groupby("CUSTOMER")["REVENUE"].sum()

    out = latest.merge(total_in(90).rename("Rev 90d"), on="CUSTOMER", how="left") \
                .merge(total_in(180).rename("Rev 180d"), on="CUSTOMER", how="left") \
                .merge(total_in(365).rename("Rev 365d"), on="CUSTOMER", how="left")
    out["Days Since Last Invoice"] = (today - out["Last Invoice Date"]).dt.days
    return out

# ─── Scoring & Tactics ───────────────────────────────────────────────────
def _segment_row(r, momentum_thresh=0.20, recapture_thresh=100000):
    r12 = r.get(C["r12"]) or 0.0
    breadth = r.get("Breadth") or 0
    momentum = r.get("Momentum %") or 0.0
    recapture = r.get("Re-capture Potential ($)") or 0.0
    at_risk = bool(r.get("At Risk"))

    if at_risk or recapture >= recapture_thresh:
        return "ATTACK"
    if (r12 > 0 and breadth <= 1) or momentum >= momentum_thresh:
        return "GROW"
    if r12 > 0:
        return "MAINTAIN"
    return "TEST/EXPAND"

def _tactic(seg):
    return {
        "ATTACK": "Win-back/displace: multi-thread, demo, sharp pricing, service rescue.",
        "GROW": "Upsell/cross-sell: add PM contract, attachments, lithium conversion.",
        "MAINTAIN": "Defend & delight: QBR cadence, SLA adherence, renewal focus.",
        "TEST/EXPAND": "Low-friction pilot: starter order, trial service, quick win."
    }.get(seg, "")

# ─── Map ─────────────────────────────────────────────────────────────────
def _make_map(df, segment_colors):
    if not FOLIUM_AVAILABLE:
        return "<div style='padding:16px;background:#000;color:#ccc'>Map unavailable (install 'folium' to enable).</div>"

    if df["Latitude"].notna().any() and df["Longitude"].notna().any():
        lat = float(df["Latitude"].dropna().astype(float).mean())
        lon = float(df["Longitude"].dropna().astype(float).mean())
        zoom = 5
    else:
        lat, lon, zoom = 39.8283, -98.5795, 4  # US center

    m = folium.Map(location=[lat, lon], tiles="cartodbpositron", zoom_start=zoom)
    cluster = MarkerCluster().add_to(m)

    for _, r in df.iterrows():
        lt, ln = r.get("Latitude"), r.get("Longitude")
        if pd.isna(lt) or pd.isna(ln):
            continue
        seg = r["Segment"]
        color = segment_colors.get(seg, "gray")

        # bubble scale by recapture
        radius = 6
        try:
            rec = float(r["Re-capture Potential ($)"] or 0)
            if rec > 0:
                radius = min(22, 6 + math.log10(rec + 10) * 6)
        except Exception:
            pass

        acct = r.get(C["sold_to_name"]) or r.get(C["ship_to_name"]) or ""
        city_state = f"{r.get(C['city'])}, {r.get('State') or ''}".strip().strip(",")
        tooltip = folium.Tooltip(f"""
            <b>{acct}</b><br>
            {city_state}<br>
            R12: ${ (r.get(C['r12']) or 0):,.0f }<br>
            R13–24: ${ (r.get(C['r13_24']) or 0):,.0f }<br>
            R25–36: ${ (r.get(C['r25_36']) or 0):,.0f }<br>
            Momentum: { (r.get('Momentum %') or 0)*100:,.1f }%<br>
            Re-capture: ${ (r.get('Re-capture Potential ($)') or 0):,.0f }<br>
            Segment: {seg}
        """)
        folium.CircleMarker(
            location=[float(lt), float(ln)],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            opacity=0.9
        ).add_child(tooltip).add_to(cluster)

    return m._repr_html_()

# ─── Routes ──────────────────────────────────────────────────────────────
@targeting_bp.route("/targeting")
def targeting_page():
    momentum = float(request.args.get("momentum", 0.20))         # default 20%
    recapture = float(request.args.get("recapture", 100000))     # default $100k

    df = _load_customer()
    billing = _load_billing()

    # Merge recency on Sold-to first, fallback Ship-to
    if billing is not None:
        df = df.merge(billing, left_on=C["sold_to_name"], right_on="CUSTOMER", how="left")
        missing_mask = df["Last Invoice Date"].isna()
        if missing_mask.any():
            df2 = df[missing_mask].drop(
                columns=["CUSTOMER", "Last Invoice Date", "Rev 90d", "Rev 180d", "Rev 365d"],
                errors="ignore"
            ).merge(billing, left_on=C["ship_to_name"], right_on="CUSTOMER", how="left")
            df.loc[missing_mask, ["CUSTOMER", "Last Invoice Date", "Rev 90d", "Rev 180d", "Rev 365d", "Days Since Last Invoice"]] = \
                df2[["CUSTOMER", "Last Invoice Date", "Rev 90d", "Rev 180d", "Rev 365d", "Days Since Last Invoice"]].values

    # Segment + tactic
    df["Segment"] = df.apply(lambda r: _segment_row(r, momentum, recapture), axis=1)
    df["Recommended Tactic"] = df["Segment"].apply(_tactic)

    # Map
    segment_colors = {
        "ATTACK": "#ff3333",
        "GROW": "#ff9933",
        "TEST/EXPAND": "#33aaff",
        "MAINTAIN": "#66cc66"
    }
    map_html = _make_map(df, segment_colors)

    # Table view (pretty strings)
    df_out = df.copy()
    df_out["R12 ($)"]        = df_out[C["r12"]].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    df_out["R13–24 ($)"]     = df_out[C["r13_24"]].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    df_out["R25–36 ($)"]     = df_out[C["r25_36"]].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    df_out["Momentum %"]     = df_out["Momentum %"].map(lambda x: f"{x*100:.1f}%")
    df_out["Re-capture ($)"] = df_out["Re-capture Potential ($)"].map(lambda x: f"${x:,.0f}")
    for col in ["Rev 90d", "Rev 180d", "Rev 365d"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
        else:
            df_out[col] = ""
    df_out["Days Since Last Invoice"] = df_out.get("Days Since Last Invoice", pd.Series([None]*len(df_out))).fillna("")

    cols = [
        C["sold_to_name"], C["ship_to_name"], C["rep"], C["city"], "State",
        "R12 ($)", "R13–24 ($)", "R25–36 ($)", "Momentum %", "Re-capture ($)",
        "Breadth", "Segment", "Recommended Tactic", "Days Since Last Invoice",
        "Rev 90d", "Rev 180d", "Rev 365d"
    ]
    table = df_out[cols].copy()

    # Sort by most actionable: recapture desc, then worst momentum
    def _key(v):
        s = str(v)
        if "$" in s:
            try: return float(s.replace("$", "").replace(",", ""))
            except Exception: return 0.0
        if "%" in s:
            try: return float(s.replace("%", ""))
            except Exception: return 0.0
        try: return float(s)
        except Exception: return 0.0

    table = table.sort_values(by=["Re-capture ($)", "Momentum %"],
                              ascending=[False, True],
                              key=lambda s: s.map(_key))

    return render_template(
        "targeting.html",
        map_html=map_html,
        table_columns=table.columns.tolist(),
        table_rows=table.values.tolist(),
        momentum=momentum,
        recapture=recapture
    )

@targeting_bp.route("/targeting/download")
def targeting_download():
    momentum = float(request.args.get("momentum", 0.20))
    recapture = float(request.args.get("recapture", 100000))

    df = _load_customer()
    billing = _load_billing()
    if billing is not None:
        df = df.merge(billing, left_on=C["sold_to_name"], right_on="CUSTOMER", how="left")
        missing_mask = df["Last Invoice Date"].isna()
        if missing_mask.any():
            df2 = df[missing_mask].drop(
                columns=["CUSTOMER", "Last Invoice Date", "Rev 90d", "Rev 180d", "Rev 365d"],
                errors="ignore"
            ).merge(billing, left_on=C["ship_to_name"], right_on="CUSTOMER", how="left")
            df.loc[missing_mask, ["CUSTOMER", "Last Invoice Date", "Rev 90d", "Rev 180d", "Rev 365d", "Days Since Last Invoice"]] = \
                df2[["CUSTOMER", "Last Invoice Date", "Rev 90d", "Rev 180d", "Rev 365d", "Days Since Last Invoice"]].values

    df["Segment"] = df.apply(lambda r: _segment_row(r, momentum, recapture), axis=1)
    df["Recommended Tactic"] = df["Segment"].apply(_tactic)

    raw_cols = [
        C["sold_to_name"], C["ship_to_name"], C["rep"], C["city"], "State",
        C["r12"], C["r13_24"], C["r25_36"], "Momentum %", "Re-capture Potential ($)",
        "Breadth", "Segment", "Recommended Tactic",
        "Days Since Last Invoice", "Rev 90d", "Rev 180d", "Rev 365d"
    ]
    for col in ["Rev 90d", "Rev 180d", "Rev 365d"]:
        if col not in df.columns:
            df[col] = None

    out = df[raw_cols].sort_values(by="Re-capture Potential ($)", ascending=False)
    buf = io.BytesIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="target_accounts.csv", mimetype="text/csv")
