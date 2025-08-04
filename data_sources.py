# data_sources.py
"""
Light-weight loaders for:
  • customer_report.csv      (wide “after-market revenue” data)
  • customer_billing.csv     (invoice-level billing data)

They expose:
  ▸ load_customer_report()          → list[dict]
  ▸ load_customer_billing()         → list[dict]
  ▸ make_inquiry_targets()          → list[{id,label}]  # for dropdown
  ▸ find_inquiry_rows(id)           → {"report":[…], "billing":[…]}
"""

from __future__ import annotations
import os, pandas as pd
from functools import lru_cache
from typing import List, Dict, Optional

# ── file paths ────────────────────────────────────────────────────────────
CUSTOMER_REPORT_CSV  = os.getenv("CUSTOMER_REPORT_CSV",  "customer_report.csv")
CUSTOMER_BILLING_CSV = os.getenv("CUSTOMER_BILLING_CSV", "customer_billing.csv")

# ── helpers ───────────────────────────────────────────────────────────────
def _read_csv(path: str) -> pd.DataFrame:
    """Return an empty DataFrame (not an error) if the file is missing."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")

def _pick_id(row: Dict, fallback_index: int) -> str:
    """Choose the best column to act as a unique customer ID."""
    for key in ("Ship to ID", "Sold to ID", "Ship to ID ", "Sold To BP", "Ship to code"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return str(fallback_index)

def _pick_name(row: Dict) -> str:
    """Choose a friendly name for the dropdown label."""
    for key in ("Ship to Name", "Sold to Name", "CUSTOMER", "Customer", "Ship to code"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    # fall back to any non-empty value
    for v in row.values():
        if str(v).strip():
            return str(v).strip()
    return "Unnamed"

# ── cached loaders ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_customer_report() -> List[Dict]:
    return _read_csv(CUSTOMER_REPORT_CSV).to_dict(orient="records")

@lru_cache(maxsize=1)
def load_customer_billing() -> List[Dict]:
    return _read_csv(CUSTOMER_BILLING_CSV).to_dict(orient="records")

# ── public helpers ────────────────────────────────────────────────────────
def make_inquiry_targets() -> List[Dict]:
    """
    Build the dropdown list for *Customer Inquiry* mode.
    De-duplicates customers found in either CSV by the chosen ID.
    """
    targets: Dict[str, Dict[str, str]] = {}
    combined = load_customer_report() + load_customer_billing()

    for idx, row in enumerate(combined):
        cid   = _pick_id(row, idx)
        label = _pick_name(row)
        if cid not in targets or targets[cid]["label"] == "Unnamed" and label != "Unnamed":
            targets[cid] = {"id": cid, "label": label}

    # stable order, nicely sorted by label
    return sorted(targets.values(), key=lambda x: x["label"].lower())

def find_inquiry_rows(customer_id: str) -> Dict[str, List[Dict]]:
    """
    Return all rows from both CSVs that match the given customer ID
    (using the same _pick_id logic).
    """
    report_hits  = []
    billing_hits = []

    for idx, row in enumerate(load_customer_report()):
        if _pick_id(row, idx) == customer_id:
            report_hits.append(row)

    for idx, row in enumerate(load_customer_billing()):
        if _pick_id(row, idx) == customer_id:
            billing_hits.append(row)

    return {"report": report_hits, "billing": billing_hits}
