# promotions.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional

@dataclass(frozen=True)
class Promo:
    id: str
    title: str
    window_start: date
    window_end: date
    applies_to: str          # "warehouse", "ic", "li-ion", "classII"
    details: str
    notes: str

PROMOS: List[Promo] = [
    # Warehouse Equipment — Electric Pallet Jacks
    Promo(
        id="wh_buy5_get5",
        title="Full Electric Pallet Jacks • Buy 5 Get 5 Free",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="warehouse",
        details="Light duty CBD15J-Li3 / CBD20J-Li3.",
        notes="Choose one promo only per model; cannot combine with others."
    ),
    Promo(
        id="wh_buy5_get3",
        title="New Pallet Jack Models • Buy 5 Get 3 Free",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="warehouse",
        details="Normal duty CBD15-BLIH / CBD20-BLIH.",
        notes="Choose one promo only per model; cannot combine with others."
    ),

    # Internal Combustion Forklifts
    Promo(
        id="ic_4k_6k_psi",
        title="IC 4,000–6,000 lb PSI (LPG/Dual Fuel)",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="ic",
        details="Buy 1 forklift → get 2× CBD15J-Li3 + $5,000 extra discount per unit. Up to 90-day interest-free via US Bank.",
        notes="Choose one promo only; cannot combine."
    ),
    Promo(
        id="ic_3k_7k_kubota",
        title="IC 3,000–7,000 lb Kubota (Diesel/LPG/Dual Fuel)",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="ic",
        details="Buy 1 → get 2× CBD15J-Li3 + $3,000 extra discount per unit. Up to 90-day interest-free.",
        notes="Choose one promo only; cannot combine."
    ),
    Promo(
        id="ic_5k_7k_gct",
        title="IC 5,000–7,000 lb GCT-K25 LPG (NEW)",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="ic",
        details="Buy 1 → get 2× CBD15J-Li3 + $5,000 extra discount per unit. Up to 90-day interest-free.",
        notes="Choose one promo only; cannot combine."
    ),
    Promo(
        id="ic_8k_22k",
        title="IC 8,000–22,000 lb (Diesel/LPG/Dual Fuel)",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="ic",
        details="Buy 1 → get 2× CBD15J-Li3 + $2,000 extra discount per unit. Up to 90-day interest-free.",
        notes="Choose one promo only; cannot combine."
    ),
    Promo(
        id="ic_rt_6k_7k",
        title="Rough-Terrain 6,000–7,000 lb (2WD/4WD)",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="ic",
        details="Buy 1 → get 2× CBD15J-Li3 + $2,000 extra discount per unit. Up to 90-day interest-free.",
        notes="Choose one promo only; cannot combine."
    ),

    # Lithium-ion Battery Forklifts
    Promo(
        id="li_classI_cpd12sq",
        title="Class I CPD12SQ (Li) • $3,000 Extra Discount/Unit",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="li-ion",
        details="Buy 1 → $3,000 off per unit. Up to 90-day interest-free.",
        notes="Choose one promo only; cannot combine."
    ),
    Promo(
        id="li_classI_3k_7k",
        title="Class I 3,000–7,000 lb (Li) • + 2× CBD20J-Li3",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="li-ion",
        details="Buy 1 → get 2× CBD20J-Li3 + $3,500 extra discount per unit. Up to 90-day interest-free.",
        notes="Choose one promo only; cannot combine."
    ),
    Promo(
        id="li_classI_8k_22k",
        title="Class I 8,000–22,000 lb (Li) • + 2× CBD20J-Li3",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="li-ion",
        details="Buy 1 → get 2× CBD20J-Li3 + $2,000 extra discount per unit. Up to 90-day interest-free.",
        notes="Choose one promo only; cannot combine."
    ),
    Promo(
        id="li_classII_3k_4k",
        title="Class II 3,000–4,000 lb (Li, excl. OPSM) • + 2× CBD20J-Li3",
        window_start=date(2025, 9, 1),
        window_end=date(2025, 9, 15),
        applies_to="classII",
        details="Buy 1 → get 2× CBD20J-Li3 + $4,000 extra discount per unit. Up to 90-day interest-free.",
        notes="Choose one promo only; cannot combine."
    ),
]

def active_promos(today: Optional[date] = None) -> List[Promo]:
    d = today or date.today()
    return [p for p in PROMOS if p.window_start <= d <= p.window_end]

def promos_for_context(model_code: str, cls: str, power: str) -> List[Promo]:
    """
    model_code: e.g., CPD18SQ-..., CBD20J-Li3, etc.
    cls:       "I", "II", "III", "RT", etc. (from your models.json if available)
    power:     "Lithium", "Diesel", "LPG", "Dual Fuel", etc.
    """
    m = model_code.upper()
    candidates = active_promos()
    out: List[Promo] = []

    # Pallet jacks
    if m.startswith("CBD"):
        out += [p for p in candidates if p.applies_to == "warehouse"]

    # Li-ion counterbalance (Class I) by code CPD… or power flag
    if m.startswith("CPD") or power.lower().startswith("lith"):
        out += [p for p in candidates if p.applies_to in ("li-ion",)]

    # Class II electrics
    if cls == "II":
        out += [p for p in candidates if p.applies_to in ("classII",)]

    # IC families (Diesel/LPG/Dual Fuel)
    if power.lower() in ("diesel", "lpg", "dual fuel"):
        out += [p for p in candidates if p.applies_to == "ic"]

    # Deduplicate by id
    seen = set()
    unique: List[Promo] = []
    for p in out:
        if p.id not in seen:
            unique.append(p); seen.add(p.id)
    return unique

def render_promo_lines(promos: Iterable[Promo]) -> List[str]:
    lines = []
    for p in promos:
        lines.append(f"• {p.title} — {p.details} (Valid Sep 1–15, 2025)")
    if lines:
        lines.append("Notes: choose only one promo; cannot combine; up to 90-day interest-free via US Bank where indicated. HELI America reserves final interpretation.")
    return lines
