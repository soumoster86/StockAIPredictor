# =============================
# ui/sectors.py — lightweight sector tags for screener filters
# =============================
"""Heuristic sector from company name (no network).

Used for "Only my sector" on the Screener. Not a formal industry taxonomy —
names without a match land in Other.
"""
from __future__ import annotations

import re

# Ordered: first match wins (more specific rules first)
_SECTOR_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Banking", (
        r"\bbank\b", r"\bbanking\b", r"\bhfcl\b", r"\bhousing finance\b",
        r"\bnbfc\b", r"\bfinance limited\b", r"\bfinancial\b", r"\bfinserv\b",
        r"\bcapital\b", r"\bcredit\b",
    )),
    ("IT", (
        r"\binfotech\b", r"\bsoftware\b", r"\btechnologies\b", r"\btechnology\b",
        r"\bsystems\b", r"\bdigital\b", r"\bcyber\b", r"\btech\b",
        r"\binfosys\b", r"\bwipro\b", r"\bhcl\b", r"\btcs\b",
    )),
    ("Pharma", (
        r"\bpharma\b", r"\bpharmaceutical\b", r"\blaborator\b", r"\bdrug\b",
        r"\bhealth\b", r"\bhospital\b", r"\bbiotech\b", r"\blifescience\b",
        r"\bhealthcare\b", r"\bmed\b",
    )),
    ("Auto", (
        r"\bauto\b", r"\bmotor\b", r"\bvehicle\b", r"\btyre\b", r"\btire\b",
        r"\bautomotive\b", r"\bengine\b",
    )),
    ("Metal", (
        r"\bsteel\b", r"\bmetal\b", r"\bcopper\b", r"\balumin\b", r"\biron\b",
        r"\bmining\b", r"\bore\b", r"\bzinc\b",
    )),
    ("Energy", (
        r"\boil\b", r"\bgas\b", r"\bpetroleum\b", r"\bpower\b", r"\benergy\b",
        r"\brefiner\b", r"\belectric\b", r"\brenewable\b", r"\bsolar\b",
        r"\bcoal\b", r"\blng\b",
    )),
    ("FMCG", (
        r"\bfmcg\b", r"\bfood\b", r"\bbeverage\b", r"\bconsumer\b",
        r"\bdairy\b", r"\bpackaged\b", r"\bpersonal care\b",
    )),
    ("Realty", (
        r"\brealty\b", r"\breal estate\b", r"\bproperty\b", r"\bhousing\b",
        r"\binfrastructure\b", r"\bconstruction\b", r"\bdevelopers?\b",
    )),
    ("Telecom", (
        r"\btelecom\b", r"\bcommunication\b", r"\bmobile\b", r"\bbroadband\b",
    )),
    ("Media", (
        r"\bmedia\b", r"\bentertainment\b", r"\bbroadcast\b", r"\bcinema\b",
        r"\bfilm\b", r"\btv\b",
    )),
    ("Chemical", (
        r"\bchemical\b", r"\bfertiliz\b", r"\bpaint\b", r"\bpolymer\b",
        r"\bpetrochemical\b",
    )),
    ("Textile", (
        r"\btextile\b", r"\bfabric\b", r"\byarn\b", r"\bgarment\b", r"\bcotton\b",
    )),
    ("Insurance", (
        r"\binsurance\b", r"\blife insurance\b", r"\bgeneral insurance\b",
    )),
    ("Logistics", (
        r"\blogistic\b", r"\bshipping\b", r"\bport\b", r"\btransport\b",
        r"\bcourier\b", r"\bwarehouse\b",
    )),
]

_COMPILED = [
    (sector, [re.compile(p, re.I) for p in pats])
    for sector, pats in _SECTOR_RULES
]


def classify_sector(name: str | None) -> str:
    """Map a company display name to a coarse sector bucket."""
    text = str(name or "").strip()
    if not text:
        return "Other"
    for sector, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(text):
                return sector
    return "Other"


def enrich_with_sector(df):
    """Add Sector column from Name if missing. Returns a copy."""
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    if "Sector" not in out.columns:
        name_col = "Name" if "Name" in out.columns else None
        if name_col:
            out["Sector"] = out[name_col].map(classify_sector)
        else:
            out["Sector"] = "Other"
    else:
        # Fill blanks
        out["Sector"] = out["Sector"].fillna("Other").replace("", "Other")
        if "Name" in out.columns:
            mask = out["Sector"].astype(str).str.strip().isin(("", "Other", "nan"))
            if mask.any():
                out.loc[mask, "Sector"] = out.loc[mask, "Name"].map(classify_sector)
    return out
