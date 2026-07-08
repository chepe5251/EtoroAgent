"""
Heuristic sector classification by keyword matching on the symbol/company
name string. eToro's bulk instruments endpoint does not expose a sector or
industry field (only instrumentID, displayName, instrumentTypeID,
exchangeID — confirmed by inspection), so there's no authoritative source
to join against. This is a pragmatic proxy for "these symbols tend to move
together" — good enough to cap correlated exposure, not a real GICS
classification.

Symbols in this codebase are company names transformed to
SCREAMING_SNAKE_CASE (e.g. "JPMORGAN_CHASE", "EXXON_MOBIL") — keyword
matching works directly against that string.
"""
from __future__ import annotations

# Order matters: first match wins. More specific keywords first where a
# name could plausibly match multiple sectors.
_SECTOR_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Energy", [
        "OIL", "GAS_", "_GAS", "ENERGY", "PETROLEUM", "PETRO", "DRILLING",
        "COAL", "SOLAR", "RENEWABLE", "NATURAL_RESOURCES", "PIPELINE",
        "REFINING", "EXPLORATION", "OFFSHORE",
    ]),
    ("Financials", [
        "BANK", "FINANCIAL", "INSURANCE", "CAPITAL", "INVESTMENT", "ASSET",
        "TRUST", "SECURITIES", "HOLDINGS_GROUP", "CREDIT", "MORTGAGE",
        "BROKERAGE", "WEALTH", "FUND", "EXCHANGE_GROUP", "PAYMENTS",
    ]),
    ("Healthcare", [
        "PHARMA", "HEALTH", "BIO", "MEDICAL", "THERAPEUT", "DRUG",
        "HOSPITAL", "LABORATOR", "DIAGNOSTIC", "SURGICAL", "CLINIC",
        "GENOMICS", "VACCINE", "DENTAL", "MEDTECH",
    ]),
    ("Technology", [
        "TECH", "SOFTWARE", "SEMICONDUCTOR", "MICRO", "_DATA_", "CLOUD",
        "CYBER", "DIGITAL", "COMPUTER", "ELECTRONIC", "_CHIP", "INTERNET",
        "NETWORK", "SYSTEMS", "PLATFORMS", "ROBOTICS", "CIRCUITS", "ANALYTICS",
    ]),
    ("Telecommunications", [
        "TELECOM", "COMMUNICATIONS", "WIRELESS", "MOBILE", "BROADBAND",
        "TELEPHONE", "SATELLITE",
    ]),
    ("Utilities", [
        "ELECTRIC_POWER", "ELECTRIC_", "_ELECTRIC", "POWER_", "UTILITY",
        "UTILITIES", "WATER_", "GRID", "HYDRO",
    ]),
    ("RealEstate", [
        "REIT", "REAL_ESTATE", "PROPERTIES", "PROPERTY", "REALTY", "LAND_",
        "ESTATES",
    ]),
    ("Materials", [
        "MINING", "STEEL", "METAL", "CHEMICAL", "MATERIALS", "GOLD_",
        "_GOLD", "COPPER", "ALUMINIUM", "ALUMINUM", "CEMENT", "PAPER",
        "FOREST", "ZINC", "SILVER", "PLATINUM",
    ]),
    ("Automotive", [
        "MOTOR", "AUTOMOTIVE", "_AUTO_", "TIRE", "RUBBER",
    ]),
    ("Transportation", [
        "AIRLINES", "AIRWAYS", "AVIATION", "SHIPPING", "LOGISTICS",
        "FREIGHT", "RAILWAY", "RAILROAD", "PORTS_",
    ]),
    ("Industrials", [
        "INDUSTRIAL", "MACHINERY", "AEROSPACE", "DEFENSE", "DEFENCE",
        "CONSTRUCTION", "ENGINEERING", "ELECTRIC_EQUIPMENT", "TOOLS_",
    ]),
    ("ConsumerStaples", [
        "FOODS", "BEVERAGE", "TOBACCO", "GROCERY", "BREWING", "DISTILL",
        "AGRICULTURE", "DAIRY",
    ]),
    ("ConsumerDiscretionary", [
        "RETAIL", "STORE", "RESTAURANT", "APPAREL", "FASHION", "LEISURE",
        "HOTEL", "GAMING", "ENTERTAINMENT", "TOYS", "FURNITURE",
    ]),
]


def classify_sector(symbol: str) -> str:
    """Best-effort sector for a SCREAMING_SNAKE_CASE symbol/company name.
    Falls back to "Other" when nothing matches."""
    upper = f"_{symbol.upper()}_"
    for sector, keywords in _SECTOR_KEYWORDS:
        for kw in keywords:
            if kw in upper:
                return sector
    return "Other"
