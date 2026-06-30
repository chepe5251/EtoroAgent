"""
Read-only diagnostic for the eToro candle API (DEMO mode only).

Detects: real field names in the JSON response, history depth, which interval
values the API accepts, and candles that look like unadjusted splits/dividends.

Usage:
    python -m src.backtest.verify_candles [--symbols SYM,...] [--apply]

This script makes REAL API calls when you run it — you supply your credentials
via the usual env vars (ETORO_PUBLIC_API_KEY, ETORO_USER_KEY).
ETORO_MODE is forced to "demo" here regardless of the env setting.

The author of this script has NOT executed it against the real API and has NOT
verified any field names, interval strings, or depth limits.  Everything in the
"ACCIÓN REQUERIDA" section is derived from whatever the API returns when YOU run
this tool; it is not pre-confirmed knowledge.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Force DEMO before EtoroClient reads the env var ──────────────────────────
if os.environ.get("ETORO_MODE", "").lower() == "real":
    print(
        "WARNING: ETORO_MODE was 'real' — forced to 'demo' for this diagnostic. "
        "No orders will be placed."
    )
os.environ["ETORO_MODE"] = "demo"

from src.core.etoro_client import EtoroClient
from src.backtest.data import _ETORO_FIELD_MAP, _normalise_candle, _pick

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATA_PY = Path(__file__).parent / "data.py"
DEFAULT_SYMBOLS = ["AAPL", "SAP.DE", "BTC"]
INTERVAL_CANDIDATES = ["OneDay", "OneWeek", "OneHour", "FifteenMinutes"]
SPLIT_THRESHOLD = 0.25
_MIN_BARS = 250


# ────────────────────────────────────────────────────────────────────────────────
# Pure analysis functions — no network, fully unit-testable
# ────────────────────────────────────────────────────────────────────────────────

def analyze_fields(candle: dict) -> dict:
    """
    Compare a raw candle dict against _ETORO_FIELD_MAP candidates.

    Returns:
        matched   – {logical_field: actual_key_found_in_candle}
        missing   – logical fields with no matching key in the candle
        unmatched – candle keys that are not present in any candidate list
    """
    all_candidates: set[str] = {
        k for candidates in _ETORO_FIELD_MAP.values() for k in candidates
    }
    matched: dict[str, str] = {}
    for field, candidates in _ETORO_FIELD_MAP.items():
        for k in candidates:
            if k in candle:
                matched[field] = k
                break
    return {
        "matched":   matched,
        "missing":   [f for f in _ETORO_FIELD_MAP if f not in matched],
        "unmatched": [k for k in candle if k not in all_candidates],
    }


def _try_parse_date(s: str) -> datetime | None:
    """Parse an ISO-style date string (with or without time/timezone)."""
    clean = s.replace("Z", "").split("+")[0].split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            pass
    return None


def compute_depth(raw_candles: list[dict]) -> dict:
    """
    Compute history coverage from a raw candle list.

    Extracts date strings using _ETORO_FIELD_MAP["date"] candidates (so the
    function works even when the real field name is "time", "timestamp", etc.).

    Returns:
        count          – number of candles
        first_date     – earliest date string (lexicographic min), or None
        last_date      – latest date string (lexicographic max), or None
        years_coverage – (last - first).days / 365, or None if unparseable
        too_few        – True when count < _MIN_BARS
    """
    count = len(raw_candles)
    if count == 0:
        return {
            "count": 0, "first_date": None, "last_date": None,
            "years_coverage": None, "too_few": True,
        }

    dates: list[str] = []
    for c in raw_candles:
        d = _pick(c, _ETORO_FIELD_MAP["date"])
        if d is not None:
            dates.append(str(d))

    first_date = min(dates) if dates else None
    last_date  = max(dates) if dates else None

    years_coverage: float | None = None
    if first_date and last_date:
        d0 = _try_parse_date(first_date)
        d1 = _try_parse_date(last_date)
        if d0 and d1:
            years_coverage = (d1 - d0).days / 365.0

    return {
        "count":          count,
        "first_date":     first_date,
        "last_date":      last_date,
        "years_coverage": years_coverage,
        "too_few":        count < _MIN_BARS,
    }


def detect_splits(
    raw_candles: list[dict], threshold: float = SPLIT_THRESHOLD
) -> list[dict]:
    """
    Find consecutive bars where |close[i] / close[i-1] - 1| > threshold.

    Uses _normalise_candle() so detection works regardless of which field-name
    variant the API actually returns.

    Returns a list of {date, prev_close, curr_close, pct_change}.
    """
    jumps: list[dict] = []
    prev: dict | None = None
    for raw in raw_candles:
        norm = _normalise_candle(raw)
        if norm is None:
            prev = None
            continue
        if prev is not None and prev["close"] > 0:
            pct = abs(norm["close"] / prev["close"] - 1)
            if pct > threshold:
                jumps.append({
                    "date":       norm["date"],
                    "prev_close": prev["close"],
                    "curr_close": norm["close"],
                    "pct_change": pct,
                })
        prev = norm
    return jumps


def suggest_field_map(
    matched: dict[str, str],
    unmatched_keys: list[str],
    sample_candle: dict,
) -> tuple[dict[str, list[str]], bool]:
    """
    Build a suggested _ETORO_FIELD_MAP with detected keys promoted to front.

    is_unambiguous is True only when:
    - every logical field has a confirmed match, AND
    - no unmatched candle key has a numeric value (which would imply a possible
      OHLC field we haven't classified yet).

    When is_unambiguous is False, --apply will NOT rewrite data.py.
    """
    suggested: dict[str, list[str]] = {}
    for field, candidates in _ETORO_FIELD_MAP.items():
        if field in matched:
            actual = matched[field]
            suggested[field] = [actual] + [c for c in candidates if c != actual]
        else:
            suggested[field] = list(candidates)

    numeric_unknowns = [
        k for k in unmatched_keys
        if isinstance(sample_candle.get(k), (int, float))
    ]
    is_unambiguous = (set(matched) == set(_ETORO_FIELD_MAP)) and not numeric_unknowns
    return suggested, is_unambiguous


# ────────────────────────────────────────────────────────────────────────────────
# Async I/O
# ────────────────────────────────────────────────────────────────────────────────

async def _probe_intervals(
    client: EtoroClient, symbol: str
) -> dict[str, tuple[bool, Any]]:
    """
    Try each INTERVAL_CANDIDATES value with count=5.
    Catches per-interval exceptions so one failure doesn't abort the whole probe.

    Returns {interval: (ok, count_or_error_string)}.
    """
    results: dict[str, tuple[bool, Any]] = {}
    for iv in INTERVAL_CANDIDATES:
        try:
            candles = await client.get_candles(symbol, interval=iv, count=5)
            results[iv] = (True, len(candles))
        except Exception as exc:
            results[iv] = (False, str(exc)[:120])
    return results


# ────────────────────────────────────────────────────────────────────────────────
# Output formatting
# ────────────────────────────────────────────────────────────────────────────────

def _rule(n: int = 60, c: str = "─") -> None:
    print(c * n)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _report_fields(
    symbol: str, raw_candles: list[dict]
) -> tuple[dict, list[str], dict]:
    """Print field analysis. Returns (matched, unmatched_keys, sample_candle)."""
    if not raw_candles:
        print(f"  {symbol}: no candles returned — field analysis skipped")
        return {}, [], {}

    sample = raw_candles[0]
    last   = raw_candles[-1]

    print(f"\n[{symbol}] First candle (raw JSON):")
    print(json.dumps(sample, indent=2, default=str))
    print(f"\n[{symbol}] Last candle (raw JSON):")
    print(json.dumps(last, indent=2, default=str))

    result    = analyze_fields(sample)
    matched   = result["matched"]
    unmatched = result["unmatched"]

    print(f"\n  Field mapping (analysed from first candle):")
    for field, candidates in _ETORO_FIELD_MAP.items():
        if field in matched:
            print(f"    {field:8s} ✓  matched key: '{matched[field]}'")
        else:
            print(f"    {field:8s} ✗  NO MATCH (tried: {candidates})")

    if unmatched:
        print(f"\n  Keys in candle not in any candidate list: {unmatched}")
        print("  → Inspect the raw JSON above — these may be the real field names.")
    else:
        print("\n  All candle keys are accounted for in the candidate lists.")

    return matched, unmatched, sample


def _report_depth(
    symbol: str, asc: list[dict], desc: list[dict]
) -> None:
    a = compute_depth(asc)
    d = compute_depth(desc)

    yrs_a = f" ({a['years_coverage']:.1f}y)" if a["years_coverage"] is not None else ""
    yrs_d = f" ({d['years_coverage']:.1f}y)" if d["years_coverage"] is not None else ""

    print(f"\n  [{symbol}] History depth:")
    print(f"    direction=asc:  {a['count']:4d} bars  "
          f"[{a['first_date']} → {a['last_date']}]{yrs_a}")
    print(f"    direction=desc: {d['count']:4d} bars  "
          f"[{d['first_date']} → {d['last_date']}]{yrs_d}")

    if a["too_few"]:
        print(f"  ⚠ WARNING: direction=asc returned <{_MIN_BARS} bars — "
              "possible free-tier depth cap")

    # Did the two directions extend overall coverage?
    a_first, a_last = a["first_date"], a["last_date"]
    d_first, d_last = d["first_date"], d["last_date"]
    if a_first and d_first and a_last and d_last:
        overall_first = min(a_first, d_first)
        overall_last  = max(a_last,  d_last)
        if overall_first < a_first or overall_last > a_last:
            print(f"  → asc+desc EXTENDS range to [{overall_first} → {overall_last}]")
            print("    fetch_symbol() two-page trick WORKS for this symbol ✓")
        else:
            print("  → asc+desc returns the SAME range — two-page trick does NOT extend coverage")


def _report_intervals(symbol: str, results: dict[str, tuple[bool, Any]]) -> None:
    print(f"\n  [{symbol}] Interval probe:")
    for iv, (ok, info) in results.items():
        if ok:
            status = f"✓ OK  ({info} bars)" if info > 0 else "⚠ OK but 0 bars returned"
        else:
            status = f"✗ FAIL: {info}"
        print(f"    {iv:20s} → {status}")


def _report_splits(symbol: str, jumps: list[dict]) -> None:
    if not jumps:
        print(f"  [{symbol}] No jump >{SPLIT_THRESHOLD * 100:.0f}% between consecutive bars ✓")
        return
    print(f"  [{symbol}] {len(jumps)} potential split / corporate action(s):")
    for j in jumps:
        print(f"    {j['date']}: {j['prev_close']:.4f} → {j['curr_close']:.4f} "
              f"({j['pct_change'] * 100:+.1f}%)")
    print("  NOTE: these may be real price moves. Verify against the eToro chart "
          "and a split calendar before concluding prices are unadjusted.")


# ────────────────────────────────────────────────────────────────────────────────
# Per-symbol orchestration
# ────────────────────────────────────────────────────────────────────────────────

async def _diagnose_symbol(client: EtoroClient, symbol: str) -> dict:
    """Run full diagnostic for one symbol. Returns a structured result dict."""
    print(f"\n{'═' * 60}\n  SYMBOL: {symbol}\n{'═' * 60}")

    _section("1 · FIELD NAMES")
    asc_candles = await client.get_candles(symbol, interval="D1", count=1000, direction="asc")
    matched, unmatched_keys, sample = _report_fields(symbol, asc_candles)

    _section("2 · HISTORY DEPTH")
    desc_candles = await client.get_candles(symbol, interval="D1", count=1000, direction="desc")
    _report_depth(symbol, asc_candles, desc_candles)

    _section("3 · INTERVAL SUPPORT")
    interval_results = await _probe_intervals(client, symbol)
    _report_intervals(symbol, interval_results)

    _section("4 · SPLIT / ADJUSTMENT HEURISTIC")
    jumps = detect_splits(asc_candles)
    _report_splits(symbol, jumps)

    suggested_map, is_unambiguous = suggest_field_map(matched, unmatched_keys, sample)
    return {
        "symbol":           symbol,
        "asc_candles":      asc_candles,
        "desc_candles":     desc_candles,
        "matched":          matched,
        "unmatched_keys":   unmatched_keys,
        "sample":           sample,
        "interval_results": interval_results,
        "jumps":            jumps,
        "suggested_map":    suggested_map,
        "is_unambiguous":   is_unambiguous,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Summary + optional --apply
# ────────────────────────────────────────────────────────────────────────────────

def _print_action_required(results: list[dict], apply: bool) -> None:
    print(f"\n\n{'═' * 60}\n  === ACCIÓN REQUERIDA ===\n{'═' * 60}")

    ref_map: dict[str, list[str]] | None = (
        results[0]["suggested_map"] if results else None
    )
    all_unambiguous = all(r["is_unambiguous"] for r in results)
    all_matched     = all(set(r["matched"]) == set(_ETORO_FIELD_MAP) for r in results)

    # ── [1] Field map ──
    print("\n[1] _ETORO_FIELD_MAP")
    if ref_map is None:
        print("  Sin datos.")
    else:
        needs_update = any(
            ref_map.get(f, [None])[0] != _ETORO_FIELD_MAP[f][0]
            for f in _ETORO_FIELD_MAP
        )
        if not all_matched:
            print("  PROBLEMA: uno o más campos no matchearon. Revisá el JSON crudo arriba.")
        elif needs_update:
            print("  Los primeros candidatos NO son los que realmente matchearon.")
            print("  Corrección sugerida (pegá esto en data.py):\n")
        else:
            print("  Los candidatos actuales ya matchean como primera opción ✓")

        print()
        print("  _ETORO_FIELD_MAP = {")
        for field, candidates in ref_map.items():
            print(f"      {field!r:10s}: {candidates!r},")
        print("  }")

    # ── [2] Intervals ──
    print("\n[2] _INTERVAL_MAP — intervalos confirmados por símbolo")
    for r in results:
        print(f"  {r['symbol']}:")
        for iv, (ok, info) in r["interval_results"].items():
            if ok and isinstance(info, int) and info > 0:
                status = "✓ FUNCIONA"
            elif ok:
                status = "⚠ FUNCIONA pero 0 barras devueltas"
            else:
                status = "✗ ERROR"
            print(f"    {iv:20s} → {status}")

    # ── [3] Depth ──
    print("\n[3] Profundidad de historia disponible")
    for r in results:
        d = compute_depth(r["asc_candles"])
        yrs = f", {d['years_coverage']:.1f}y" if d["years_coverage"] is not None else ""
        flag = "  ⚠ INSUFICIENTE para backtest multi-año" if d["too_few"] else "  ✓"
        print(f"  {r['symbol']:10s}: {d['count']} barras{yrs}{flag}")

    # ── [4] Splits ──
    print("\n[4] Posibles splits / precios sin ajustar")
    any_jumps = False
    for r in results:
        if r["jumps"]:
            any_jumps = True
            print(f"  {r['symbol']}: {len(r['jumps'])} salto(s) — contrastar con split calendar:")
            for j in r["jumps"]:
                print(f"    {j['date']}: {j['prev_close']:.4f} → {j['curr_close']:.4f}")
    if not any_jumps:
        print("  Sin saltos >25% en ningún símbolo ✓")

    # ── [5] Apply ──
    if apply:
        print("\n[5] --apply")
        if ref_map and all_unambiguous and all_matched:
            _apply_field_map(ref_map)
        else:
            print("  NO SE APLICÓ: hay ambigüedad o campos sin match.")
            print("  Corregí _ETORO_FIELD_MAP manualmente con la sugerencia de [1].")


def _apply_field_map(suggested: dict[str, list[str]]) -> None:
    """Rewrite _ETORO_FIELD_MAP in data.py with detected keys promoted to front."""
    text = _DATA_PY.read_text()

    # Match the entire _ETORO_FIELD_MAP = {...} block (values are lists, not nested dicts,
    # so [^}]* correctly stops at the outer closing brace)
    pattern = r"(_ETORO_FIELD_MAP\s*=\s*\{[^}]*\})"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        print("  ERROR: no se encontró _ETORO_FIELD_MAP en data.py — nada modificado.")
        return

    old_block = m.group(1)
    new_lines = ["_ETORO_FIELD_MAP = {"]
    for field, candidates in suggested.items():
        new_lines.append(f"    {field!r:10s}: {candidates!r},")
    new_lines.append("}")
    new_block = "\n".join(new_lines)

    if old_block == new_block:
        print("  Sin cambios — _ETORO_FIELD_MAP ya está actualizado ✓")
        return

    diff = list(difflib.unified_diff(
        old_block.splitlines(keepends=True),
        new_block.splitlines(keepends=True),
        fromfile="data.py (antes)",
        tofile="data.py (después)",
    ))
    print("  Diff a aplicar:\n")
    print("".join(diff))
    _DATA_PY.write_text(text[: m.start()] + new_block + text[m.end() :])
    print("  ✓ _ETORO_FIELD_MAP reescrito en data.py")


# ────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────

async def main(symbols: list[str], apply: bool = False) -> None:
    # Belt-and-suspenders: guard against any race with env var
    assert os.environ.get("ETORO_MODE") == "demo", "BUG: not in demo mode"

    async with EtoroClient() as client:
        assert client.mode == "demo", "BUG: EtoroClient.mode is not 'demo'"
        results: list[dict] = []
        for symbol in symbols:
            try:
                result = await _diagnose_symbol(client, symbol)
                results.append(result)
            except Exception as exc:
                print(f"\n  ERROR diagnosing {symbol}: {exc}")

    if results:
        _print_action_required(results, apply)


def _cli() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Diagnose eToro candle API — field names, depth, intervals, splits. "
            "Read-only. DEMO mode only. Requires ETORO_PUBLIC_API_KEY + ETORO_USER_KEY."
        )
    )
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated list of symbols (default: {','.join(DEFAULT_SYMBOLS)})",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Rewrite _ETORO_FIELD_MAP in data.py if detected field names are "
            "unambiguously identified. Safe: only applies when there is no ambiguity."
        ),
    )
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    asyncio.run(main(symbols, args.apply))


if __name__ == "__main__":
    _cli()
