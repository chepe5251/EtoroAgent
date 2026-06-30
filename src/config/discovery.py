"""
eToro instrument discovery.

Queries the eToro API to validate which symbols from the static universe are
actually tradeable today, resolves instrument IDs, and saves to universe_cache.json.

Run: python src/config/discovery.py [--regions US,EU,ASIA,CRYPTO] [--force]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.etoro_client import EtoroClient
from src.config.universe import REGION_SYMBOLS, _CACHE_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def discover_region(
    client: EtoroClient, region: str, symbols: list[str]
) -> dict[str, str]:
    """
    For each symbol in the list, attempt to resolve its eToro instrument ID.
    Returns {symbol: instrument_id} for every symbol that resolves successfully.
    """
    resolved: dict[str, str] = {}
    logger.info("Discovering %d symbols for region %s...", len(symbols), region)

    batch_size = 10
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        tasks = {sym: client.get_instrument_id(sym) for sym in batch}
        results = await asyncio.gather(
            *[tasks[s] for s in batch], return_exceptions=True
        )
        for sym, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.debug("  %s — error: %s", sym, result)
            elif result:
                resolved[sym] = result
                logger.debug("  %s → %s", sym, result)
            else:
                logger.debug("  %s — not found", sym)
        # Respect rate limits
        if i + batch_size < len(symbols):
            await asyncio.sleep(2.0)

    logger.info(
        "Region %s: %d/%d symbols resolved", region, len(resolved), len(symbols)
    )
    return resolved


async def run_discovery(regions: list[str], force: bool = False):
    """Run discovery for the given regions and update the cache file."""
    if not force and _CACHE_FILE.exists():
        data = json.loads(_CACHE_FILE.read_text())
        saved_at = datetime.fromisoformat(data.get("saved_at", "2000-01-01T00:00:00+00:00"))
        age_days = (datetime.now(timezone.utc) - saved_at).days
        if age_days < 7:
            logger.info("Cache is %d days old (TTL=7d). Use --force to refresh.", age_days)
            return

    async with EtoroClient() as client:
        cache: dict = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "regions": {},
            "instrument_ids": {},
        }

        for region in regions:
            symbols = REGION_SYMBOLS.get(region, [])
            if not symbols:
                logger.warning("No symbols defined for region %s", region)
                continue

            resolved = await discover_region(client, region, symbols)
            cache["regions"][region] = list(resolved.keys())
            cache["instrument_ids"].update(resolved)

        _CACHE_FILE.write_text(json.dumps(cache, indent=2))
        total = sum(len(v) for v in cache["regions"].values())
        logger.info(
            "Discovery complete — %d instruments cached to %s", total, _CACHE_FILE
        )
        for region, syms in cache["regions"].items():
            logger.info("  %s: %d instruments", region, len(syms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover eToro tradeable instruments")
    parser.add_argument(
        "--regions",
        default="US,EU,ASIA,CRYPTO",
        help="Comma-separated regions (default: US,EU,ASIA,CRYPTO)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if cache is fresh",
    )
    args = parser.parse_args()
    regions = [r.strip().upper() for r in args.regions.split(",")]
    asyncio.run(run_discovery(regions, force=args.force))
