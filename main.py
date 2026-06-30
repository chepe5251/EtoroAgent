import asyncio
import logging
import sys
from pathlib import Path

# Ensure project src is on the path when running as `python main.py`
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/etoroAgent.log", encoding="utf-8"),
    ],
)

from src.core.orchestrator import Orchestrator


async def main():
    orchestrator = Orchestrator()
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())
