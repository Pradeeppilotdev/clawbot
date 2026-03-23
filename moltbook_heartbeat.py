#!/usr/bin/env python
"""
Moltbook Heartbeat for Clawbot
Run this periodically (cron, systemd timer, or OpenClaw heartbeat) to:
- Respond to agent-related posts
- Post original insights occasionally
- Engage with the community

Usage:
  # Run engagement loop every 30 minutes
  */30 * * * * cd /home/pradeep/Downloads/clawbot && python moltbook_heartbeat.py
  
  # Or manually
  python moltbook_heartbeat.py --limit 10 --insight
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clawbot.agent import ClawbotAgent
from clawbot.config import get_settings
from clawbot.llm import create_llm_client
from clawbot.tools import build_default_registry
from clawbot.engagement import MoltbookEngagement
from clawbot.moltbook_client import MoltbookClient


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Moltbook heartbeat engagement for Clawbot"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of posts to check (default: 10)",
    )
    parser.add_argument(
        "--insight",
        action="store_true",
        help="Post an original insight with this run",
    )
    parser.add_argument(
        "--log",
        type=str,
        help="Log file path (optional)",
    )
    args = parser.parse_args()

    # Setup logging if requested
    if args.log:
        log_file = Path(args.log)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(log_file, "a")
        sys.stderr = sys.stdout

    print(f"\n{'='*60}")
    print(f"Moltbook Heartbeat: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    # Get API key
    api_key = os.getenv("MOLTBOOK_API_KEY", "")
    if not api_key:
        print("❌ MOLTBOOK_API_KEY not set in environment")
        sys.exit(1)

    try:
        # Initialize agent and Moltbook client
        settings = get_settings()
        llm = create_llm_client(settings)
        tools = build_default_registry()
        agent = ClawbotAgent(settings=settings, llm=llm, tools=tools)
        moltbook = MoltbookClient(api_key)

        # Run engagement
        engagement = MoltbookEngagement(agent, moltbook)
        stats = engagement.engage_batch(
            limit=args.limit,
            respond=True,
            post_insight=args.insight,
        )

        print(f"\n✅ Heartbeat complete")
        print(json.dumps(stats, indent=2))

    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        if args.log and sys.stdout != sys.__stdout__:
            sys.stdout.close()


if __name__ == "__main__":
    main()
