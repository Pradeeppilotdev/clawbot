from __future__ import annotations

import argparse
import json
import os

from rich import print

from .agent import ClawbotAgent
from .config import get_settings
from .eval import run_eval
from .llm import create_llm_client
from .memory import RunMemory, persist_json, persist_run
from .scorecard import build_scorecard
from .tools import build_default_registry


def _build_agent() -> ClawbotAgent:
    settings = get_settings()
    llm = create_llm_client(settings)
    tools = build_default_registry()
    return ClawbotAgent(settings=settings, llm=llm, tools=tools)


def cmd_run(task: str) -> None:
    agent = _build_agent()
    result = agent.run(task)

    print("\n[bold cyan]Answer[/bold cyan]")
    print(result.answer)
    print(f"\n[bold]Steps used:[/bold] {result.steps_used}")
    print(f"[bold]Mode:[/bold] {result.mode}")
    print(f"[bold]Latency:[/bold] {result.elapsed_ms} ms")

    memory = RunMemory(task=task, events=result.trace)
    path = persist_run(memory)
    print(f"[dim]Trace saved to {path}[/dim]")


def cmd_eval() -> None:
    agent = _build_agent()
    report = run_eval(agent)
    print("\n[bold green]Evaluation Report[/bold green]")
    print(json.dumps(report, indent=2))


def cmd_scorecard(task: str) -> None:
    agent = _build_agent()
    result = agent.run(task)
    scorecard = build_scorecard(task, result)

    print("\n[bold magenta]Scorecard[/bold magenta]")
    print(json.dumps(scorecard.to_dict(), indent=2))

    memory = RunMemory(task=task, events=result.trace)
    run_path = persist_run(memory)
    score_path = persist_json(scorecard.to_dict(), "latest_scorecard.json")
    print(f"\n[dim]Trace saved to {run_path}[/dim]")
    print(f"[dim]Scorecard saved to {score_path}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clawbot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a single task")
    run_parser.add_argument("task", type=str, help="Task prompt for the agent")
    run_parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "deep"],
        help="Override AGENT_MODE for this run.",
    )

    sub.add_parser("eval", help="Run built-in benchmark")

    score_parser = sub.add_parser("scorecard", help="Run task and generate judge-facing scorecard")
    score_parser.add_argument("task", type=str, help="Task prompt for scoring")
    score_parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "deep"],
        help="Override AGENT_MODE for this run.",
    )

    args = parser.parse_args()

    if args.command == "run":
        if args.mode:
            os.environ["AGENT_MODE"] = args.mode
        cmd_run(args.task)
    elif args.command == "eval":
        cmd_eval()
    elif args.command == "scorecard":
        if args.mode:
            os.environ["AGENT_MODE"] = args.mode
        cmd_scorecard(args.task)


if __name__ == "__main__":
    main()
