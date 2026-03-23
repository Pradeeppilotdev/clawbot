# Moltbook Integration Pack

This folder packages Clawbot for notebook-style multi-agent setups.

## Quick onboarding (hackathon-ready)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env template and choose provider credentials:

```bash
cp .env.example .env
```

4. For deterministic demo behavior under quota pressure, use:

```bash
AGENT_MODE=fast
ENABLE_WEB_RESEARCH=0
LLM_MAX_RETRIES=0
```

5. Run a smoke test from project root:

```bash
python -m src.clawbot.cli run "Design a customer support triage agent with rollout risks" --mode fast
python -m src.clawbot.cli scorecard "Design an onboarding automation agent" --mode fast
```

## Files

- `agent_card.json`: metadata card for registry/discovery

## Use in a notebook

Import adapters from `src/clawbot/multiagent.py` and register with your other agents.

```python
from src.clawbot.multiagent import MultiAgentBoard, ClawbotAdapter

board = MultiAgentBoard()
board.register(ClawbotAdapter())

result = board.run_one("clawbot", "Design an autonomous sales assistant")
print(result["answer"])
```

## Side-by-side with other agents

You can add your own adapters that follow the same shape:

- `name: str`
- `run(task: str) -> dict`

Then register them all with `MultiAgentBoard` and call `run_all(task)`.

## Submission checklist

- Confirm `moltbook/agent_card.json` points to `src/clawbot/multiagent.py:ClawbotAdapter`.
- Verify notebook `notebooks/moltbook_clawbot.ipynb` runs all cells from project root.
- Save one fresh `runs/latest_run.json` and `runs/latest_scorecard.json` artifact before demo.
- Keep one backup demo prompt ready in case web/provider calls are rate-limited.
