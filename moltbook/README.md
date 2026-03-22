# Moltbook Integration Pack

This folder packages Clawbot for notebook-style multi-agent setups.

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
