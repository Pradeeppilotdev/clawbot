# Clawbot: Hackathon-Ready Agent Starter

Clawbot is a practical, extensible AI agent scaffold designed for rapid iteration.

## What you get

- Fast planner -> research -> writer pipeline
- Provider abstraction for LLM calls (OpenAI, Gemini, local mock)
- Circuit-breaker provider failover for resilient runs
- Tool execution framework with safety rails
- Built-in web tools (`web_search`, `fetch_url`)
- Local data tools (`read_text_file`, `csv_profile`)
- JSON short-term memory traces for each run
- Lightweight benchmark runner to track quality over time
- Judge-facing scorecard output (`task_success`, `latency_score`, `recovery_score`)
- CLI entrypoint for quick demos

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment template:

```bash
copy .env.example .env
```

4. In `.env`, choose provider and key.

Gemini example:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

OpenAI example:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Optional resilience + speed tuning:

```bash
GEMINI_FALLBACK_MODELS=gemini-2.0-flash-lite,gemini-flash-latest,gemini-2.5-flash
LLM_MAX_RETRIES=2
LLM_RETRY_BASE_DELAY_SECONDS=1.5
LLM_MIN_INTERVAL_SECONDS=0.8
LLM_CB_FAILURES=2
LLM_CB_COOLDOWN_SECONDS=60
LLM_CACHE_TTL_SECONDS=180
AGENT_MODE=fast
ENABLE_WEB_RESEARCH=1
RESEARCH_QUERY_BUDGET=2
ENABLE_CRITIC=0
```

5. Run a sample task:

```bash
python -m src.clawbot.cli run "Research top 3 product ideas and propose one MVP plan"
```

Run with explicit mode:

```bash
python -m src.clawbot.cli run "Design a GTM plan for an AI tool" --mode fast
python -m src.clawbot.cli run "Design a GTM plan for an AI tool" --mode balanced
python -m src.clawbot.cli run "Design a GTM plan for an AI tool" --mode deep
```

Generate a judge-facing scorecard:

```bash
python -m src.clawbot.cli scorecard "Design an onboarding automation agent" --mode fast
```

6. Run benchmark tasks:

```bash
python -m src.clawbot.cli eval
```

## Performance profiles

- `fast`: lowest latency, minimal research, no reviewer pass by default.
- `balanced`: better depth/latency trade-off.
- `deep`: richer research and optional reviewer pass for final-quality output.

## Project structure

- `src/clawbot/config.py`: environment and runtime settings
- `src/clawbot/llm.py`: provider clients, failover, circuit breaker, cache
- `src/clawbot/tools.py`: tool interface + web/data tools
- `src/clawbot/memory.py`: run memory storage
- `src/clawbot/agent.py`: planner/research/writer pipeline
- `src/clawbot/scorecard.py`: judge-facing score model
- `src/clawbot/eval.py`: benchmark harness
- `src/clawbot/cli.py`: command-line entrypoint

## Notes

If a provider is repeatedly failing, the circuit breaker opens that provider temporarily and routes to healthy alternatives.

## Moltbook Integration

- Multi-agent adapter module: `src/clawbot/multiagent.py`
- Integration pack: `moltbook/agent_card.json`
- Notebook starter: `notebooks/moltbook_clawbot.ipynb`

You can register Clawbot with other custom adapters using `MultiAgentBoard` and run side-by-side with `run_all(task)`.
