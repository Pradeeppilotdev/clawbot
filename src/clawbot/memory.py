from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunMemory:
    task: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def to_json(self) -> str:
        return json.dumps({"task": self.task, "events": self.events}, indent=2)


def persist_run(memory: RunMemory, output_dir: Path = Path("runs")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "latest_run.json"
    file_path.write_text(memory.to_json(), encoding="utf-8")
    return file_path


def persist_json(payload: dict[str, Any], file_name: str, output_dir: Path = Path("runs")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / file_name
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return file_path
