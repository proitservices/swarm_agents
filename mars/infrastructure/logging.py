# ./mars/infrastructure/logging.py
"""
Centralized logging for MARS swarm agents
Handles orchestrator reasoning trace and per-thought memory agent logs
"""

# imports
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mars.types import Thought


# helper classes
class AppendOnlyFileLogger:
    """Simple append-only file logger with ISO timestamp prefix"""
    def __init__(self, filepath: str | Path):
        self.path = Path(filepath)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, message: str) -> None:
        """Append a line with timestamp prefix"""
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"[{ts}] {message.strip()}\n"
        content = self.path.read_text(encoding="utf-8") + line if self.path.exists() else line
        self.path.write_text(content, encoding="utf-8")

    def append_json(self, data: Dict[str, Any]) -> None:
        """Append one JSON line (structured event)"""
        self.append(json.dumps(data, ensure_ascii=False))


# operational classes
class OrchestratorTraceLogger(AppendOnlyFileLogger):
    """
    Logs each orchestrator reasoning step with injected context and active thoughts.
    File: ./mars/memories/orchestrator_dialogue.log
    """
    PATH = Path("mars/memories/orchestrator_dialogue.log")

    def __init__(self):
        super().__init__(self.PATH)

    def log_step(
        self,
        step_number: int,
        orchestrator_output: str,
        injected_summary: Optional[str] = None,
        active_thought_ids: Optional[List[str]] = None
    ) -> None:
        """
        Record one complete orchestrator step
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_number,
            "orchestrator_output": orchestrator_output.strip(),
            "injected_summary": injected_summary.strip() if injected_summary else None,
            "active_thoughts": active_thought_ids or []
        }
        self.append_json(entry)


class ThoughtLogger(AppendOnlyFileLogger):
    """
    Per-thought logger for memory agent evaluations.
    File pattern: ./mars/memories/raw/thought-<id>.log
    """
    BASE_DIR = Path("mars/memories/raw")

    def __init__(self, thought_id: str):
        super().__init__(self.BASE_DIR / f"{thought_id}.log")

    def log_evaluation(
        self,
        current_orchestrator_snippet: str,
        relevance_score: float,
        decision: str,           # "inject" | "reframe" | "generate_new"
        reasoning: str
    ) -> None:
        """
        Record one evaluation pass by this memory agent
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "orchestrator_snippet": current_orchestrator_snippet.strip(),
            "relevance_score": relevance_score,
            "decision": decision,
            "reasoning": reasoning.strip()
        }
        self.append_json(entry)


# Core utilities (factories / singletons)
def get_orchestrator_logger() -> OrchestratorTraceLogger:
    """Get the shared orchestrator dialogue logger"""
    return OrchestratorTraceLogger()


def get_thought_logger(thought: Thought) -> ThoughtLogger:
    """Get logger for a specific thought"""
    return ThoughtLogger(thought["thought_id"])