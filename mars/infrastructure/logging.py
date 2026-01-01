"""
Centralized logging for MARS swarm agents
Handles orchestrator reasoning trace, per-thought memory agent logs, and thought generator steps
File location: ./mars/infrastructure/logging.py
"""

# imports
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mars.types import Thought

# ──────────────────────────────────────────────────────────────────────────────
# Helper classes
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Operational classes
# ──────────────────────────────────────────────────────────────────────────────

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
        """Record one complete orchestrator step"""
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
        super().__init__(self.BASE_DIR / f"thought-{thought_id}.log")

    def log_evaluation(
        self,
        current_orchestrator_snippet: str,
        relevance_score: float,
        decision: str,  # "inject" | "reframe" | "generate_new"
        reasoning: str
    ) -> None:
        """Record one evaluation pass by this memory agent"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "orchestrator_snippet": current_orchestrator_snippet.strip(),
            "relevance_score": relevance_score,
            "decision": decision,
            "reasoning": reasoning.strip()
        }
        self.append_json(entry)


class ThoughtGeneratorLogger(AppendOnlyFileLogger):
    """
    Logs each step of the thought generation process (prompt + reply).
    File: ./mars/memories/thoughts_generator.log
    """

    PATH = Path("mars/memories/thoughts_generator.log")

    def __init__(self):
        super().__init__(self.PATH)

    def log_step(
        self,
        step: int,
        prompt: str,
        reply: str,
        **extra: Any
    ) -> None:
        """
        Record one guided prompt and its reply.

        Args:
            step: Sequence number of this generation step
            prompt: The full prompt sent to the LLM
            reply: The raw response from the LLM
            **extra: Any additional metadata (optional)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "prompt": prompt.strip(),
            "reply": reply.strip(),
            **extra
        }
        self.append_json(entry)

    def log_start(self, num_steps: int) -> None:
        """Log the beginning of a generation cycle"""
        self.append(f"Starting thought generation cycle — {num_steps} steps planned")

    def log_warning(self, message: str) -> None:
        """Log a non-fatal issue during generation"""
        self.append(f"WARNING: {message}")


# ──────────────────────────────────────────────────────────────────────────────
# Core utilities (factories / singletons)
# ──────────────────────────────────────────────────────────────────────────────

def get_orchestrator_logger() -> OrchestratorTraceLogger:
    """Get the shared orchestrator dialogue logger"""
    return OrchestratorTraceLogger()


def get_thought_logger(thought: Thought) -> ThoughtLogger:
    """Get logger for a specific thought"""
    return ThoughtLogger(thought["thought_id"])


def get_thought_generator_logger() -> ThoughtGeneratorLogger:
    """Get the shared thought generator logger"""
    return ThoughtGeneratorLogger()