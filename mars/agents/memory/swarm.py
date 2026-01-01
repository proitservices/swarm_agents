"""
Memory Swarm coordinator
Manages sequential evaluation of all active memory agents (simulating parallelism)
Ensures prime seed thoughts are loaded only once and never duplicated or renamed
New thoughts from generator get fresh IDs and are appended only when truly novel
File location: ./mars/agents/memory/swarm.py
"""

# imports
from typing import Dict, Any, List, Set
from datetime import datetime

from mars.types import MARSState, Thought
from mars.agents.memory.single import create_single_memory_node
from mars.infrastructure.logging import get_thought_logger

# ──────────────────────────────────────────────────────────────────────────────
# Helper classes / functions
# ──────────────────────────────────────────────────────────────────────────────

def _deduplicate_thoughts(thoughts: List[Thought]) -> List[Thought]:
    """Return list of thoughts keeping only the most recent version of each thought_id"""
    if not thoughts:
        return []
    
    # Keep the latest version according to last_evaluated timestamp
    latest_by_id: Dict[str, Thought] = {}
    for t in thoughts:
        tid = t["thought_id"]
        current_ts = t.get("last_evaluated", "1970-01-01T00:00:00")
        existing_ts = latest_by_id.get(tid, {}).get("last_evaluated", "1970-01-01T00:00:00")
        if tid not in latest_by_id or current_ts > existing_ts:
            latest_by_id[tid] = t
    
    return list(latest_by_id.values())

def _safe_extract_content(last_msg: Any) -> str:
    """Safely extracts content from any LangChain message object (AIMessage, HumanMessage, etc.)"""
    if last_msg is None:
        return ""
    return getattr(last_msg, 'content', str(last_msg)) or ""

# ──────────────────────────────────────────────────────────────────────────────
# Core factory
# ──────────────────────────────────────────────────────────────────────────────

def create_memory_swarm_node(llm):
    """
    Factory that creates the memory swarm evaluation node.
    The node evaluates existing thoughts, decides on injection/reframing/new generation,
    prevents any thought ID duplication and keeps the state clean.
    """
    def swarm_node(state: MARSState) -> Dict[str, Any]:
        """
        Main swarm processing logic:
          - Deduplicates thoughts by id (keeping most recent version)
          - Evaluates each unique thought against current orchestrator context
          - Applies simple relevance decision (mock scoring for now)
          - Logs evaluation decisions
          - Queues highly relevant thoughts for injection
          - Reframes low-relevance thoughts in place
          - Triggers new thought generation on strong mismatch/conflict
          - Appends only genuinely new thoughts with unique IDs

        Args:
            state: Current MARS shared state

        Returns:
            Dict[str, Any]: Delta updates containing:
                - thoughts: updated list (with possible new entries)
                - injection_queue: thoughts ready for summary merging
                - active_agent: marker for current active component
        """
        thoughts = state.get("thoughts", [])
        if not thoughts:
            return {"injection_queue": [], "active_agent": "memory_swarm"}

        # ── Phase 1: Defensive deduplication ───────────────────────────────
        unique_thoughts = _deduplicate_thoughts(thoughts)
        existing_ids: Set[str] = {t["thought_id"] for t in unique_thoughts}

        working_thoughts = unique_thoughts.copy()  # we will append here if needed
        injection_queue: List[Thought] = []

        # ── Safe context extraction (fixes AIMessage.get() error) ──────────
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        context_text = _safe_extract_content(last_msg)
        context_snippet = (context_text[:400] + "...") if len(context_text) > 400 else context_text

        # ── Phase 2: Evaluate each thought ─────────────────────────────────
        for thought in working_thoughts:
            # Create specialized memory evaluation agent for this thought
            memory_node = create_single_memory_node(llm, thought)
            eval_result = memory_node(state)
            
            if not eval_result.get("messages"):
                continue

            # Safe content extraction from evaluation result
            eval_msg = eval_result["messages"][-1]
            response_text = _safe_extract_content(eval_msg).lower()

            # ── Simple mock scoring (to be replaced with real CoT + embedding scoring)
            is_applicable = any(word in response_text for word in ["applicable", "yes", "relevant", "supports"])
            relevance_score = 0.85 if is_applicable else 0.42

            # Decision logic
            decision = "inject" if relevance_score > 0.70 else "reframe"
            if any(word in response_text for word in ["mismatch", "conflict", "contradict", "outdated"]):
                decision = "generate_new"

            # Logging
            thought_logger = get_thought_logger(thought)
            thought_logger.log_evaluation(
                current_orchestrator_snippet=context_snippet,
                relevance_score=relevance_score,
                decision=decision,
                reasoning=f"Response snippet: {response_text[:180]}..."
            )

            # Update thought metadata (mutates in place)
            thought["relevance_score"] = relevance_score
            thought["last_evaluated"] = datetime.utcnow().isoformat()

            # ── Apply decision ─────────────────────────────────────────────
            if relevance_score > 0.70:
                injection_queue.append(thought)

            if decision == "reframe":
                # Minimal re-framing marker - can be greatly improved later
                current_narrative = thought.get("narrative", "")
                thought["narrative"] = f"{current_narrative} [re-framed with new context: {context_snippet[:100]}...]"

            elif decision == "generate_new":
                # Controlled generation of new thought (limited to prevent hangs)
                try:
                    from mars.agents.thought_generator.agent import create_thought_generator_node
                    thought_gen = create_thought_generator_node(llm, max_steps=2)  # Limit to 2 steps during swarm
                    gen_delta = thought_gen(state)
                    new_thoughts = gen_delta.get("thoughts", [])
                    
                    for new_thought in new_thoughts:
                        new_id = new_thought["thought_id"]
                        if new_id not in existing_ids:
                            working_thoughts.append(new_thought)
                            existing_ids.add(new_id)
                        # Silently skip duplicates
                        
                except (ImportError, Exception) as e:
                    print(f"Warning: New thought generation failed → {str(e)}")

        # ── Final result delta ─────────────────────────────────────────────
        return {
            "thoughts": working_thoughts,
            "injection_queue": injection_queue,
            "active_agent": "memory_swarm"
        }

    return swarm_node