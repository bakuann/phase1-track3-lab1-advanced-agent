from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .mock_runtime import actor_answer, evaluator, failure_mode_for_qid, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            actor = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            answer = actor.text
            judge = evaluator(example, answer)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=actor.token_estimate + judge.token_estimate,
                latency_ms=actor.latency_ms + judge.latency_ms,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, answer, judge)
                reflections.append(reflection)
                reflection_memory.append(
                    f"Attempt {attempt_id} failed: {reflection.lesson} "
                    f"Next strategy: {reflection.next_strategy}"
                )
                trace.reflection = reflection
                trace.token_estimate += reflection.token_estimate
                trace.latency_ms += reflection.latency_ms

            traces.append(trace)
            if judge.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else failure_mode_for_qid(example.qid)
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
