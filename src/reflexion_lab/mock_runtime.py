from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


FIRST_ATTEMPT_WRONG = {
    "hp2": "London",
    "hp4": "Atlantic Ocean",
    "hp6": "Red Sea",
    "hp8": "Andes",
}
FAILURE_MODE_BY_QID = {
    "hp2": "incomplete_multi_hop",
    "hp4": "wrong_final_answer",
    "hp6": "entity_drift",
    "hp8": "entity_drift",
}


@dataclass
class RuntimeTextResult:
    text: str
    token_estimate: int
    latency_ms: int


def runtime_mode() -> str:
    return os.getenv("REFLEXION_RUNTIME", "mock").strip().lower()


def failure_mode_for_qid(qid: str) -> str:
    return FAILURE_MODE_BY_QID.get(_base_qid(qid), "wrong_final_answer")


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> RuntimeTextResult:
    if runtime_mode() == "mock":
        text = _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)
        return _mock_result(text, 120 + 35 * attempt_id + 30 * len(reflection_memory))

    user = "\n\n".join(
        [
            f"Question:\n{example.question}",
            f"Context:\n{_format_context(example)}",
            f"Reflection memory:\n{_format_reflections(reflection_memory)}",
        ]
    )
    return _chat(ACTOR_SYSTEM, user, json_mode=False)


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if runtime_mode() == "mock":
        return _mock_evaluator(example, answer)

    user = "\n\n".join(
        [
            f"Question:\n{example.question}",
            f"Gold answer:\n{example.gold_answer}",
            f"Predicted answer:\n{answer}",
            f"Context:\n{_format_context(example)}",
        ]
    )
    response = _chat(EVALUATOR_SYSTEM, user, json_mode=True)
    payload = _parse_json_object(response.text)
    if not payload:
        score = int(normalize_answer(example.gold_answer) == normalize_answer(answer))
        payload = {
            "score": score,
            "reason": "Evaluator response was not valid JSON; fell back to exact normalized match.",
            "missing_evidence": [] if score else ["Could not parse evaluator evidence."],
            "spurious_claims": [] if score else [answer],
            "confidence": 0.3,
        }
    payload["token_estimate"] = response.token_estimate
    payload["latency_ms"] = response.latency_ms
    return JudgeResult.model_validate(payload)


def reflector(example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> ReflectionEntry:
    if runtime_mode() == "mock":
        return _mock_reflector(example, attempt_id, judge)

    user = "\n\n".join(
        [
            f"Question:\n{example.question}",
            f"Context:\n{_format_context(example)}",
            f"Failed answer:\n{answer}",
            f"Evaluator reason:\n{judge.reason}",
            f"Missing evidence:\n{json.dumps(judge.missing_evidence)}",
            f"Spurious claims:\n{json.dumps(judge.spurious_claims)}",
        ]
    )
    response = _chat(REFLECTOR_SYSTEM, user, json_mode=True)
    payload = _parse_json_object(response.text) or {
        "failure_reason": judge.reason,
        "lesson": "The next attempt must verify each reasoning hop against the provided context.",
        "next_strategy": "Re-read the context, identify the first-hop entity, then verify the final answer in the second-hop evidence.",
    }
    payload["attempt_id"] = attempt_id
    payload["token_estimate"] = response.token_estimate
    payload["latency_ms"] = response.latency_ms
    return ReflectionEntry.model_validate(payload)


def _mock_actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> str:
    base_qid = _base_qid(example.qid)
    if base_qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[base_qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[base_qid]
    return example.gold_answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(
            score=1,
            reason="Final answer matches the gold answer after normalization.",
            confidence=1.0,
            token_estimate=55,
            latency_ms=25,
        )
    if normalize_answer(answer) == "london":
        return JudgeResult(
            score=0,
            reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
            missing_evidence=["Need to identify the river that flows through London."],
            spurious_claims=[],
            confidence=0.95,
            token_estimate=70,
            latency_ms=30,
        )
    return JudgeResult(
        score=0,
        reason="The final answer selected the wrong second-hop entity.",
        missing_evidence=["Need to ground the answer in the second paragraph."],
        spurious_claims=[answer],
        confidence=0.9,
        token_estimate=68,
        latency_ms=30,
    )


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = (
        "Do the second hop explicitly: birthplace city -> river through that city."
        if _base_qid(example.qid) == "hp2"
        else "Verify the final entity against the second paragraph before answering."
    )
    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
        next_strategy=strategy,
        token_estimate=90,
        latency_ms=35,
    )


def _chat(system: str, user: str, json_mode: bool) -> RuntimeTextResult:
    mode = runtime_mode()
    started = time.perf_counter()
    if mode == "openai":
        text, tokens = _openai_chat(system, user, json_mode)
    elif mode == "ollama":
        text, tokens = _ollama_chat(system, user, json_mode)
    else:
        raise ValueError(f"Unsupported REFLEXION_RUNTIME={mode!r}. Use mock, openai, or ollama.")
    latency_ms = int((time.perf_counter() - started) * 1000)
    return RuntimeTextResult(text=text.strip(), token_estimate=tokens, latency_ms=latency_ms)


def _openai_chat(system: str, user: str, json_mode: bool) -> tuple[str, int]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when REFLEXION_RUNTIME=openai.")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = _post_json(
        "https://api.openai.com/v1/chat/completions",
        payload,
        {"Authorization": f"Bearer {api_key}"},
    )
    text = data["choices"][0]["message"]["content"]
    tokens = int(data.get("usage", {}).get("total_tokens") or _estimate_tokens(system, user, text))
    return text, tokens


def _ollama_chat(system: str, user: str, json_mode: bool) -> tuple[str, int]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0},
    }
    if json_mode:
        payload["format"] = "json"
    data = _post_json(f"{base_url}/api/chat", payload, {})
    text = data.get("message", {}).get("content", "")
    tokens = int(data.get("prompt_eval_count", 0) + data.get("eval_count", 0))
    return text, tokens or _estimate_tokens(system, user, text)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc


def _mock_result(text: str, tokens: int) -> RuntimeTextResult:
    return RuntimeTextResult(text=text, token_estimate=tokens, latency_ms=20 + tokens // 6)


def _format_context(example: QAExample) -> str:
    return "\n".join(f"[{chunk.title}] {chunk.text}" for chunk in example.context)


def _format_reflections(reflection_memory: list[str]) -> str:
    if not reflection_memory:
        return "None"
    return "\n".join(f"- {item}" for item in reflection_memory)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _estimate_tokens(*parts: str) -> int:
    return max(1, sum(max(1, len(part) // 4) for part in parts))


def _base_qid(qid: str) -> str:
    return qid.split("_", 1)[0]
