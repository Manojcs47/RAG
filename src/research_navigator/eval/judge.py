"""Citation-faithfulness judge (LLM-as-judge) with an explicit, written rubric.

The rubric is stated in full below and reproduced in ADR-0008 so the grading criteria
are transparent (the assignment asks us to "define the rubric explicitly"). The judge
asks the model two questions per answer:

1. **Grounding** — does every cited claim follow from the text of the source it cites?
2. **Attribution** — are there substantive claims left uncited?

The model must reply as a single JSON object. Parsing is defensive: a fenced ```json
block is tolerated, and an unparseable reply is *logged and recorded* as
``parse_ok=False`` (counted as non-faithful) rather than silently dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from research_navigator.eval.ports import CitedSource, SupportsComplete

log = structlog.get_logger(__name__)

CITATION_JUDGE_RUBRIC = """\
You are a strict evaluator of citation faithfulness for a retrieval-augmented answer.
You are given a QUESTION, an ANSWER whose sentences carry inline markers like [1], [2],
and the SOURCES those markers refer to (each with its exact retrieved text).

Judge the answer on two axes:
1. GROUNDING: for every claim that carries a citation marker, does that claim actually
   follow from the text of the cited source? A claim is UNSUPPORTED if the cited source
   does not state or clearly imply it, or if it cites the wrong source.
2. ATTRIBUTION: does every substantive factual claim carry a citation marker? A
   substantive claim stated with no marker is an UNCITED claim. (Generic transitions,
   restatements of the question, and hedging are not substantive.)

Do not use outside knowledge; judge only against the provided SOURCES.

Reply with a SINGLE JSON object and nothing else, with exactly these keys:
{
  "supported_claims": <int>,      // cited claims that follow from their source
  "unsupported_claims": <int>,    // cited claims that do NOT follow from their source
  "uncited_claims": <int>,        // substantive claims with no citation marker
  "faithful": <bool>,      // true iff unsupported_claims and uncited_claims are 0
  "rationale": "<one or two sentences>"
}
"""


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Structured, serialisable outcome of one faithfulness judgement."""

    supported_claims: int
    unsupported_claims: int
    uncited_claims: int
    faithful: bool
    score: float
    rationale: str
    parse_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "uncited_claims": self.uncited_claims,
            "faithful": self.faithful,
            "score": round(self.score, 4),
            "rationale": self.rationale,
            "parse_ok": self.parse_ok,
        }

    @classmethod
    def unparseable(cls, rationale: str) -> JudgeVerdict:
        return cls(0, 0, 0, faithful=False, score=0.0, rationale=rationale, parse_ok=False)

    @classmethod
    def no_citations(cls) -> JudgeVerdict:
        """An answer that makes claims but cites nothing is maximally unfaithful."""
        return cls(
            0,
            0,
            0,
            faithful=False,
            score=0.0,
            rationale="answer has no citations",
            parse_ok=True,
        )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(reply: str) -> dict[str, Any] | None:
    fenced = _JSON_FENCE.search(reply)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        loose = _FIRST_OBJECT.search(reply)
        candidate = loose.group(0) if loose else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _score(supported: int, unsupported: int, uncited: int) -> float:
    """Faithfulness score in [0, 1].

    Grounding fraction = supported / cited, penalised by uncited claims. With no cited
    claims at all the grounding term is 0 (nothing was grounded).
    """
    cited = supported + unsupported
    grounding = supported / cited if cited else 0.0
    total_claims = cited + uncited
    attribution = cited / total_claims if total_claims else 0.0
    return round(0.5 * grounding + 0.5 * attribution, 4)


class CitationJudge:
    """Runs the rubric against a :class:`SupportsComplete` model."""

    def __init__(self, model: SupportsComplete, *, min_score: float) -> None:
        self._model = model
        self._min_score = min_score

    def build_prompt(self, question: str, answer_text: str, sources: Sequence[CitedSource]) -> str:
        blocks = [
            (
                f"[{i + 1}] doc_id={s.doc_id} title={s.title!r} section={s.section!r}\n{s.text}"
            ).strip()
            for i, s in enumerate(sources)
        ]
        return (
            f"{CITATION_JUDGE_RUBRIC}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer_text}\n\n"
            f"SOURCES:\n" + "\n\n".join(blocks) + "\n"
        )

    def judge(
        self,
        question: str,
        answer_text: str,
        sources: Sequence[CitedSource],
    ) -> JudgeVerdict:
        if not sources:
            return JudgeVerdict.no_citations()
        prompt = self.build_prompt(question, answer_text, sources)
        try:
            reply = self._model.complete(prompt)
        except Exception as exc:
            log.warning("judge_model_call_failed", error=str(exc))
            return JudgeVerdict.unparseable(f"judge model error: {exc}")

        parsed = _extract_json(reply)
        if parsed is None:
            log.warning("judge_reply_unparseable", reply_preview=reply[:200])
            return JudgeVerdict.unparseable("judge reply was not valid JSON")

        try:
            supported = int(parsed["supported_claims"])
            unsupported = int(parsed["unsupported_claims"])
            uncited = int(parsed["uncited_claims"])
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("judge_reply_missing_fields", error=str(exc), parsed=parsed)
            return JudgeVerdict.unparseable(f"judge reply missing/invalid fields: {exc}")

        score = _score(supported, unsupported, uncited)
        # Trust the model's boolean if present and coherent; else derive from counts.
        raw_faithful = parsed.get("faithful")
        faithful = (
            bool(raw_faithful)
            if isinstance(raw_faithful, bool)
            else (unsupported == 0 and uncited == 0)
        )
        # Gate the headline 'faithful' by the configured score threshold too.
        faithful = faithful and score >= self._min_score
        rationale = str(parsed.get("rationale", "")).strip()
        return JudgeVerdict(
            supported_claims=supported,
            unsupported_claims=unsupported,
            uncited_claims=uncited,
            faithful=faithful,
            score=score,
            rationale=rationale,
            parse_ok=True,
        )
