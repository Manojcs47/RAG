from __future__ import annotations

from research_navigator.generate import (
    Answer,
    ChatMessage,
    GenerateSettings,
    Generator,
    build_sources,
    format_authors,
    parse_markers,
    render_citations,
    source_label,
)
from research_navigator.generate.citations import extract_arxiv_id, to_citation
from research_navigator.retrieve.models import (
    InferredFilters,
    QueryAnalysis,
    QueryIntent,
    RetrievalResult,
    RetrievedChunk,
)

SETTINGS = GenerateSettings()


# ------------------------------ fakes --------------------------------------- #
class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(messages)
        return self.reply


def _chunk(pid, doc_id, score, **payload) -> RetrievedChunk:  # type: ignore[no-untyped-def]
    payload.setdefault("doc_id", doc_id)
    return RetrievedChunk(point_id=pid, score=score, payload=payload)


def _result(chunks, *, refused=False, confidence=0.8, intent=QueryIntent.CONCEPT):  # type: ignore[no-untyped-def]
    analysis = QueryAnalysis(raw="q", normalized="q", intent=intent, filters=InferredFilters())
    return RetrievalResult(
        analysis=analysis,
        chunks=tuple(chunks),
        confidence=confidence,
        refused=refused,
        fusion="rrf",
        dense_only=False,
        top_k=8,
        filtered=False,
    )


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result

    def retrieve(self, query: str, *, now_year: int | None = None) -> RetrievalResult:
        return self.result


# ------------------------- citation formatting ------------------------------ #
def test_format_authors_variants() -> None:
    assert format_authors([]) == ""
    assert format_authors(["Ada"]) == "Ada"
    assert format_authors(["Ada", "Bob"]) == "Ada and Bob"
    assert format_authors(["Ada", "Bob", "Cy"]) == "Ada et al."


def test_extract_arxiv_id() -> None:
    assert extract_arxiv_id("arxiv-2305.14314") == "2305.14314"
    assert extract_arxiv_id(None, "https://arxiv.org/abs/2408.00118") == "2408.00118"
    assert extract_arxiv_id("hf-nlp-ch3") is None


def test_source_label_maps() -> None:
    labels = SETTINGS.source_labels
    assert (
        source_label("arxiv_paper", doc_id="arxiv-2305.14314", source_url=None, labels=labels)
        == "arXiv:2305.14314"
    )
    assert (
        source_label("survey_blog", doc_id="lilog-x", source_url=None, labels=labels) == "Lil'Log"
    )
    assert (
        source_label("course_chapter", doc_id="hf-1", source_url=None, labels=labels)
        == "Hugging Face Learn"
    )


# --------------------------- source dedup ----------------------------------- #
def test_build_sources_collapses_same_doc() -> None:
    chunks = [
        _chunk(1, "d1", 0.9, title="Paper One", section_title="Intro", text="alpha"),
        _chunk(2, "d1", 0.5, title="Paper One", section_title="Method", text="beta"),
        _chunk(3, "d2", 0.7, title="Paper Two", section_title="Abstract", text="gamma"),
    ]
    sources = build_sources(chunks, SETTINGS)
    assert len(sources) == 2  # d1 collapsed
    assert sources[0].doc_id == "d1"  # best score first
    assert sources[0].section_title == "Intro"  # most-relevant chunk's section
    assert "alpha" in sources[0].text and "beta" in sources[0].text


# --------------------------- marker validation ------------------------------ #
def test_parse_markers_grouped() -> None:
    assert parse_markers("a [1] b [2, 3] c [10]") == [1, 2, 3, 10]


def test_render_drops_fabricated_and_renumbers() -> None:
    src = build_sources(
        [
            _chunk(1, "d1", 0.9, title="One", text="x"),
            _chunk(2, "d2", 0.8, title="Two", text="y"),
        ],
        SETTINGS,
    )
    # model cited [1], a fabricated [9], and [2]
    text = "Claim A [1]. Claim B [9]. Claim C [2]."
    clean, used, dropped = render_citations(text, src)
    assert dropped == [9]
    assert "[9]" not in clean
    # renumbered by first appearance: 1 -> 1, 2 -> 2
    assert [s.index for s in used] == [1, 2]
    assert {s.doc_id for s in used} == {"d1", "d2"}


def test_render_renumbers_noncontiguous() -> None:
    src = build_sources(
        [
            _chunk(1, "d1", 0.9, title="One", text="x"),
            _chunk(2, "d2", 0.8, title="Two", text="y"),
            _chunk(3, "d3", 0.7, title="Three", text="z"),
        ],
        SETTINGS,
    )
    clean, used, dropped = render_citations("P [3]. Q [1].", src)
    assert dropped == []
    assert clean.startswith("P [1]. Q [2].")
    assert [s.doc_id for s in used] == ["d3", "d1"]  # appearance order


# ------------------------------ generator ----------------------------------- #
def _sources_result():  # type: ignore[no-untyped-def]
    return _result(
        [
            _chunk(
                1,
                "arxiv-2305.14314",
                0.9,
                title="QLoRA",
                authors=["A", "B", "C"],
                year=2023,
                content_type="arxiv_paper",
                section_title="Abstract",
                source_url="https://arxiv.org/abs/2305.14314",
                text="QLoRA fine-tunes.",
            ),
            _chunk(
                2,
                "hf-ch3",
                0.6,
                title="NLP Course",
                authors=[],
                year=2024,
                content_type="course_chapter",
                section_title="Tokenizers",
                source_url="https://hf.co/learn",
                text="Tokenizers split text.",
            ),
        ]
    )


def test_generator_produces_cited_answer() -> None:
    llm = FakeLLM("QLoRA enables efficient fine-tuning [1]. Tokenizers split text [2].")
    gen = Generator(retriever=FakeRetriever(_sources_result()), llm=llm, settings=SETTINGS)
    ans = gen.answer("what is qlora")
    assert not ans.refused
    assert len(ans.citations) == 2
    assert ans.citations[0].source == "arXiv:2305.14314"
    assert ans.citations[0].authors == "A et al."
    # every marker in the text maps to a listed citation
    used_idx = set(parse_markers(ans.text))
    assert used_idx <= {c.index for c in ans.citations}


def test_generator_refuses_on_low_confidence() -> None:
    llm = FakeLLM("should not be called")
    gen = Generator(
        retriever=FakeRetriever(_result([], refused=True, confidence=0.1)),
        llm=llm,
        settings=SETTINGS,
    )
    ans = gen.answer("off corpus")
    assert ans.refused and ans.reason == "low_confidence"
    assert ans.citations == ()
    assert llm.calls == []  # LLM not invoked when retrieval already refused


def test_generator_refuses_on_sentinel() -> None:
    llm = FakeLLM(SETTINGS.refusal_sentinel)
    gen = Generator(retriever=FakeRetriever(_sources_result()), llm=llm, settings=SETTINGS)
    ans = gen.answer("q")
    assert ans.refused and ans.reason == "model_insufficient"


def test_generator_refuses_when_only_fabricated_citations() -> None:
    llm = FakeLLM("Confident nonsense [9].")  # only an out-of-range marker
    gen = Generator(retriever=FakeRetriever(_sources_result()), llm=llm, settings=SETTINGS)
    ans = gen.answer("q")
    assert ans.refused and ans.reason == "no_valid_citations"


def test_answer_render_includes_sources_block() -> None:
    llm = FakeLLM("QLoRA fine-tunes models [1].")
    gen = Generator(retriever=FakeRetriever(_sources_result()), llm=llm, settings=SETTINGS)
    ans = gen.answer("q")
    rendered = ans.render()
    assert "Sources:" in rendered
    assert "[1] QLoRA." in rendered
    assert "arXiv:2305.14314" in rendered


def test_to_citation_two_authors() -> None:
    src = build_sources(
        [
            _chunk(
                1,
                "d1",
                0.9,
                title="T",
                authors=["X", "Y"],
                year=2022,
                content_type="arxiv_paper",
                source_url="https://arxiv.org/abs/2201.00001",
            )
        ],
        SETTINGS,
    )[0]
    cit = to_citation(src, 1, etal_threshold=3, labels=SETTINGS.source_labels)
    assert cit.authors == "X and Y"
    assert isinstance(cit, type(cit))


def test_answer_is_frozen_dataclass() -> None:
    a = Answer(
        query="q",
        text="t",
        citations=(),
        refused=True,
        reason="x",
        intent="concept",
        confidence=0.0,
    )
    assert a.refused
