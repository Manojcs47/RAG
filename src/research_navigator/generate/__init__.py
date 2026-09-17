"""M2 generation: grounded answers with validated inline citations."""

from __future__ import annotations

from .citations import format_authors, source_label, to_citation
from .factory import build_generator
from .generator import Generator, Retrieving
from .llm import ChatMessage, LanguageModel, OpenAIChatModel, build_llm
from .markers import parse_markers, render_citations, uncited_sentences
from .models import Answer, Citation, Source
from .settings import GenerateSettings
from .sources import build_sources

__all__ = [
    "Answer",
    "ChatMessage",
    "Citation",
    "GenerateSettings",
    "Generator",
    "LanguageModel",
    "OpenAIChatModel",
    "Retrieving",
    "Source",
    "build_generator",
    "build_llm",
    "build_sources",
    "format_authors",
    "parse_markers",
    "render_citations",
    "source_label",
    "to_citation",
    "uncited_sentences",
]
