"""Assemble a Generator from a retriever, an LLM, and settings."""

from __future__ import annotations

from .generator import Generator, Retrieving
from .llm import LanguageModel
from .settings import GenerateSettings


def build_generator(
    *,
    retriever: Retrieving,
    llm: LanguageModel,
    settings: GenerateSettings,
) -> Generator:
    return Generator(retriever=retriever, llm=llm, settings=settings)
