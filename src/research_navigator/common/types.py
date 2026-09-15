"""Low-level shared enums."""

from __future__ import annotations

from enum import StrEnum


class ContentType(StrEnum):
    arxiv_paper = "arxiv_paper"
    course_chapter = "course_chapter"
    survey_blog = "survey_blog"
    lab_blog_post = "lab_blog_post"
