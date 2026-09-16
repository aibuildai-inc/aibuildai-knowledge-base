"""What a produced skill looks like on disk: directory naming + reference frontmatter.

Two halves that together define a producer's on-disk output shape:

  - Directory naming. A skill's address is the triple (marketplace, plugin, skill),
    so its directory sits exactly one level under its plugin's ``skills/`` and holds
    a ``SKILL.md``; kb's ``SkillCorpus`` rejects any other shape. Two producers may
    use the same skill name because their plugins differ. Where a producer's own
    name carries several semantic segments, it joins each slugified segment with
    ``__``; slugify removes underscores, so no segment can itself contain ``__`` and
    the join stays unambiguous (``split("__", 1)`` round-trips).

  - Reference frontmatter. The cross-pipeline schema every ``reference/*.md``
    frontmatter is validated against (Phase 5).
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter


def slugify(text: str) -> str:
    """Lowercase, hyphenate spaces, keep only ``[a-z0-9-]`` (underscores removed)."""
    s = re.sub(r"[^a-z0-9-]", "", text.lower().replace(" ", "-"))
    return s or "uncategorized"


def skill_dir_name(venue_year: str, category: str) -> str:
    """Flat single-component skill directory name: ``<venue-year>__<category>``."""
    return f"{slugify(venue_year)}__{slugify(category)}"


class _BaseEntry(BaseModel):
    """Common fields every reference/*.md frontmatter carries."""
    model_config = ConfigDict(extra="forbid")
    entry_type: str
    title: str
    source: str
    tags: list[str] = Field(default_factory=list, max_length=8)
    tldr: str = ""
    categories: list[str] = Field(default_factory=list, max_length=3)


class RepoEntry(_BaseEntry):
    entry_type: Literal["repo"]
    language: str | None = None
    license: str | None = None
    install: str = ""


class PaperEntry(_BaseEntry):
    entry_type: Literal["paper"]
    venue: str = ""
    year: int | None = None
    pdf_url: str | None = None


class AwesomeListEntry(_BaseEntry):
    entry_type: Literal["awesome-list"]
    upstream_list: str
    arxiv_id: str | None = None
    doi: str | None = None


Entry = Annotated[
    Union[RepoEntry, PaperEntry, AwesomeListEntry],
    Discriminator("entry_type"),
]

_ENTRY_ADAPTER: TypeAdapter = TypeAdapter(Entry)


def validate_entry(frontmatter: dict) -> Entry:
    """Parse a frontmatter dict into the correct Entry subclass via entry_type."""
    return _ENTRY_ADAPTER.validate_python(frontmatter)


def dump_frontmatter(entry: _BaseEntry) -> str:
    """Serialize an Entry to a YAML frontmatter block (no leading/trailing `---`)."""
    data = entry.model_dump(exclude_none=True)
    data = {k: v for k, v in data.items() if v != ""}
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
