"""Definition and generation of a single playbook skill.

A playbook skill is the unit this pipeline produces: a principle-index routing
table (SKILL.md), a synthesised conclusion layer (principles/NN.md), and the raw
ranked writeups it was summarized from (references/<slug>-rank-N.md).

`pipeline.py` drives the source -> corpus build; this module owns what one skill
IS: its schema, the frozen writer prompt, and the on-disk layout writer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


# ---------- The writeup record (raw source text for one solution) ----------

@dataclass
class WriteupRecord:
    """One resolved top-solution writeup: the raw text behind a references/ file."""
    comp_slug: str
    rank: int
    topic_id: int
    topic_title: str
    link: str
    kind: str
    text: str


# ---------- Output schema ----------

class Principle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short action-oriented title, e.g. 'Choose Spectrogram Representation Based on Domain Characteristics'.")
    consensus: str = Field(description="How many top solutions use this pattern, e.g. '5/5', '3/5'. If only 1-2 use it, explain why it's still worth including.")
    description: str = Field(description="1-2 sentence statement of the principle.")
    why_it_matters: str = Field(description="Explanation of the underlying mechanism -- why this affects outcomes.")
    when_to_use: str = Field(description="Conditions under which this principle applies. What constraints or problem characteristics make it relevant?")
    when_not_to_use: str = Field(description="Conditions under which this principle does NOT apply or is harmful. What did top solutions that SKIPPED this technique do instead, and why did that work?")
    how_to_reason: str = Field(description="How to reason about this principle. Use whatever format fits: conditionals, trade-off analysis, process steps, questions to ask.")
    code_snippet: str = Field("", description="Optional: Python code pattern showing the decision logic or implementation structure. Leave empty if the principle is about reasoning, not implementation.")


class PlaybookOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Playbook title, e.g. 'Audio Species Identification Playbook'.")
    intro: str = Field(description="2-3 sentence description of the task type, what makes it distinctive, and the core challenge.")
    source_summary: str = Field(description="Source material summary, e.g. '12 top-solution writeups across 4 competitions'.")
    principles: list[Principle] = Field(description="8-15 principles ordered by decision sequence: architectural choices first, then training, then inference/post-processing.")
    final_note: str = Field(description="1-2 sentence closing note on how to use these principles together.")


# ---------- Image / noise stripping ----------

_IMAGE_PATTERN = re.compile(
    r'!\[([^\]]*)\]\([^)]+\)'
    r'|'
    r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*>'
    r'|'
    r'<img[^>]*>'
)


def strip_images(text: str) -> str:
    """Replace markdown / HTML images with their alt text (or drop them).

    Lightweight de-noising applied both to workspace writeups (writer input) and
    to references/ files (embedded by the kb index): images dilute the embedding
    and carry no transferable reasoning.
    """
    def replace(m: re.Match[str]) -> str:
        alt = m.group(1) or m.group(2) or ""
        return f"[{alt}]" if alt else ""
    return _IMAGE_PATTERN.sub(replace, text)


# ---------- Frozen creator prompt (INITIAL_CREATOR_PROMPT) ----------

# The "Handle methodology diversity" bullet asks for conditional principles that
# branch on observable data characteristics.
INITIAL_CREATOR_PROMPT = """\
You are an expert ML practitioner creating a competition playbook for a specific \
task type. You analyze top-solution patterns across multiple Kaggle competitions \
and summarize them into a reasoning framework.

# What is a Playbook?

A playbook is a set of principles that teaches an AI agent how to THINK about \
a competition, not what to DO. Each principle explains:
- The underlying reasoning (why this matters for this task type)
- How to analyze the specific competition to make the right decision
- Code snippets showing the implementation pattern (not the final answer)

# Playbook Design Philosophy

Playbooks are mid-level reasoning frameworks, NOT prescriptive instructions.

- Explain the WHY. For every principle, explain the underlying mechanism. \
An agent that understands WHY a technique works can adapt it to new situations.
- Generalize, don't overfit. Extract the general principle, not the specific \
conclusion. "Check whether your data has a gravity axis before applying vertical \
flip" generalizes. "Don't use vertical flip on seismic data" is overfit to one domain.
- Decision criteria, not decisions. The same principle should lead to DIFFERENT \
decisions for different competitions. Use whatever reasoning format fits: \
conditionals (IF X THEN Y), trade-off analysis, process steps, questions to ask, \
or anti-patterns. Choose the format that best conveys the insight.
- Code snippets show the PATTERN. Include comments indicating what to customize.
- Keep it lean. Every principle should earn its place by teaching something \
non-obvious that affects the outcome.
- Order principles by decision sequence: architectural choices first, then \
training, then inference/post-processing.
- Handle methodology diversity. Competitions within a subcategory may require \
different approaches. When writeups reveal 2+ distinct methodological clusters \
(e.g., EEG vs accelerometer data, ranking vs classification, image vs tabular), \
write CONDITIONAL principles that branch on observable data characteristics: \
"If the data is [type A] (indicators: ...), use [approach A]. If [type B], \
use [approach B]." Never assume all competitions share the same methodology.
- Target 8-15 principles per playbook.\
"""


# ---------- Writer prompt ----------

def build_writer_prompt(item: dict) -> tuple[str, str, str]:
    """Build the (user_prompt, system_prompt, label) for the PlaybookWriter.

    The writer reads the per-task workspace (overview + per-competition writeups)
    via Read/Grep/Glob and emits a PlaybookOutput. The system prompt is the frozen
    INITIAL_CREATOR_PROMPT -- a fixed methodology, no optimizer loop.
    """
    display_name = item["display_name"]
    train_slugs = item["train_slugs"]
    cwd = item["cwd"]

    comp_overviews = "\n".join(
        f"  - {cwd}/competitions/{slug}/overview.md" for slug in train_slugs
    )
    n_comps = len(train_slugs)
    prompt = f"""Create a playbook for the task type: {display_name}

There are {n_comps} competitions in this task type. All files use ABSOLUTE paths.

Overview: {cwd}/overview.md

Competition overviews:
{comp_overviews}

Each competition directory contains:
  - index.md: list of all writeups with titles and sizes
  - writeup_*.md: individual writeup files

Tools available:
  - Read: read a file by absolute path
  - Grep: search across all writeups with path="{cwd}"
  - Glob: list files matching a pattern"""

    return prompt, INITIAL_CREATOR_PROMPT, item["skill_key"]


# ---------- Skill directory writer ----------

PRINCIPLE_TEMPLATE = """\
## Principle {i}: {name}

*Consensus: {consensus}*

{description}

**Why this matters:** {why_it_matters}

**When to use:** {when_to_use}

**When NOT to use:** {when_not_to_use}

**How to reason about it:**
{rules}
{code_block}
---"""


def write_skill_directory(
    pb: dict,
    skill_dir: Path,
    skill_key: str,
    writeups_by_comp: dict[str, list[WriteupRecord]],
) -> None:
    """Write a PlaybookOutput dict as a multi-file skill directory.

    Produces three layers:
      SKILL.md                       routing index (the embedded principle table)
      principles/NN.md               one synthesised principle per file
      references/<slug>-rank-N.md     the raw ranked writeup it was summarized from
    """
    if "principles" not in pb:
        raise AssertionError(f"PlaybookOutput missing 'principles' key: {list(pb.keys())}")
    principles_dir = skill_dir / "principles"
    principles_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for i, p in enumerate(pb["principles"], 1):
        code = p.get("code_snippet", "")
        code_block = f"\n```python\n{code}\n```\n" if code else ""
        content = PRINCIPLE_TEMPLATE.format(
            i=i, rules=p["how_to_reason"], code_block=code_block,
            **{k: v for k, v in p.items() if k not in ("how_to_reason", "code_snippet")},
        )
        (principles_dir / f"{i:02d}.md").write_text(content + "\n")
        index_rows.append(f"| {i} | {p['name']} | {p.get('consensus', '')} | principles/{i:02d}.md |")

    index_table = (
        "| # | Principle | Consensus | File |\n|---|-----------|-----------|------|\n"
        + "\n".join(index_rows)
    )

    # Raw ranked writeups -> references/, the fidelity layer (also embedded; read_reference drills in).
    references_dir = skill_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    reference_names = [
        f"{slug}-rank-{rec.rank}-topic-{rec.topic_id}.md"
        for slug, records in writeups_by_comp.items()
        for rec in records
    ]
    if len(reference_names) != len(set(reference_names)):
        raise AssertionError(f"duplicate reference filenames in {skill_key}")
    for slug, records in writeups_by_comp.items():
        for rec in records:
            if rec.rank is None:
                raise AssertionError(f"writeup for {slug} has no rank: {rec.topic_title!r}")
            text = strip_images(rec.text or "")
            filename = f"{slug}-rank-{rec.rank}-topic-{rec.topic_id}.md"
            (references_dir / filename).write_text(
                f"# {rec.topic_title}\n\nCompetition: {slug}\nRank: #{rec.rank}\nSource: {rec.link}\n\n{text}\n"
            )

    principle_topics = ", ".join(p["name"].lower() for p in pb["principles"][:4])
    display_name = skill_key.replace("-", " ")
    description = (
        f"ML playbook for {display_name} competitions. "
        f"Use when tackling a Kaggle-style competition involving {display_name}. "
        f"Teaches how to reason about {principle_topics}. "
        f"{pb['source_summary']}"
    )

    skill_md = (
        "---\n"
        f"description: >-\n"
        f"  {description}\n"
        "---\n\n"
        f"# {pb['title']}\n\n"
        f"{pb['intro']}\n\n"
        f"**Source material:** {pb['source_summary']}\n\n"
        f"## Principle Index\n\n"
        f"{index_table}\n\n"
        f"## How to Use\n\n"
        f"Use Glob to list principles/*.md, then Read the principles relevant to your task. "
        f"The references/ directory holds the raw ranked writeups these principles were summarized from.\n\n"
        f"{pb['final_note']}\n"
    )
    (skill_dir / "SKILL.md").write_text(skill_md)
