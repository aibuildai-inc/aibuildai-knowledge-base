---
title: "Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory"
entry_type: paper
source: "https://arxiv.org/pdf/2504.07952"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• The paper proposes Dynamic Cheatsheet (DC)—a continuously updated “sticky-note” external memory for black-box LLMs at inference time. It distills verified solving patterns and reuses them across problems, enabling test-time learning without training. • DC consists of a Generator (Gen) and a Memory Curator (Cur): Gen produces an answer using the current memory, and Cur then refines/filters/compresses the information. A retrieval-based variant selects the most relevant past examples and solutions by similarity to assist generation, while preventing memory bloat. • DC is evaluated across multiple tasks and models (e.g., AIME, GPQA-Diamond, Game of 24, MMLU-Pro; GPT-4o, Claude 3.5 Sonnet, etc.) using metrics such as Soft Match and Functionally Correct. Results show substantial gains; for example, the jump on Game of 24 is largely driven by reusable Python solver code being repeatedly “written and reused” in memory."
---

# Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory

**Source**: [https://arxiv.org/pdf/2504.07952](https://arxiv.org/pdf/2504.07952)

**Category**: Framework & Methods

## Description

• The paper proposes Dynamic Cheatsheet (DC)—a continuously updated “sticky-note” external memory for black-box LLMs at inference time. It distills verified solving patterns and reuses them across problems, enabling test-time learning without training. • DC consists of a Generator (Gen) and a Memory Curator (Cur): Gen produces an answer using the current memory, and Cur then refines/filters/compresses the information. A retrieval-based variant selects the most relevant past examples and solutions by similarity to assist generation, while preventing memory bloat. • DC is evaluated across multiple tasks and models (e.g., AIME, GPQA-Diamond, Game of 24, MMLU-Pro; GPT-4o, Claude 3.5 Sonnet, etc.) using metrics such as Soft Match and Functionally Correct. Results show substantial gains; for example, the jump on Game of 24 is largely driven by reusable Python solver code being repeatedly “written and reused” in memory.
