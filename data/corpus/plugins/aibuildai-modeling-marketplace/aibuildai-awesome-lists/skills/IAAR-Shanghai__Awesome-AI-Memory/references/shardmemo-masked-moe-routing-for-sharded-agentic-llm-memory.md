---
title: "ShardMemo: Masked MoE Routing for Sharded Agentic LLM Memory"
entry_type: paper
source: "https://arxiv.org/pdf/2601.21545"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• Proposed ShardMemo, a tiered memory architecture: Tier A (working state), Tier B (sharded evidence), and Tier C (versioned skill library). • Enforced a \"scope-before-routing\" strategy in Tier B and modeled shard selection as a Masked MoE routing problem under fixed budgets, using cost-aware gating. • Improved F1 by +6.87 on LoCoMo and HotpotQA compared to cosine similarity routing, while reducing retrieval work and latency by 20.5%."
---

# ShardMemo: Masked MoE Routing for Sharded Agentic LLM Memory

**Source**: [https://arxiv.org/pdf/2601.21545](https://arxiv.org/pdf/2601.21545)

**Category**: Framework & Methods

## Description

• Proposed ShardMemo, a tiered memory architecture: Tier A (working state), Tier B (sharded evidence), and Tier C (versioned skill library). • Enforced a "scope-before-routing" strategy in Tier B and modeled shard selection as a Masked MoE routing problem under fixed budgets, using cost-aware gating. • Improved F1 by +6.87 on LoCoMo and HotpotQA compared to cosine similarity routing, while reducing retrieval work and latency by 20.5%.
