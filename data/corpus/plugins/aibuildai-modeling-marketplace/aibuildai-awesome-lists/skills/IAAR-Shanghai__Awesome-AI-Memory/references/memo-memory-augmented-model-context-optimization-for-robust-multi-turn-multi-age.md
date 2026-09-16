---
title: "MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games"
entry_type: paper
source: "https://arxiv.org/pdf/2603.09022"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• In long-horizon multi-agent games, early mistakes are easily amplified, and fixed prompts can lead to rigid strategies and highly variable evaluation results. To address this, this paper proposes MEMO, a self-play framework that requires no updates to model weights. • MEMO cleverly decouples and combines the mechanisms of “retention” and “exploration.” It builds a persistent memory bank that uses CRUD (create, read, update, delete) operations to extract structured strategic insights from self-play trajectories and injects them as prior knowledge for subsequent reasoning; meanwhile, it employs tournament-style prompt evolution based on TrueSkill ratings and a prioritized experience replay mechanism to efficiently explore strategies and revisit critical decision states. • In five text-based game benchmarks, MEMO demonstrates remarkable learning efficiency: with just 2,000 self-play episodes, it boosts GPT-4o-mini’s average win rate from 25.1% to 49.5%, while simultaneously causing a substantial reduction in the variance of its performance."
---

# MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games

**Source**: [https://arxiv.org/pdf/2603.09022](https://arxiv.org/pdf/2603.09022)

**Category**: Framework & Methods

## Description

• In long-horizon multi-agent games, early mistakes are easily amplified, and fixed prompts can lead to rigid strategies and highly variable evaluation results. To address this, this paper proposes MEMO, a self-play framework that requires no updates to model weights. • MEMO cleverly decouples and combines the mechanisms of “retention” and “exploration.” It builds a persistent memory bank that uses CRUD (create, read, update, delete) operations to extract structured strategic insights from self-play trajectories and injects them as prior knowledge for subsequent reasoning; meanwhile, it employs tournament-style prompt evolution based on TrueSkill ratings and a prioritized experience replay mechanism to efficiently explore strategies and revisit critical decision states. • In five text-based game benchmarks, MEMO demonstrates remarkable learning efficiency: with just 2,000 self-play episodes, it boosts GPT-4o-mini’s average win rate from 25.1% to 49.5%, while simultaneously causing a substantial reduction in the variance of its performance.
