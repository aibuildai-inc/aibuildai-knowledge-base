---
title: "ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries"
entry_type: paper
source: "https://arxiv.org/pdf/2603.12942"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• ReMem-VLA introduces two sets of learnable recurrent memory queries: frame-level queries, which are updated frame by frame to capture short-term memory, and chunk-level queries, which are updated over longer temporal spans to stably maintain long-term memory. • A visual prediction head is added, introducing past observation prediction as an auxiliary training objective, which forces the model to recall visual details by reconstructing historical RGB frames. To address the batching challenge of recurrent training on variable-length sequences, a slot-based streaming training paradigm is proposed, which preserves temporal continuity while preventing state leakage across episodes. • To overcome the bottleneck of traditional truncated backpropagation through time on long-sequence optimization, the model creatively adopts a gradient-free recurrent update path that combines a frozen VLM with a fixed exponential moving average, allowing the queries to focus solely on learning “what task-relevant information to extract” rather than “how to propagate it.”"
---

# ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries

**Source**: [https://arxiv.org/pdf/2603.12942](https://arxiv.org/pdf/2603.12942)

**Category**: Framework & Methods

## Description

• ReMem-VLA introduces two sets of learnable recurrent memory queries: frame-level queries, which are updated frame by frame to capture short-term memory, and chunk-level queries, which are updated over longer temporal spans to stably maintain long-term memory. • A visual prediction head is added, introducing past observation prediction as an auxiliary training objective, which forces the model to recall visual details by reconstructing historical RGB frames. To address the batching challenge of recurrent training on variable-length sequences, a slot-based streaming training paradigm is proposed, which preserves temporal continuity while preventing state leakage across episodes. • To overcome the bottleneck of traditional truncated backpropagation through time on long-sequence optimization, the model creatively adopts a gradient-free recurrent update path that combines a frozen VLM with a fixed exponential moving average, allowing the queries to focus solely on learning “what task-relevant information to extract” rather than “how to propagate it.”
