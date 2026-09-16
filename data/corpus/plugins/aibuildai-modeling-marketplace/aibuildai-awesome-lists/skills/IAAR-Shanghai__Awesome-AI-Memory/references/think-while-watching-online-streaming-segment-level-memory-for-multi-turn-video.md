---
title: "Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models"
entry_type: paper
source: "https://arxiv.org/pdf/2603.11896"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• Existing streaming multimodal large models typically adopt a serial “perception–generation alternation” paradigm, where text decoding blocks the continuous intake of video, and as long videos progress, the model is prone to forgetting key information from earlier segments. To address this, this paper proposes a novel streaming inference framework that “thinks while watching.” • The framework divides a video into multiple segments and, during system operation, dynamically generates and maintains persistent segment-level memory notes online, which support multi-turn question answering via implicit retrieval. • Constructs a dedicated three-stage streaming chain-of-thought (CoT) dataset—covering single-turn adaptation, multi-turn interaction, and long-range capability training—and pairs it with segment-level streaming causal masks to ensure strict temporal causality."
---

# Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models

**Source**: [https://arxiv.org/pdf/2603.11896](https://arxiv.org/pdf/2603.11896)

**Category**: Framework & Methods

## Description

• Existing streaming multimodal large models typically adopt a serial “perception–generation alternation” paradigm, where text decoding blocks the continuous intake of video, and as long videos progress, the model is prone to forgetting key information from earlier segments. To address this, this paper proposes a novel streaming inference framework that “thinks while watching.” • The framework divides a video into multiple segments and, during system operation, dynamically generates and maintains persistent segment-level memory notes online, which support multi-turn question answering via implicit retrieval. • Constructs a dedicated three-stage streaming chain-of-thought (CoT) dataset—covering single-turn adaptation, multi-turn interaction, and long-range capability training—and pairs it with segment-level streaming causal masks to ensure strict temporal causality.
