---
title: "TS-Memory: Plug-and-Play Memory for Time Series Foundation Models"
entry_type: paper
source: "https://arxiv.org/pdf/2602.11550"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• It proposes TS-Memory, a plug-and-play lightweight memory adapter that uses Parametric Memory Distillation to adapt frozen Time Series Foundation Models (TSFMs) to downstream domain shifts. • It employs a two-stage training strategy: first constructing an offline kNN teacher to generate privileged supervision signals, then distilling retrieval-induced distributional corrections into a parametric module via confidence-gated supervision. • It enables retrieval-free inference with constant-time complexity, consistently improving forecasting accuracy across benchmarks while maintaining the efficiency of the frozen backbone and avoiding catastrophic forgetting."
---

# TS-Memory: Plug-and-Play Memory for Time Series Foundation Models

**Source**: [https://arxiv.org/pdf/2602.11550](https://arxiv.org/pdf/2602.11550)

**Category**: Framework & Methods

## Description

• It proposes TS-Memory, a plug-and-play lightweight memory adapter that uses Parametric Memory Distillation to adapt frozen Time Series Foundation Models (TSFMs) to downstream domain shifts. • It employs a two-stage training strategy: first constructing an offline kNN teacher to generate privileged supervision signals, then distilling retrieval-induced distributional corrections into a parametric module via confidence-gated supervision. • It enables retrieval-free inference with constant-time complexity, consistently improving forecasting accuracy across benchmarks while maintaining the efficiency of the frozen backbone and avoiding catastrophic forgetting.
