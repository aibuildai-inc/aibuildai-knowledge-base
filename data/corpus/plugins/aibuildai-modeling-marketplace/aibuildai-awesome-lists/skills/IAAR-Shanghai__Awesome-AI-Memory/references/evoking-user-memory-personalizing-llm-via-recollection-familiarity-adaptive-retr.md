---
title: "EVOKING USER MEMORY: PERSONALIZING LLM VIA RECOLLECTION-FAMILIARITY ADAPTIVE RETRIEVAL"
entry_type: paper
source: "https://arxiv.org/pdf/2603.09250"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• It points out that current memory retrieval in personalized large models either feeds in the entire history, causing context overload, or relies only on a single similarity search, leading to shallow understanding. To address this, the research team draws inspiration from the dual-process theory of human memory (“recollection–familiarity”) and proposes an adaptive memory retrieval framework called RF-Mem. • RF-Mem measures “familiarity” by probing the average similarity scores and entropy of the retrieval. When familiarity is high and uncertainty is low, the system takes a fast “familiarity route,” directly returning the single top-K result; when familiarity is low and uncertainty is high, it instead activates a deeper “recollection route.” • When the “recollection route” is triggered, the system clusters the candidate memories and uses an α-mix strategy to update the original query by blending it with the cluster centroids."
---

# EVOKING USER MEMORY: PERSONALIZING LLM VIA RECOLLECTION-FAMILIARITY ADAPTIVE RETRIEVAL

**Source**: [https://arxiv.org/pdf/2603.09250](https://arxiv.org/pdf/2603.09250)

**Category**: Framework & Methods

## Description

• It points out that current memory retrieval in personalized large models either feeds in the entire history, causing context overload, or relies only on a single similarity search, leading to shallow understanding. To address this, the research team draws inspiration from the dual-process theory of human memory (“recollection–familiarity”) and proposes an adaptive memory retrieval framework called RF-Mem. • RF-Mem measures “familiarity” by probing the average similarity scores and entropy of the retrieval. When familiarity is high and uncertainty is low, the system takes a fast “familiarity route,” directly returning the single top-K result; when familiarity is low and uncertainty is high, it instead activates a deeper “recollection route.” • When the “recollection route” is triggered, the system clusters the candidate memories and uses an α-mix strategy to update the original query by blending it with the cluster centroids.
