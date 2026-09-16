---
title: "Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning"
entry_type: paper
source: "https://aclanthology.org/2025.findings-acl.1234.pdf"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• MemQ is proposed to decouple reasoning (natural-language steps) from query generation/execution (SPARQL): the LLM produces a clear reasoning plan, while the actual query is obtained via memory retrieval + rule-based reconstruction, reducing errors and hallucinations caused by entangling tool calls with reasoning. • During training, gold SPARQL queries are decomposed into query fragments, and a natural-language explanation is generated for each fragment to build a query memory bank of (explanation → fragment) pairs. At inference time, the LLM generates step-by-step plans; reconstruction uses semantic retrieval (Sentence-BERT) to fetch an adaptive Top-N set of fragments, then assembles them with rules and fills entity slots to produce the final executable query. • Experiments on WebQSP and CWQ use Hits@1 and F1, where MemQ achieves the best overall performance. Additional analyses with structural consistency / edge hit rate show the reconstructed queries are closer to the gold graphs, and ablation studies confirm that the main gains come from the memory bank + decoupling design."
---

# Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning

**Source**: [https://aclanthology.org/2025.findings-acl.1234.pdf](https://aclanthology.org/2025.findings-acl.1234.pdf)

**Category**: Framework & Methods

## Description

• MemQ is proposed to decouple reasoning (natural-language steps) from query generation/execution (SPARQL): the LLM produces a clear reasoning plan, while the actual query is obtained via memory retrieval + rule-based reconstruction, reducing errors and hallucinations caused by entangling tool calls with reasoning. • During training, gold SPARQL queries are decomposed into query fragments, and a natural-language explanation is generated for each fragment to build a query memory bank of (explanation → fragment) pairs. At inference time, the LLM generates step-by-step plans; reconstruction uses semantic retrieval (Sentence-BERT) to fetch an adaptive Top-N set of fragments, then assembles them with rules and fills entity slots to produce the final executable query. • Experiments on WebQSP and CWQ use Hits@1 and F1, where MemQ achieves the best overall performance. Additional analyses with structural consistency / edge hit rate show the reconstructed queries are closer to the gold graphs, and ablation studies confirm that the main gains come from the memory bank + decoupling design.
