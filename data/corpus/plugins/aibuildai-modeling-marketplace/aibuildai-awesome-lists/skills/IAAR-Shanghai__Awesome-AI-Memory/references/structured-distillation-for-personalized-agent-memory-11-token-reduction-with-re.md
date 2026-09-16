---
title: "Structured Distillation for Personalized Agent Memory: 11× Token Reduction with Retrieval Preservation"
entry_type: paper
source: "https://arxiv.org/pdf/2603.13017"
upstream_list: "IAAR-Shanghai/Awesome-AI-Memory"
category: "Framework & Methods"
description: "• It proposes extracting each conversational interaction into a structured composite object that includes core content, specific context, topic classification, and related files. This approach follows a “surviving vocabulary” principle, avoiding arbitrary rewriting of technical terms, and successfully compresses the average number of tokens per interaction from 371 to 38, achieving an 11× compression efficiency. • In experiments covering 107 retrieval configurations, it was found that the best pure distilled-text setup can retain 96% of the retrieval quality (MRR) of the original verbatim text. Furthermore, retrieval performance is highly mechanism-dependent: vector search shows virtually no noticeable degradation even under 11× compression, whereas keyword search (BM25) degrades significantly. • The agent carries only the compressed distilled text within the context as a “routing index” for efficient retrieval, while the original full conversation text is stored locally and is only brought up for display when the user needs to inspect it in detail."
---

# Structured Distillation for Personalized Agent Memory: 11× Token Reduction with Retrieval Preservation

**Source**: [https://arxiv.org/pdf/2603.13017](https://arxiv.org/pdf/2603.13017)

**Category**: Framework & Methods

## Description

• It proposes extracting each conversational interaction into a structured composite object that includes core content, specific context, topic classification, and related files. This approach follows a “surviving vocabulary” principle, avoiding arbitrary rewriting of technical terms, and successfully compresses the average number of tokens per interaction from 371 to 38, achieving an 11× compression efficiency. • In experiments covering 107 retrieval configurations, it was found that the best pure distilled-text setup can retain 96% of the retrieval quality (MRR) of the original verbatim text. Furthermore, retrieval performance is highly mechanism-dependent: vector search shows virtually no noticeable degradation even under 11× compression, whereas keyword search (BM25) degrades significantly. • The agent carries only the compressed distilled text within the context as a “routing index” for efficient retrieval, while the original full conversation text is stored locally and is only brought up for display when the user needs to inspect it in detail.
