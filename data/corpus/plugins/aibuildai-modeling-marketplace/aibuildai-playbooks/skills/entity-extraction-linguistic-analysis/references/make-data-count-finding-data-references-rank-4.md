# 4th Place Solution

Competition: make-data-count-finding-data-references
Rank: #4
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/4th-place-solution

Thank you Kaggle, MDC, and all participants for making this an exciting competition. We had a great time working together to develop our solution as below!

The competition had two main sub-tasks: (a) finding dataset mentions in scientific papers (candidate extraction) and (b) classifying the citations as primary or secondary (type classification).


# Candidate Extraction

- **DOI Mentions**: retrieved from [Data Citation Corpus v3.0](https://zenodo.org/records/14897662) (DCC). We removed records coming from the `czi` source and a few DOI repositories such as `HEPData`, `Cambridge Crystallographic Data Centre` and `figshare`. The retrieved DOI ids were kept for type classification only if they were present in PDF/XML article text.
- **Accession Mentions**: retrieved from the [PMC TextMinedTerms corpus](https://europepmc.org/pub/databases/pmc/TextMinedTerms/) (PMC), filtering out repositories like `gca`, `go`, `hgnc`, `rrid` and `nct`, which were mostly false positives as per the competition data. Similar to DOI, the retrieved candidates were kept only if they were present in PDF/XML article text.
- **Regex Fallback**: when DCC/PMC corpus did not contain any candidates, we used regex to source possible candidates, followed by an LLM based filtering to remove potential false positives. We handled edge cases where DOI mentions were spread across multiple lines due to PDF to text artifacts using an LLM. This was our original approach before realizing DCC and PMC corpus can offer near perfect recall with high precision.

# Type Classification

We trained LLM classifiers that handled both DOI and Accession subsets without separate models for each. Training a robust model was tricky due to competition dataset being small and noisy. We adopted a few strategies to address these challenges:
- Implemented a tool calling agent that generates synthetic labels using articles from the [Europe PMC open access subset](https://europepmc.org/downloads/openaccess). The agent combined keyword and semantic search tools to gather sufficient context/evidence from an article before making a citation type classification. We warmed up public LLMs (Qwen 2.5 family) using the synthetic dataset before finetuning with the competition examples. We shared a demo notebook [here](https://www.kaggle.com/code/conjuring92/mdc-tool-calling-agent-demo) showing the agent setup and classification trajectory. Agent generated synthetic dataset can be found [here](https://www.kaggle.com/datasets/conjuring92/mdc-type-synthetic-v1/settings).
- For model diversity in ensemble, we also trained a few models by adding pseudo labeled examples (generated with Qwen-2.5-72B) to the competition dataset.
- A few articles contained lots of accession ids (32+), having an outsized impact on model training. So, we limited dataset mentions to a maximum of 24 per article in the fine-tuning datamix. For data augmentation, we masked dataset ids with <mask_i> tokens, where i represents distinct dataset mentions within a given context.
- Averaged checkpoints using Exponential Moving Average (EMA).

### Context for classification
We attempted to maximize relevant information coverage in the LLM context for an informed classification. Specifically, the context included:
- First `n` (= 1400) chars from the article text, as a proxy for structured metadata such as title, authors, published year, and abstract
- Snippets containing the dataset id mention(s)
- Data availability or similar sections
- Text snippets containing other detected DOI mentions
- Table headers and footers in the case of long tables with many accession ids.

# Inference
- To speed up inference, we assumed secondary type for certain accession ids like `cath`, `alphafold`, `cellosaurus`, `chembl`, `dbgap`, `igsr`, `pfam`, `reactome`, and `refseq`. This was informed by stats from the synthetic labels, where they were predominantly secondary.
- We removed accession IDs when a given article had DOI mentions in the data availability (or similar) sections OR when sum of primary DOI probs was greater than 0.8.
- We used a cascaded inference approach: initial type classification with a fine-tuned Qwen-2.5-14B model, then routing uncertain examples (50% of DOI cases + 20% of accession cases) to a fine-tuned Qwen-2.5-32B model. Finally, the top 10% most difficult predictions were handled by a fine-tuned Qwen-2.5-72B. 

### Links:

- [Inference notebook](https://www.kaggle.com/code/conjuring92/mdc-a12-mdc-pipeline/notebook?scriptVersionId=260735888)
- [Training scripts] https://github.com/rbiswasfc/mdc-4th-place-solution

Special shoutouts to @mccocoful for sharing valuable insights during the competition and @rdmpage for the excellent work on improving the dataset labels!
