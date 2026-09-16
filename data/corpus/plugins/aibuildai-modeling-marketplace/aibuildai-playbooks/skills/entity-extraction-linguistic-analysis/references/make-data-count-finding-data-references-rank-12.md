# 12th place solution

Competition: make-data-count-finding-data-references
Rank: #12
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/12th-place-solution

Thanks to everyone involved in organizing this competition, all the participants, and my teammates [@chome0910](https://www.kaggle.com/chome0910) and [@takai380](https://www.kaggle.com/takai380).

# Candidates extraction

- All DOIs are from [MDC Data Citation Corpus v4](https://zenodo.org/records/16901115) using only `datacite` source.
    - DOIs containing `figshare` are filtered.
- All accession IDs are from [EuropePMC](https://europepmc.org/pub/databases/pmc/TextMinedTerms/).
    - IDs containing `:` or `GCA` are filtered.

# Type classification
## DOIs
LLM-based classification leveraging context and metadata. 
- `Qwen2.5-72B-Instruct-AWQ` (zero-shot)
- Input features
    - Context (~3000 chars)
    - Data Citation Corpus based features
        - Number of dataset citations
        - Whether it is the first paper that cited the dataset in the corpus
        - Elapsed days from dataset release date to paper publication date
    - Paper and dataset metadata
        - Title
        - Authors
        - Abstract
- Force a binary choice (A: Primary / B: Secondary) and select the option with the larger logit.

## Accession IDs
Rule-based classification. 
- Except for accession IDs related to `biosample`, classify everything as secondary.
- Biosample accession IDs are classified as secondary if over three years elapsed between id issuance and paper publication; otherwise, as primary.

# Not worked
- LLM-based type classification of accession IDs
