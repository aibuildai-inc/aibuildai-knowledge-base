# 13th Place Solution

Competition: make-data-count-finding-data-references
Rank: #13
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/13th-place-solution

Thank you Kaggle and MDC for hosting this competition.  Although we are disappointed  to loose gold by one rank again 🫠, we are grateful for the opportunity to learn and grow.


# Problem


The competition required identifying and classifying data citations in scientific literature from Europe PMC. Given PDF/XML texts of research papers, we needed to:


1. **Detect** all dataset citations (DOIs and Accession IDs)
2. **Classify** each citation as:
  - **Primary**: Data generated specifically for the study
  - **Secondary**: Data reused from existing sources


See more here -


https://www.kaggle.com/competitions/make-data-count-finding-data-references/overview) and https://www.kaggle.com/competitions/make-data-count-finding-data-references/data


---


**Example:** Paper [10.1021/jacs.2c06519](https://pubs.acs.org/doi/10.1021/jacs.2c06519)


- **Ground Truth Labels:**
 - `https://doi.org/10.25377/sussex.21184705` (Primary)
 - `https://doi.org/10.25377/sussex.21184705.v1` (Primary)
 - **Actual Text in PDF Contains:**
 - `10.25377/sussex.21184705` (without version)




# External Datasets
We utilized following external data sources:
- **Data Citation Corpus** ([Data Citation Corpus](https://zenodo.org/records/16901115)): Comprehensive DOI citations
- **Europe PMC Text-Mined Terms**: [Accession IDs](https://europepmc.org/pub/databases/pmc/TextMinedTerms/)


---


# Solution Overview


### Two separate pipeline architecture
Two Pipelines because one wasn't enough - :) - one for DOI and one for Accession IDs


### Pipeline 1: DOI Citations
**Input:** GROBID-generated XMLs + Data Citation Corpus


**Process:**
1. Start with known citations from Data Citation Corpus
2. Parse GROBID XMLs for structured citation extraction and consistent format
3. Extract features - Author, Abstract, Title, Chapters etc
4. Qwen2.5 32B LLM (vLLM Inference)




### Pipeline 2: Accession ID Citations 
**Input:** Provided XMLs only (no PDFs) + Europe PMC Text-Mined Terms
All accession citations in the training set can be found in XMLs, this also means if an article doesn't have XML (has only PDF) - it has no accession citations.


**Process:**
1. Extract known accession IDs from [Europe PMC Text-Mined Terms](https://europepmc.org/pub/databases/pmc/TextMinedTerms/)
2. Map PMCID → DOI for paper identification
3. Exclude - GO:, HGNC: , GCA_, 10. -> Assumption - These are citations that EUPMC considers as a dataset but competition doesn't.
4. Qwen2.5 32B LLM (vLLM Inference)
5. Drop duplicates and low confidence predictions




### Merge
Combine outputs from both pipelines - exclude figshare and hepdata citations (Improves public LB by 0.004). Both of these citations appear only in Missing labels in the training set.


## Other Dataset Insights
- If an article has DOI citation, it cannot have Accession Id citation (DOI Articles and Accession ID articles are mutually exclusive) - we missed selecting this 0.742 private submission (gold) because it didn't work on public LB.


- All accession citations in the training set can be found in XMLs, this also means if an article doesn't have XML (has only PDF) - it has no accession citations.


- All IPR citations were Secondary, Competition didn't consider software programs as dataset citations.


- DOI citations were available either in PDF only or XML only or both


## Things that didn't work
- Fixing versioning like in the example above
- Fine tuning classifier
- Direct regex based approach without external data
- Other models like Qwen72B, 14B
- Labeling open access data with GPT OSS

----

Finally, A big Thank you to my fabulous team mates 🌟 @nbroad  and 🌟 @benbla for your hard work and collaboration on this competition !!
