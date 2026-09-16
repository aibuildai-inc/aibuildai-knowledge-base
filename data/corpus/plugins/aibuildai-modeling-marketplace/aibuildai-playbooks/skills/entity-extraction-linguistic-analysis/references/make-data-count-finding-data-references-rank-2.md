# 2nd Place Solution [Updated]

Competition: make-data-count-finding-data-references
Rank: #2
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/2nd-place-solution

Our solution for the competition could be broken down in 2 stages.

* **Stage 1.** Find the datasets in paper (Either **Dataset DOIs** or **Accession IDs** for select Bio repositories)
* **Stage 2.** Classifying whether a given data source is `Primary` or `Secondary`. 

Further for each stage, we developed separate solutions for DOIs and Accession IDs. (For classification, single model approaches didn't work for me)

Analysis of dataset references in provided ground truth labels hinted at some kind of automated process for fetching datasets. E.g. In many cases, some Accession numbers from the same table and repository were picked while others were not. The speculation here was that some kind of NER model is being used, thresholding upon which leaves out some relevant Accession numbers. Similary, for DOI versions for dryad datasets were included but not present in paper.
And so the hunt for finding data sources/code began.

### Stage 1 - DOIs

There are 2 relevant data sources here:
1. Data Citation Corpus Data File (https://doi.org/10.5281/zenodo.13376773) - v2 version gave us best validation score.
2. DataCite Public Data File 2024 (https://datafiles.datacite.org/)

As mentioned in Data Citation Corpus description, it uses events data from datacite to build the corpus. As such, datacite public data file can be thought as superset of data citation corpus. Joining data citation file with training article_ids, gives **100% recall for DOIs.** . 
To improve precision, based on observations from training data following preprocessing was applied.
1.  Dataset from these repositories were filtered out  -> `figshare`, `cambridge crystallographic data centre`, `hepdata` as they were tagged as `Missing` in training data.
2. Dataset ids that were not present in either PDF or XML test were dropped
     * Regex - re.compile("".join([ch + r"[\s]*" for ch in base_dataset_id.replace("https://doi.org/", "")]).strip(), re.IGNORECASE)
     * Stripped version tags (.v1, .v2 .v3 etc) for dryad repo datasets

With just these steps, we get 0.964 validation score (without taking `type` in account) on training data.
[Stage 1 DOI score]

### Stage 1 - Accession IDs

The accession numbers come from: https://europepmc.org/pub/databases/pmc/TextMinedTerms as can be seen in branch here: https://github.com/Make-Data-Count-Community/corpus-data-file/tree/ks-data-dump-v4-eumpc. I think there is PR for it to merge to main branch.

Once again, joining the data downloaded from here to train data gives **100% recall**.  Based on observations from training data it seemed, some repositories were being excluded from training labels -> `hgnc`, `gca`, `go`, `rrid`. Since hgnc, gca, go all had `:` in their identifiers, remaining repos with `:` were also dropped ( "chebi",  "orpha",  "rhea",  "efo"). Further as per PR, ( "ebisc",  "hipsci",  "omim", "eudract") were dropped as well. I considered dropping `NCT` as per PR but didn't do it as it was not part of public set. It would have led to further +0.003 on private set.

LB probing experiments suggested they had no impact on public test score but dropping them improves private by 0.001. 
Ofcourse, DOI from eupmc data were also dropped.

Finally, we dropped accession numbers from all the articles where a DOI candidate was present as was evident in training data.

This gives us 29 False positives of which 25 were `SAMN**` from `10.1038_s41597-019-0101-y` . Carefully looking at XML it seems, they were part of online-only table. So, I added a rule for that (this gives 0.003 boost on public test and 0.004 on private as well)
[Online-only SAMN]

This gives 0.98 F1 (not accounting for `type`) for Accession number from Stage 1
[Accession Stage 1 score]

### Stage 2 heuristics

On top Stage 1 rules, if we add following rules we get 0.869 on public and 0.739 on private  **(No model required to get a gold medal!)**

1. Accession: SAMN and EMDB -> Primary
2. DOI: If dataset is found in multiple papers as per datacite corpus, tag first as Primary rest as Secondary after sorting by publicationDate
3. DOI: If article_id  `isSupplementTo` dataset_id as per datacite public data file -> Primary
4. DOI: If more than 4 occurence of same repo (first 4 letters)  or more than 4 DOI mentioned around the dataset in article -> Secondary


### Stage 2 - DOI

### Mohsin - DOI classification

Most time was spent on creating context for classification (<2048 tokens) . I mapped dataset authors and abstract from datacite public data file. Then I collate context by extracting following parts of paper:
1. context around first match of dataset id regex
2. Whole paper was split in overlapping chunks of 1024 chars, then I used BM25 similarity score w.r.t dataset abstract to get top 3 chunks.

Model -> MedGemma-4B lora 

Adding this model and dropping other DOI rules except `isSupplementTo` gives 0.880/0.784

#### Mohsin - Accession classification

Context engineering: If match in Table `tag` -> get table row and caption -> use table id to find first reference in paper and get additional context -> add data availability section -> fallback to search on PDF text. 

Although OOF score was 0.91, the LB score was significantly lower. This suggested some bug or massive overfitting. Could not work on it further.


### General Commentary (Mohsin):
1. Initially I spent too much time on NER model, only to find it was not required.
2. Data was too less to train a stable classification model. I started downloading open access corpus from Pubmed to gather more data, but work on the idea.


### Stage 2 Classification Models - Nikhil

### Problem Statement

The primary challenge was **dataset size limitations** causing severe model instability:

- **Unstable thresholds** across training runs
- **Inconsistent convergence** in cross-validation:
  - Fold 1: Converged at epoch 2
  - Fold 2: Failed to converge after 10 epochs
  - Remaining folds showed similar inconsistency

## Data Sources

**Training Labels:**
- Competition Data
- RDMPage Data (including labels absent from Zenodo DataCite corpus)

## Model Architecture

### Dual-Model Approach
Two specialized models were developed:
1. **Accession ID Model** - For identifying accession identifiers
2. **DOI Model** - For identifying Digital Object Identifiers

### Base Model
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`

## Stabilization Techniques

### 1. Token Replacement Strategy
- Replace all non-target dataset IDs with: `"other dataset id"`
- Transform target dataset ID to: `"Prediction {DATASET_ID}"`
- **Purpose:** Focus model attention on specific ID being classified

### 2. Hyperparameter Configuration
- **Batch Size:** 256 (unusually large)
- **Impact:** Significantly improved training stability

### 3. Threshold Optimization
- Run models with multiple random seeds
- Average out-of-fold (OOF) predictions
- Select optimal threshold from averaged results

## Context Window Configuration

### Accession IDs
- **Primary Context:** 300 characters (left and right of ID)
- **Additional Features:**
  - Abstract text
  - Supplementary information
  - References section

### DOIs
- **Primary Context:** 150 characters (left and right of ID)
- **Rationale:** DOIs prone to overfitting with larger contexts
- **Result:** Improved generalization with reduced window

## DOI-Specific Enhancements

### 1. Mention Frequency Feature
- **Observation:** Multiple DOI mentions correlate with "Primary" classification
- **Implementation:** Add mention count as special token
- **Example:** `"More than 3 mentions"`

### 2. Repository Signal
- **Insight:** Repository source provides classification signal
- **Repositories:** Zenodo, Dryad, etc.
- **Implementation:** Prepend repository name to context

### 3. Context Construction Example
```
DOI Context = [Mention Token] + [Repository Token] + [Original Context]

Example:
"More than 3 mentions" + "Zenodo Repository" + [150 chars context]
```

## Key Success Factors

1. **Model Separation:** Treating accession IDs and DOIs as distinct problems
2. **Context Optimization:** Different window sizes for different ID types
3. **Feature Engineering:** Leveraging metadata (mentions, repository)
4. **Stability Focus:** Large batch size and multi-seed averaging
5. **Token Strategy:** Replacing irrelevant IDs to reduce noise

## Results

- Stabilized model performance across folds
- Consistent convergence behavior
- Improved threshold reliability through averaging
- Better generalization on DOI classification with reduced context


Thanks to competition host, organizers and all of Kaggle community! As always it is amazing to take part in competitions.
