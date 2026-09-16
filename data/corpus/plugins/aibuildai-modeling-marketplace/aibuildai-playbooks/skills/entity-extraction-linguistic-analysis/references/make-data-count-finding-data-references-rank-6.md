# 6th Place Solution

Competition: make-data-count-finding-data-references
Rank: #6
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/6th-place-solution

# **Team Team — 6th Place Solution**

**Data Sources**:
- DCC4: All DOIs, Accession IDs with source "eupmc" (excluding GCA/HGNC/dbGaP due to a lot of FP in the public test split).
- Europe PMC (TextMinedTerms): Accession IDs (excluding GO/RRID/OMIM in all submissions due to a lot of FP in the public test split).

    *Note*: inclusion of the following databases NCT/EudraCT/EBiSC/ChEBI/EFO didn't change the public LB score. So either articles with datasets from these databases were in the private test split, or weren't present at all. We probed some of them and found out that the 1st case holds. In order to increase a diversity of our subs we have excluded them from one of the 2 final subs. As a result, we have equal scores for both of our private subs, so these articles were sparse labeled (i.e. with "Missing" labels).
    
**DOI / Accession IDs retrieval**:
- DOIs:
    Include if the unnormalized DOI appears anywhere in the PDF text, if the DOI is from an eLife article we also look at the XML. If the DOI is from Dryad/Zenodo/f1000 we also check if the DOI without the final version (".v1", etc.), equivalent Zenodo record URL, or base dataset record appears in the PDF text.
- Accession IDs:
    Include if the Accession ID appears anywhere in the PDF text or XML, we only look at the PDF text if the Accession ID is SAMN.


**Heuristic-based classification**:
- DOIs (all rules checked in the following order):
    - Default to Primary;
    - Secondary if there are multiple entries on multiple dates for the DOI and the current article_id is from one of the later dates;
    - Secondary if the DOI only occurs in the reference section of the paper;
    - Secondary if there are more than 4 DOIs found;
    - Secondary if the paper authors don't overlap with the reference that appears right after the DOI, i.e. "(Smith, J., 2019)");
    - Primary if the DOI is from Dryad/Zenodo/f1000;
    - Primary if the paper authors overlap with names listed in the same sentence as the DOI reference, i.e. self-citations;
    - Primary if any specific context phrases (e.g. "we have deposited", "have been submitted" etc.) are in the same sentence as the DOI reference.

- Accession IDs (all rules checked in the following order):
    - Default to Secondary;
    - Primary if SAMN (in one of our final subs), Primary if SAMN and any of the terms SAMN/SAMD/BioSample/BioProject appear near to any of a number of context phrases (in our other final sub);
    - Primary if the Accession ID appears in the same sentence as a number of context phrases.

**Other Notes**:
- If "eLife" is in the article ID, then we skip all DOIs and If "elife" is in the article ID, then we skip all Accession IDs. That's just an observation from the train data.
