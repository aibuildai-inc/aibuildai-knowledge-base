# 11th place solution

Competition: make-data-count-finding-data-references
Rank: #11
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/11th-place-solution

First and foremost, a huge thank you to the Kaggle organizers for hosting this competition. We learned a great deal, even though luck wasn't entirely on our side this time.
I’d like to acknowledge my teammates, with whom I collaborated to complete this competition:
@tang0310 @phenixalano @patricklo01 @junhaowang0182 

Our solution is organized as follows:
Part 1: DOI Extraction
Our DOI extraction process was a multi-step pipeline:

> 1. Initial Extraction: We began by using regular expressions to extract potential DOI candidates.
2. Building a Filter: To refine our results, we downloaded the complete DataCite metadata corpus to build a comprehensive set of known dataset DOI prefixes.
3. Filtering: We then filtered our extracted DOIs against two sources: the v4 corpus provided in the competition and our custom-built dataset prefix list.
4. LLM Re-extraction: For any potential DOI that matched our dataset prefix list but was not found in the v4 corpus, we took the original text chunk containing it and fed it to an LLM to perform a new extraction. Any DOIs identified by the LLM in this step were then sent back through the filtering process (Step 3) for validation.

Part 2: DOI Classification (Primary vs. Secondary)
We used two distinct methods for classifying DOIs:
> - Main Text DOIs: For DOIs found in the main body of the paper, we used a semantic approach, analyzing the context to determine their role.
- Reference Section DOIs: For DOIs in the references, we based our classification on authorship. A reference DOI was classified as primary if its author list overlapped with the authors of the main paper. To get the paper's authors, we extracted them by having an LLM parse the first 2,000 characters of the text.

Part 3: Accession ID Extraction and Classification
> For Accession IDs, we also started with a regex-based extraction. However, our filtering and classification process was rule-based: we identified specific keywords or patterns in the text chunks surrounding a potential ID. Only if these specific patterns were present did we then use an LLM for final classification.
This is likely where our solution overfit. A pure LLM-based approach without these rules gave us an LB score of 0.80 and a corresponding Public LB score of 0.76. Ultimately, we chose not to trust this simpler, more generalized solution, which is something we now regret.

Once again, thank you to Kaggle and to our entire team for this challenging and rewarding experience.
