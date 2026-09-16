# 2nd place solution overview

Competition: coleridgeinitiative-show-us-the-data
Rank: #2
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248296

Thanks to Coleridge & Kaggle for organizing this competition, and congrats to the other prize winners and medallists. This was a challenging competition in contrast to the standard Kaggle NLP classification/regression task, and I hope Coleridge will consider sponsoring future competitions in this vein. I suspect a future iteration will have the kinks shaken out :)

## TL;DR: Search + Classify

1. Search for named entities using the Schwartz-Hearst algorithm
2. Filter candidates using a fine-tuned roberta-base binary classifier
3. Threshold and propagate candidates

CV strategy: It should be clear that there was no way to perform clean local validation. I used public LB while aware of its flaws. 

I spent the first month of the competition working on an end-to-end Transformer solution with minimal text processing. However, it became clear that a tweaked Transformer approach was problematic - they’re a) unsuited for the long document sequences we’re dealing with, even with the specialized variants designed for such texts, and b) ended up requiring heavy pre/post-processing to perform well. My processing heuristics effectively turned into a model parallel to the main Transformer model. I began thinking about how to simplify this - 

#### Insight #1: No context needed
Any attempt to inject semantic context into my models seemed to worsen public LB accuracy. I suspected the annotation process was noisy and decided to focus on IDing datasets purely by title. 

#### Insight #2: Search for LONG-FORM (ACRONYM) 
Data sets in academic papers often have a regular structure - mixed cap words followed by an acronym. An example is “Baltimore Longitudinal Study of Aging (BLSA)”. I needed something lighter than a Transformer to search for them efficiently - the answer was the [Schwartz-Hearst algorithm](https://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf). Schwartz-Hearst is a non-learning string search algorithm used to extract strings from text in the form of “LONG-FORM (ACRONYM)” - it’s basically a more involved regexp search. A bonus from using this algorithm is that I could also use the acronym form of a given dataset for additional points. 

#### Insight #3: No scraped data needed
The third insight (hypothesis?) was that scraped data was unnecessary. I did not use any externally derived list of datasets or scraped data - my sense was that 1) positive examples of datasets weren’t as useful as negative examples, and 2) it wasn’t even clear whether these lists hewed to the dataset definition that annotators were using. 

#### Putting it all together
Schwartz-Hearst worked well at identifying named entities but there’d be many flagged candidates that weren’t data sets (e.g., organization names, scientific paraphernalia) so I needed a binary classifier to filter candidate strings. I hand-annotated (using my own judgment of what a dataset was) a miniature training corpus of positive and negative examples based on examples that Schwartz-Hearst IDed from training data and fine-tuned a roberta-base binary classifier to filter these named entities using the corpus. I’ll re-emphasize that this corpus was built wholly from the training data without any scraped data. 

The last piece of the puzzle was dealing with the fact that not all papers referred to datasets using the full “LONG-NAME (ACRONYM)” format. Many referenced only the LONG-FORM. A simple solution was to generate a dynamic table of datasets and search for LONG-FORM in documents. I noticed that thresholding this search on a minimum document frequency improved public LB. The intuition is that in any given document corpus, a dataset is probably referenced by multiple documents so the more frequently they’re referenced, the more likely they’re a hit and not a spurious find. 

This search + classify approach dramatically reduced inference time since I did not have to pass entire documents through a Transformer model. Inference on the test set took ~10 minutes vs >>1hr using end-to-end Transformer QA/NER models. 

## End-to-end dummy example:
1. The LONG-FORM (ACRONYM) string of “Singapore Longitudinal Study on Diabetes (SLSD)” is found in 64 documents

2. “Singapore Longitudinal Study on Diabetes” is classified by the Transformer binary classifier to be a dataset with probability 0.99, above the prediction threshold of 0.9 

3. The document frequency of 64 exceeds the search threshold frequency of 50 - a search of LONG-NAME is conducted across all documents and another 200 documents are found to contain the label

4. For the combined 264 documents, the LONG-FORM is added as a prediction. Additionally, if any of the 264 documents contains the string ‘ SLSD ‘ (checked against the raw document texts), an additional prediction of the acronym is added

Note: this is a summary of the model logic, there are other rules applied that I’ve elided.

Code [here](https://github.com/leecming/kaggle_coleridge)
