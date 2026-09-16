# 37th place solution

Competition: kaggle-llm-science-exam
Rank: #36
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446455

# 37th Place Solution

First off, I would like to thank my teammate, Simon (@simonveitner) for his collaboration.  He is definitely responsible for most of the work.   Second, I would like to thank all the people who contributed datasets and public notebooks.  It seemed like any idea we had was published publicly within the next week (looking at you @cdeotte, @mgoksu, @mbanei, and @itsuki9180).  This was my first foray into NLP and LLMs, and it has been quite instructive.

Our solution is based on high quality RAG and an ensemble of models.  We pulled results from @simjeg, @jasonzhang2022, @itsuki9180, and @mgoksu for modelling.  We would like to thank @radek1 for his datasets, especially when he included the retrieval context, and @yalickj for his 300 high quality samples.  We also used @itsuki9180 for his optimization of ensemble weights.

## Context Retrieval

Early on, it was clear from the way the train and test datasets were generated we needed some context to do the reverse and then would rely on the language model to do reasoning over the context.  Given that it was an easier task to pick from 5 answers than to generate a question in the first place, we had hope that the smaller models would be able to match the work of the larger LLMs used for generating questions.  As is pointed out by @phillippsinger [here](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446240), there is a limit to the best you can do because GPT-3.5 occasionally gets it wrong, as for example the *triskeles* question pointed out in discussions early in the competition (sorry, I could not find the original discussion now).

Initial work focused on the 60k dataset from @nbroad, but we soon realized it was missing many key articles.  We then moved to the larger full wikipedia set, which would allow us to train with non-science related questions.  Further research showed the full set was missing articles, so we parsed and uploaded our own full set.  After some initial small successes with a custom reranking model and some tests with section sized inputs, @mbanei published the now famous longformer notebook and 270K article set, and we abandoned this work to do modeling building. 

Simon did the hard work of testing different embedding models, and we settled on `bge-base-en-v1-5`, uploaded by @wuwenmin.   Incidentally this did better than the large model.  We then used `LuceneSearcher` in the pyserini package for the lookup.  

## Models and Ensembling

There is not much to add here.  We did look into some multiple choice models besides the DebertaV2ForMultipleChoice, but did not get anywhere.  We settled on an ensemble of 3 different public notebooks using context lookup with the DebertaV2ForMultipleChoice.  We also used one model without any context.

Using the 200 training + 300 high quality samples as our training set for ensembling, we used the ensemble weight optimization introduced by @itsuki9180.  

With the probability table results from the ensemble, we calculated how much time we had left in the 9 hour window, and calculated  the max number of samples we could redo (basically 800 samples per 2 hours left).  We used a Platypus 70B model introduced by @simjeg, after retraining the `lm_head` on a limited number of samples.  *This turned out to be the game changer, as this model performed better on the private board than the public leading to a huge jump in rankings from public leaderboard to private leaderboard!* 

## Breakdown

| Addition | Score Change  |
| --- | --- |
| BM25 context change | +0.03 |
| BM25 base model | + 0.02  |
| DebertaV3 ensemble | + 0.02 |
| Platypus 70B on unsure | + 0.02 |

## What didn't work

1. [Evidence Filter models](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=8618&context=sis_research): We altered the DebertaV2 architecture to include these, but they did not improve the result.  Perhaps it was poor training methodology, we had hoped that allowing some contrast between the answers would improve scores.  
2. Sparse attention longer context models such as `google/bigbird-roberta-large`.  These also did not train appropriately in our experiments, although we did not spend much time on them.
