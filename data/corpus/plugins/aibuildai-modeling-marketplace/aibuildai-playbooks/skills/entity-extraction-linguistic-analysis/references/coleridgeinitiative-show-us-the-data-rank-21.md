# 21st Place Solution: Deep Transformers 1400->21 and the String-Matching repulsion

Competition: coleridgeinitiative-show-us-the-data
Rank: #21
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248360

[](url)First of all, congratulations to everybody and to the new GMs and Masters, as well, and thanks to Kaggle and C.I. for hosting this competition.
This was my first one actually, but I found it very interesting and stimulating.

# **Preprocessing**
This stage appeared to be non-trivial since a huge proportion of training publications came with non-labelled datasets that would have misled the models during the training process. To address this, several labelling-enhancement methods were carried out: a list of all possible labels was created from the occurrences found in the train.csv, but it initially contained a very low number of mentions (some hundreds); dataset names (both titles and mentions) from two external dataset sources were included in the list, and then some string-based operations allowed to further improve this result. Eventually, from less than 200 mentions the list was composed of several thousands, and the labelling of training set got huge improvements. 
Moreover, 5000 new publications taken from the same source as the datasets were included in the training set to get a more consistent amount of training data after the train-val split, that was the only feasible strategy (apparently) due to hard local CV splits; after the labelling-enhancement process, in fact, there was almost no possibility to split publications in completely disjoints sets.
Thus, the split was made on thematic areas, with the Education-related mentions prevailing in the validation set. Still, the huge number of found labels and common associated acronyms led this local validation to be only a rough measure to detect overfitting (which was what I tried to avoid the most) rather than a true overview of model performance.


# **Models**



## NER
The following models are the one that apparently improved Public LB score without overfitting, and this was more or less the rough rationale used for model selection process itself.
All models have 512 as max length; bigger models such as XLNet and SciBERT come with higher regularization (0.5 Dropout), while the smaller ones with proportionally less (0.1 to 0.3); the LR was moderately set at 3e-5. After the validation assessment confirmed each model was not overfitting, the model was trained on the whole set, without any train/val split, and this apparently improved performance on both public and private LBs. Avoiding overfitting is what I focused on the most actually, and to do so I kept a very low number of fine-tuning epochs for transformers and a dropout rate proportional to transformer size. Without a suitable validation strategy, assessing whether this led to underfitting in some proportion is difficult; anyway, I preferred a slight underfit to the whole problem than overfitting training labels, given a huge amount of scores in the public Leaderboard were clearly doing so, and this appeared to be a good choice looking at the shakeups. 
Making predictions on the full text was extremely time-consuming and brought a number of avoidable false positives, hence I made NER models work only on some selected sentences: a list of words was created: 
['scans','cohort','assessment','Assessment','Estimates','Map','map','project','Project','panel','Panel','taken','sample','Sample','Program','program','obtain','include','including','Index','source','images','using','collected','used','Repository','Archive','data','Data','records','dataset','Survey','Study','Atlas','atlas','Supplement','Indicator','report','Questionnaire','questionnaire','Longitudinal','Cohort']
and only sentences which contained at least two of those words in the list were analyzed by the models. This reduced false positives and speeded up the inference process by 10 times, allowing several models to be included in such process. 
All the six models are concatenated together in a weighted voting ensemble with weights based on the single-model performance on the public LB, over a total voting weight of 7, only the predictions with at least 4 votes were propagated to the sequence classification layer; the most important hyperparameters as well as weights are described in the picture above.
-	Distilbert Base Cased
-	Electra Small
-	Distilroberta Base 
-	XLNet Base Cased
-	SciBERT Base Cased
-	SciBERT Base Uncased

## Seq. Classifiers
These models took the predictions of the NER ones and deemed them as ‘They may be datasets’ and ‘They definitely are not datasets’ based on the structure of the prediction itself, rather than contexts. For the training I randomly took pieces of text from the publications (e.g. ‘the education was thus’, ‘good results achieved’) and removed those containing training labels, then included all the true training mentions from the list created in the preprocessing phase, which were almost 20,000. Differently from the NER task, big models performed better here, with an ultimate selection of DeBERTa, RoBERTa and BERT base. This layer appeared to drastically improve performance with NER single-model, and slightly improve it when dealing with the ensemble.

# Postprocessing
After the prediction got at this point, it was assumed to be quite likely to be a true mention, so not much postprocessing was included. Only predictions which were acronyms with length <= 3 (e.g. MHS) were removed blindly: they were likely to be chosen by the NER models if the context suggested it but not likely to be always correctly labelled as ground truths, and the sequence classifiers were not able to distinguish acronyms that referred to a datasets and acronyms that did not. Overall performance increased consistently once again.

# Things that did not work
-	Strict voting (5/7 rather than 4/7); improved public LB but would have performed much worst on private (fortunately it was not selected…)
-	Larger training corpus for sequence classification models
-	Training without acronyms mentions in the corpus in the first place (they were probably closely related with some contexts that models would have missed otherwise)
-	Transformers for longer sequences (Longformer and BigBird): the great majority of sentences had length 512 at most, several ones were filled with padding, hence models for longer sequences were unlikely to work well; a Longformer was anyway included in some of the ensemble trials, but it took too much computing time with small changes in performance
-	Electra base: good for ground truths but made too many false positives
-	SqueezeBERT, Albert, ConvBERT: low performance overall regardless of the hyperparameters
-	Training without external labels, or without external publications, or without both: I had to check whether using external data improved predictive power, and it did
-	Stricter selection of sentence to perform inference on (at least 3 votes rather than 2)

# Conclusions
Starting from a 0.2 public leaderboard with a weak Roberta base, a transition to smaller models led to improvements up to 0.3-0.35. SciBERT was much stronger than them anyway, as it was more able to detect contexts in scientific publications, reaching 0.45 without many problems and 0.49 with all the inference sentence selection and postprocessing tricks. Eventually, the ensemble models broke the 0.5; about Private LB, ensemble models appeared to be slightly better than SciBERT alone.
String matching was avoided as plague, since I though labels were completely disjoint from training and private LB and in such a case relying on it would have led to an illusory overconfidence, but apparently it would have strenghtened results.

There were a couple of submission that would have led to much higher private LB score actually, but as this is my first competition overall I was not very experienced with submission selection.

Maybe I missed something in the description, in case of ambiguities just let me know.
