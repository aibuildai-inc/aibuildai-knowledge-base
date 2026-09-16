# First Kaggle competition : 10th place public leaderboard; Nth place private leaderboard

Competition: cafa-5-protein-function-prediction
Rank: #4
Source: https://www.kaggle.com/c/cafa-5-protein-function-prediction/discussion/433732

## **Introduction**
Firstly I want to thank the organizers for a very nice competition - having it based on Kaggle opened up submissions to a wider range of people compared to previous iterations when mainly specialist labs showed results so the wider net of people being exposed to CAFA is really gratifying to see. My team worked hard on this problem and although we fell short in the rankings it is humbling to look back at the progress we made. Both of us are molecular biologists with only limited bioinformatics and machine learning experience - so this competition was something of a trial by fire and we learned a lot from taking part.

##**Approach**
We used these datasets in our model: 
1) Prot-T5, ESM2, and Ankh Protein Language Model (PLM) embeddings. We carried out no further modifications or finetuning on the output of PLMs, only conversion to float32 to save memory. 
2) A single binary matrix representing species taxonomy for each protein.
3) Text information obtained by tf-idf of abstract information from academic papers associated with each protein

For GO labels, we separately identified 1500 BPO, 800 CCO, and 800 MFO terms from each ontology for each model to classify by taking the top N labels sorted by IA*frequency. 

Datasets were sorted in alphabetical order to ensure identical ordering across datasets as well as removal of single duplicate row in test data. 

We made a simple dense neural network in keras that takes these as separate inputs before concatenating them into a final layer that attempts to predict every label from a single GO ontology/domain. We used binary cross entropy (BCE) as our loss function and Adam optimizer with lr=0.0003 and a simple lr scheduler. We further used the IA weights as class_weight when calling model.fit(). We added dropout and batch normalization layers to our model as well.

Our model was trained with KFold = 5 across 5 different random splits of the training data before a simple average of predictions was made for each ontology/domain and the results concatenated. 

One interesting thing we observed is that Ridge regression sometimes performed better on PLM embeddings compared to a neural network for some ontologies, which was unexpected and might indicate underfitting for the neural network. 

##**What didn't work**
ProtBert embeddings performed poorly and were not utilized at all. 
Data from STRING did not prove helpful in the final model. 
We attempted to use a separate transformer layer that takes in label embeddings created with Anc2Vec (https://academic.oup.com/bib/article/23/2/bbac003/6523148) but it did not improve model performance.
We attempted to use a hierarchy aware loss function (https://papers.nips.cc/paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf) but it did not perform better than BCE.

##**Scores**
Using merely the PLM embeddings along with taxonomic information we obtained a public leaderboard (PL) score of about 0.55. Including text information increases the PL score to ~0.58.
Ensembling with a public ensemble notebook (https://www.kaggle.com/code/adaluodao/merge-datasets) increases PL score to ~0.62.


##**EDIT**
4th place overall! 
Code used to generate our models can be found here:
https://github.com/zongmingchua/cafa5 
Final ensembling is shown here:
https://www.kaggle.com/code/zmcxjt/merge-datasets
