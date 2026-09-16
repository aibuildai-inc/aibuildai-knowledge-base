# 36th Place Solution (and "could have been" 22nd solution)

Competition: coleridgeinitiative-show-us-the-data
Rank: #36
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248320

Congratulation to all the medal winners and everyone else who learned new things through this competition. Thanks to organizers and Kaggle for hosting the competition. 

We secured 36th using the solution we selected, but  had a submission which would have given us 22nd had we selected it (no regrets, all part of the game). Here is an overview and some high level details about our solution. 

**Overview**
1. NER with Roberta backbone using new and old competition data: ~ 0.5x / 0.3x. One modification we had done was - we trained the NER with dual objective of predicting the probability of a dataset in a sentence along with NER labels (beginning / middle / end). This seemed to help compared to just using single NER objective. 
2. Ensemble with similar pipeline using different backbones like Scibert, Conll, Biobert: ~0.56 / ~0.414 (the solution we didn't select)
3. Ensemble + some post processing that seemed work on Local and public LB but not on Private LB: ~0.56x / 0.368 (the solution we selected - 1/2)
4. Ensemble + post processing + String Match using dataset lists: 0.640 / 0.363 (the solution we selected - 2/2)

**Details**
- The training / scoring was done at a sentence level, extracted using Spacy Sentencizer 
- Scored only ~10% of the total sentences in test publications which were extracted using regular expression to be able to run Ensemble of 4 models in allotted time for the notebooks. This led to no significant deterioration in accuracy on validation / LB. (thanks to my teammate @soloway for this)
- There were some datasets with a ton of publications in new / old training data, which was causing local CV to be unstable. Used only 5-10 Publications per dataset for training / validation to address this.
- String Matching seemed to work on Public LB, but hurt us slightly on the Private. We knew the risk, so only 1 of our 2 selected solutions had string matching.
