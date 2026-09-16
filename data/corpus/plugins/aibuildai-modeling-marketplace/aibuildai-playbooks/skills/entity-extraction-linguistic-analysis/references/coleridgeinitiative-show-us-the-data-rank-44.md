# 44th place solution

Competition: coleridgeinitiative-show-us-the-data
Rank: #44
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248386

**Overview**

My solution is a straight-forward Roberta NER model, which scored 0.497 public (#1400~)  -> 0.347 private (#44). I initially wanted to use it as part of a two step pipeline to extract candidates, and then plug another model in charge of choosing whether to keep the dataset or not. 
However, the nature of the public leaderboard and the impossibility to build a reliable CV made this competition extremely frustrating, so I decided to spend my free time on something else :)

**Main points**

- Architecture : Roberta + concatenation of all the output layers + few convolutions + 0/1 output
- Loss : Dice loss, from https://arxiv.org/abs/1911.02855v3
- I built clusters using cooccurrences of datasets in the training data, this allowed me to have a (not so reliable) validation scheme with no overlap, but also to sample data more cleverly during training. 
- I trained a first model using crops of 256 tokens containing at least one dataset. This approach biases the model towards predicting more datasets, and since the metric penalizes FPs more than FNs, I figured out this won't be enough
- So I used this first model to generate ~3000 candidates, that I manually reviewed. This was quite boring, and since it's hard to tell what organizers actually considered to be datasets, I mostly relied on keywords (i.e. survey, study, data, dataset & more) for identification
- A second model was retrained using positive and negative examples from extracted candidates, which is the one I ended up using for my final submission
- I did a bunch of post processing in order to remove false positives. I don't really know how well each of these worked on LB but the first three worked on my CV :
  - Only keep predictions of length > 10
  - Only keep predictions where the maximum token probability is > 0.99
  - Only keep predictions that contains a keyword such as the one above
  - Try to merge similar predictions in order not to predict the same dataset twice
- Few ideas that didn't work and that I didn't already forget about :
  - Replacing datasets from the training data with datasets from external sources 
  - Using the previous Coleridge competition data

Feel free to ask any questions !
Also, my inference code is available [here](https://www.kaggle.com/theoviel/coleridge-ner-inference) if you wish to dig into the details.
