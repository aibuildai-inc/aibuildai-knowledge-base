# 6th place solution (lucky novices!)

Competition: coleridgeinitiative-show-us-the-data
Rank: #6
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248496

Together with my teammate, @federicomatorra, we are a couple of lucky newcomers!. But I have to say it’s was not luck at all, we invested many hours thinking of and implementing a strategy and finally results were favorable.

We were quite sure that the private test set was a completely different issue, and so it was important to think of a good statistical approach, but with the additional restriction of scarce resources and ignorance of the latest developments in NLP.

So, in summary, although we evaluated many ideas, we used spaCy 2 (v2.spacy.io) for text classification in 2 steps, first for separating sentences (ie., sentences that mention a dataset vs no dataset at all) and second for identifying datasets versus ORGs, LOCs, etc., and many other acronyms and names that were only (at least in our view) false positives.

To train the text classification models, we used the datasets names provided for training and managed to create a list of additional dataset names together with corresponding acronyms (positive and negatives) using an abbreviation detector (https://allenai.github.io/scispacy/) and matching for words such as dataset, databank, survey, etc.

We think if was simple enough to be trained and tested fast and flexible enough for adapting to unknown publications.

The notebook that was scored in 6th place it was a hybrid of text matching and the approach mentioned above but as I said, in our opinion, the public and private sets of documents were two completely different universes, so it was not possible to finish in the first ten using only string matching.

We want to thank the organizers and congratulate the immense number of participants and in particular, those who finished in the medallist positions. Also, recognize the work done by Coleridge to set up the training and testing sets.
