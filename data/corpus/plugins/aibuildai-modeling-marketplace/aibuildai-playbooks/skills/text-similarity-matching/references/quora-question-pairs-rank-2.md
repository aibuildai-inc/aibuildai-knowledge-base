# Overview of 2nd-Place Solution

Competition: quora-question-pairs
Rank: #2
Source: https://www.kaggle.com/c/quora-question-pairs/discussion/34310

First, congratulations to everyone who participated, expecially the winners, and thanks to the organizers. Below is a very brief description of our solution and some personal observations I had regarding this competition. I'll let Stanislav and Dmitry add their own thoughts.

Our final solution was a simple weighted ensemble of 7 models -- 6 lightGBM and one NN. After producing the ensembled predictions, we had a post-processing phase where we recalibrated the probabilities based on some graphicial properties, similar to what Jared described.

Each of us independently created both graphical and NLP features, which we then shared. We had one LGB model that included all useful features (many thousands, including sparse ngram vectors). The others included different subsets of the features and used different LGB architectures. Our single superset model scored about 0.116-0.117 on the LB. 

One thing that definitely helped on the NLP side was to process the text in many different ways -- lowercase and unchanged, punctuation replaced in different ways, stop words included and excluded, stemmed and not stemmed, etc. -- and to build features from all of these different representations. Each representation excelled at picking up different types of text similarities so mixing them all together in a single model was very beneficial.

While this was an interesting and challenging competition, the importance of the question distrbutions and the difference between the train and test sets were rather perplexing. The graphical features (e.g., common neighbors) were very significant and interacted with the NLP features in complex ways that we'll never understand.  If, as others have suggested, these peculiarities are artifacts of the way the questions for the train/test set were selected, and not general properties of the full Quora database, then we have modeled a very artificial data set that has little relation to the real world.

Another interesting aspect of the questions is that a disproportionate number seem to be from/about India. This caused all of the NLP features (e.g., tf and tfidf) to skew towards words relevant to India questions. Again, I'm doubtful that the resulting features will generalize well to questions from/about different regions.

Finally, the labeling was rather noisy. When I looked at the question pairs where our models failed worst, there were many cases where I felt that our model was correct and the labeling was wrong. If I were to make just one recommendation to Quora, it would be that that they use the winning models to identify these questionably-labeled pairs and then review them. Improving the labeling would allow the models to perform much better.
