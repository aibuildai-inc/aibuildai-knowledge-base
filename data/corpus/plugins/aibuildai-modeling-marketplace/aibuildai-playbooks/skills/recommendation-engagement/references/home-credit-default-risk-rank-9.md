# #9 Solution

Competition: home-credit-default-risk
Rank: #9
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64536

Firstly, I would like to say congratulations to all my team mates and to thank them for all their great efforts.

Second, here is a brief write-up of our approach. This entry was born out of my own personal disappointment at suffering in the shakeup on Porto Seguro. I dropped 800+ places. That brought into very clear focus for me the importance of a rigorous CV approach. It also taught me how misleading the PLB can be. This whole team was based around making sure that did not happen again. We tried to do things properly this time. When we reached out to potential new team mates it was explained that they would have to fit into our CV approach. The team benefited from the diverse sets of feature which different members had put together. We tried lots of different types of models including xgb, lgbm, catboost, logistic regression, random forest, extra trees and a few others. We were feeding many base models into our stacker. Although a lot of those were relatively weak models it did seem to give us quite a stable basis for the relationship between the cv score and the plb score. Although we tried to do things properly I didn't imagine the outcome would be as good as it was.

What was good? Having a stable relationship between our cv score and the plb score, seeing them move up together and the absolute numbers very close together.

What was a challenge? Using Scirpus’ features messed with that close relationship between cv and plb. We decided it wasn't worth the risk of chasing the higher cv they offered us because it didn't translate to the plb. At one stage we rebuilt our approach without them. We felt that, from what we understood of the way the features were trained, it likely did not completely fit in with our cv approach. We did have lengthy debates about that though. Maybe we could have found better ways forward on that.

What else could we have done? We did not run rNN's on the time series data. I think this could have added something to our approach.

Thanks to all those who have written about the importance of cv, and, the right way to do it. CPMP and Tilii are two names that come to mind, both in this comp and previous ones, on that topic.

Finally, thanks to all my team mates, great work guys - it has been a pleasure.
