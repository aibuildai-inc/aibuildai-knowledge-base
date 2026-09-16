# 4th - This is Great! - Shared Solution

Competition: jigsaw-toxic-severity-rating
Rank: #4
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306084

This is Great!

I just got the results and I'm super happy.

Thanks to everyone who participated in the competition, to the host who proposed the challenge and to kaggle for making it easy for me to use their GPU resources.

I won't have much time today until the end of the day but I'll quickly share my solution here, which was a final combo of roberta base (tito cv 0.6983) + roberta large (tito cv 0.7014) with detoxify (host cv 0.7036).

Thanks to everyone once again!

UPDATED EDIT: 

##### Shared Code

I just shared the solution in the following training and inference notebooks,

Training:
- https://www.kaggle.com/coreacasa/jigsaw4-luke-base-training-tito-cv-strategy/notebook
- https://www.kaggle.com/coreacasa/jigsaw4-luke-large-training-tito-cv-strategy/notebook

Inference:
- https://www.kaggle.com/coreacasa/jigsaw4-merging-luke-roberta-with-detoxify/notebook?scriptVersionId=87076833

Acknowledgment and Credits to the authors in the comments section that I repeat here (I'm sorry if I didn't include someone): 

- @yasufuminakama and his notebooks https://www.kaggle.com/yasufuminakama/jigsaw4-luke-base-starter-train and https://www.kaggle.com/yasufuminakama/jigsaw4-luke-base-starter-sub

- @its7171 for their notebook https://www.kaggle.com/its7171/jigsaw-cv-strategy

- @steubk and his dataset https://www.kaggle.com/steubk/detoxify-sourcemodels

- @pdnartreb and his thread https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/304441 

- (from previous discussion) @sorenj and his notebooks https://www.kaggle.com/sorenj/perspectiveapi-tuning and https://www.kaggle.com/sorenj/scoring-ruddit-comments

UPDATED EDIT2: 
##### Overview

Jigsaw Rate has continued  the motivation of the 2018-2019-2020 challenges. Beyond the new metric (task) that evaluated the final performance of the solutions. I would add several peculiarities in this new challenge with respect to the previous ones, one of them (1) it is that the host did not provide a specific training data set, another (2) is that the host did provide a validation data set and finally (3) that we had a scarce reference of the public board scores representing only 5% of the final result.

Once we advanced in the development of the competition and derived from (1) and (2) above, I thought that as important as the diversity of models, perhaps this time the diversity of training datasets could play an even stronger role. Likewise, derived from (2) and (3) and from the lack of gratitude that any submission to the public scoreboard was, I decided in the last week to give all priority to validation scores regardless of whether the public board confirmed improvements in performance or not.

###### Diversity of Datasets:

-From the previous Jigsaw competitions I worked with,
(2018 Jigasw1) - Toxic Comment Classification Challenge
(2019 Jigasw2) - Jigsaw Unintended Bias in Toxicity Classification
(2020 Jigsaw3) - Multilingual Toxic Comment Classification

-From this competition,
(2021 Jigasw4) - 'validation_data.csv'.

Although they are not part of the solution I also tried, without success, with external data. I'm referring to ruddit and hate_speech data, thanks @andre112 and thanks @rajkumarl. Specifically, I extracted samples of paired texts for each one of them given the enormous final extension that the combination of all the crosses supposed. Perhaps I made a mistake in this and I should have sampled the texts to work with the resulting paired texts in a number appropriate to my available resources.

###### Validation Strategies:

As I indicated in a brief reference at the beginning of this discussion, the final solution uses 2 validations,

1) tito-cv (original author's nickname) applied to validation_data.csv of this competition. This strategy extends a 5-fold cross-validation by non-overlapping Groups of 4143 leak-free (least toxic-most toxic) text pairs (14251 different texts).

2) host-cv is not cross-validation (sorry if I am confusing the vagueness of 'cv') but simply an extraction of text pairs (least toxic-most toxic) from the validation_data.csv dataset (same as above) that match with the dataset comments_to_score.csv, in total 8334 pairs found.

###### Models and Trainings:

Taking into account the datasets that are ingested, we can divide this part into two sections,

1) Two Luke-Roberta models were fine-tuned on the validation_data.csv of this competition, please visit https://github.com/studio-ousia/luke.

I forked Y.Nakama's notebook Luke-base (Roberta-Base) by replacing the original validation pipeline (groupkfold over worker_id) with the aforementioned tito-cv. Saving the best of 15 epochs for each fold, I kept the rest of the parameters untouched (among others... max_len= 64, batch_size= 64, AdamW, lr's= 1e-5, 1/2 cycle cosine) as well as the loss function Margin-Ranking-Loss with margin= 0.5.

I trained a second model this time changing the base model to Roberta-Large, only 5 epochs keeping everything else the same. On these 2 models, a result of linear optimized blending was tito-cv 0.705 (I share this new notebook [here](https://www.kaggle.com/coreacasa/jigsaw4-luke-base-large-tito-cv-merging-optimize)) although I did not use this blend in the end.

2) About the training data of the Jigsaw1 (original), Jigsaw2 (unbiased) and Jigsaw3 (multilingual) competitions I used (did not train) already trained models to predict toxic comments about those 3 competitions, please visit https://github.com/unitaryai/detoxify.

| Model name | Transformer type | Data from |
| -- | -- | -- | 
| original | bert-base-uncased | Toxic Comment Classification Challenge |
| unbiased | roberta-base | Unintended Bias in Toxicity Classification |
| multilingual | xlm-roberta-base | Multilingual Toxic Comment Classification |

With a high performance for single models (not ensembles) in the leaderboards of each challenge and evaluating their individual performance on host validation, I considered it a great match to include these predictions in the final solution.

As for the labels on which these models return a probability value, we would have,

| Model name | Labels |
| -- | -- | 
| original | {'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'} |
| unbiased | {'toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit'} |
| multilingual | {'toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit'} |

The next step was to find a method to translate these multivariable outputs into a single output that measures toxicity, beyond the 'toxicity' label itself, and that would work with the 'Average Agreement with Annotators' in this competition. 

In the first of @sorenj 's notebooks, the author uses an interesting answer on the above by proposing a random search of weights for each label in order to find a optimum in the metric for with host-cv. That same approach but with some nuances was the one I finally followed.

###### Combining All Things:

From the previous approach, I tried several alternatives to merge the detoxify's labels. Rather than looking for a solution close to the linear optimum (increasing the number of iterations), I looked at the variability of the result of each iteration on host-cv. Since nothing guaranteed that the texts in the private belonged to the same family as the validation ones, I thought it might be a good idea to try a non-optimized submission but simply play with the results of an iteration.

To later combine the 3 models I did the same. The idea was not to optimize twice on the same texts and on the same workers. I also combined the luke's like this and therefore I did not use the training oofs but directly applying them to the prediction.

Luke's final ensemble with detoxify's was with ranges and I didn't use maths or randomness this time. Simply the kaggle-art to balance dataset diversity and model diversity (1/3 for the former and 2/3 for the latter). A final action was to randomize the cases of duplicate scores in the final submission if applicable.

###### Summary of Results:

I repeat the well-known validation scores and add the evaluation on the public and private leaderboard,

| Model name | Validation | Public | Private |
| -- | -- | -- | -- | 
| luke-base | 0.6983 | 0.79993 | 0.79167 |
| luke-large | 0.7014 | 0.80287 | 0.79898 |
| original | 0.69450 | 0.74507 | 0.79883 |
| unbiased | 0.69750 | 0.73364 | 0.79768 |
| multilingual | 0.69138 | 0.73919 | 0.79579 |

Next step,

| Ensemble | Validation | Public | Private | Weights (norm) |
| -- | -- | -- | -- | -- |
| luke's | 0.75150(leaked) | 0.80439 | 0.79698 | [0.53215757 0.46784243] |
| detoxify's | 0.70362 | 0.74605 | 0.80791 | [0.38403517 0.43512979 0.18083503] |

Last step,

| Submission | Public | Private |
| -- | -- | -- |
| final ensemble | 0.77587 | 0.81282 |

###### Checking Reproducibility and Randomness:

In case some strange phenomenon had been introduced in the solution, I checked [here](https://www.kaggle.com/coreacasa/jigsaw4-checkup-reproducibility-4th-place-solution?scriptVersionId=87635669) its reproducibility.

To check how important randomness (seed) is in the solution, I ran this [this notebook](Jigsaw4-Checkup-Randomness-4th-Place-Solution) and tried various seed changes (or intermediate additions) keeping the results a gold standard.

(In any case... yes, this time luck was on my side!)

That's all folks!
