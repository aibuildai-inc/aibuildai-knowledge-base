# 1st Place Solution

Competition: um-game-playing-strength-of-mcts-variants
Rank: #1
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549801

First, I'd like to thank Kaggle and Maastricht University for hosting such an interesting competition - I learned a lot. I'd also like to congratulate everyone who finished near the top of the leaderboard. You guys had me sweating until the end!

# Overview

I believe the main trick which allowed me to win was running a tree search for a few seconds on the starting position of each game to compute additional features describing how balanced it is and how quickly the search can be executed (semi-redundant with the `AdvantageP1`, `MovesPerSecond`, and `PlayoutsPerSecond` columns provided by the competition organizer). The supplemental features (and all of the ones most other competitors were using) were then fed to a stacked ensemble of CatBoost, LightGBM, TabM, and isotonic regression models. A diagram illustrating this is included below.



RMSE scores for each of the GBDT/NN + isotonic model stacks are included below.
| Base model | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| CatBoost | 0.362 | 0.415 | 0.421
| LightGBM | 0.357 | 0.419 | 0.424
| TabM | 0.352 | 0.420 | 0.424
| All (ensemble) | 0.344 | 0.413 | 0.417

</p>Ensemble weights were tuned to my CV using Nelder-Mead.

As you might be able to tell from the table above, my local cross validation was somewhat flakey, but the ensemble which scored best in CV happened to also have the best LB scores (both public & private).

# Starting position evals
The `AdvantageP1` column is computed based on play between agents which make random moves, so it is not perfectly correlated with how much of an advantage player 1 has during non-random play. Computing how much of an advantage player 1 *really* has by simulating games between non-random agents is too CPU intensive to submit and training a language model to predict how much of an advantage (or disadvantage) player 1 has based on the rules did not yield very accurate results, so I instead generated a supplemental indication of how balanced each game was by running a tree search briefly on the game's starting position. I believe this yields results of similar accuracy to fully simulating games between non-random agents ~250x faster. Much like simulating the games there is a configurable tradeoff between speed and accuracy, with a 15 second search being roughly as good as 1 hour of simulating matches between non-random agents.

In early experiments, feeding models supplemental game balance metrics computed by running tree searches on the starting positions for 15 seconds per game ***improved my scores by ~0.045 CV, ~0.012 LB***. This is what initially got me to the #1 spot on the public LB roughly a month into the competition.

The tree search algorithm I used to compute the game balance metrics is equivalent to the `UCB1Tuned-0.6-Random200-true` algorithm the competition organizer used to play some of the games (my code links against the Ludii player's MCTS implementation as a dependency). I believe the most important aspects of that configuration for this workload are the `selection` and `playout` values, `explration_const` and `score_bounds` didn't seem to matter. A data visualization from one of my early experiments which roughly illustrates this is included below. As you can see, the configurations which worked "well" all either used a UCB1 or UCB1Tuned selection strategy and random or NST playouts. That being said, NST seemed to work worse on the leaderboard data than the training data the competition organizer provided and UCB1Tuned seemed to work marginally better than UCB1 (in CV - not sure about LB), so the winning submission used UCB1Tuned & random playouts.



# Supplemental search speed metrics
There's some random noise in the `MovesPerSecond` and `PlayoutsPerSecond` columns provided by the competition organizer. As a result, while computing the starting position eval/game balance metrics described above, my code also reports information about how quickly it was able to explore the game tree, namely the number of actions & search iterations per second. Feeding these features to my models improved my CV scores by ~0.005, so they were used in the winning solution, but they did not seem to have any significant impact on the LB scores (the impact was within random run-to-run variance).

# Additional training data
I generated a total of 14,365 extra training rows with 484 extra unique rulesets. The extra rulesets were generated via a mixture of two approaches:
1. Implementing & running my own version of [GAVEL](https://arxiv.org/pdf/2407.09388).
2. Ordinary instruction-tuned LLMs (Llama 3.1 70B & Qwen 2.5 32B) with few-shot prompting.

Of the 484 annotated rulesets, 391 were produced using approach #1, 93 were produced using #2. I generated far more games with approach #2 (over 10k of them), but wound up discarding and heavily downsampling them due to quality issues.

The rulesets generated using approach #1 were generally much higher quality than #2 (at least according to the metrics GAVEL is selecting for, namely balance and strategic depth), but #1 produced "playable" games much more slowly than #2, even accounting for the fact ~95% of the rulesets generated using #2 had to be discarded due to Ludii compilation or runtime errors. I don't recall exact throughput or error rate statistics off the top of my head, but believe there was an order of magnitude difference in the rate at which playable rulesets were produced by each approach.

As for the data annotation portion of this, I attempted to save CPU time by having each (ordered) agent pair only play 4 matches per ruleset (e.g. 4 matches with agent "A" moving first, then another 4 with agent "B" moving first), so the my labels were pretty dirty in comparison to the ones from the competition organizer, but using the extra data still improved my scores by ~0.004 CV, 0.001-0.002 LB. I mitigated the label noise by weighting the extra training samples less heavily than the ones provided by the competition organizer. Some models seemed more sensitive to the noise than others, so I tuned the extra data weight separately for each type of model. The weights used in the winning solution are outlined below. These were tuned to my local cross validation, not the leaderboard.

| Model | Extra data weight |
| --- | --- |
| CatBoost | 0.25 |
| LightGBM | 1.0 |
| TabM | 0.5 |

</p>Scaling up the amount of supplemental data seemed to produce severely diminishing returns (e.g. the first batch of ~7k extra training rows improved my CV scores by ~0.003 but the second only produced a ~0.001 improvement), with the optimal extra data weights falling as I scaled up, so I suspect label noise was becoming an increasingly large problem as I scaled up. With the benefit of hindsight, I suspect I might have gotten better results by annotating half as many games with double the compute per game.

# Data augmentation
I used two forms of data augmentation:
1. All of the tree search related features for the training examples were computed 10 times. The values included in each row fed to the models during training were randomly selected from 9 runs that were not held out for cross-validation purposes, so models were strategically exposed to the random noise in the features, rather than having the same errors duplicated across all of the rows for each ruleset.
2. All of the nondeterministic features in the data provided by the competition were recomputed 5 times and used to compute a more accurate (less noisy) estimate of the "real" values of the features. The values fed to the models during training were uniformly sampled between theses estimates and the ones provided by the competition or during my initial data generation runs.

1 improved both my CV & LB scores by ~0.002, with a good correlation between CV and LB. The impact of #2 was a bit murkier as its impact depended on whether or not I dropped some of the features impacted by (suspected) Ludii player version discrepancies discussed in the next section, but was generally in the ballpark of <= 0.002.

# Feature selection
While re-annotating the rulesets provided by the competition organizer, I noticed there were 43 features whose values were perfectly consistent from one run to the next when I generated them on my machines, but different from the values provided by the competition organizer. My best guess is that this was likely caused by some sort of Ludii player version discrepancy (I was using version 1.3.13), but I don't have super concrete evidence to prove that. Interestingly, when I tried dropping the columns with inconsistent values, my CV scores consistently improved by 0.001-0.002, which was surprising to me as no other feature selection strategy consistently resulted in a score improvement across multiple random seeds and models. It appears all of these features need to be dropped at once in order for there to be a measurable benefit (not one at a time) and importance based filtering did not do a good job identifying them as unimportant/poisonous. 

The winning submission did not use any of the features impacted by the suspected version discrepancy. The LB impact of this is a bit unclear because I submitted multiple changes that seemed to help in CV at once in the last 12 hours without testing incremental versions - more on that later.

# Isotonic regression
This was largely inspired by a trick I saw in some public notebooks in which people were multiplying their raw predictions by some coefficient then clipping to be in a range narrower than -1 to 1, but seemed to produce a better CV-LB correlation. More specifically, using the scale + clip trick with parameters tuned to my CV did more harm than good on the leaderboard, but fitting isotonic regression models which use OOF predictions as features and running them as a postprocessing step improved both my CV & LB scores by around 0.002.

For most base + isotonic model stacks, I used [centered isotonic regression](https://pypi.org/project/cir-model/), which seemed to score *slightly* better in cross validation than scikit-learn's "traditional" isotonic regression implementation, presumably due to the strict monotonicity constraint making CIR models less prone to overfitting. By "slightly" better, I mean a difference in the 4th decimal point. However, for CatBoost it caused some crashes due to lacking functionality analogous to scikit-learn's `out_of_bounds='clip'` parameter, which I did not bother working around properly for CatBoost (clipping raw predictions to the range -1 to 1 at inference time before feeding them to CIR doesn't help if the training examples were in a range narrower than that!), so the CatBoost model stack used scikit-learn's non-centered isotonic regression implementation.

# TabM
In the last month of the competition, Yandex (the authors of CatBoost) released a new tabular deep learning library and research paper describing it, TabM ([https://github.com/yandex-research/tabm](https://github.com/yandex-research/tabm), [https://arxiv.org/pdf/2410.24210](https://arxiv.org/pdf/2410.24210)). Deep learning based solutions did not seem very promising in some of my experiments earlier in the competition, but the benchmark results in their paper were impressive, so I decided to give it a try. It worked surprisingly well! More specifically, it turned out to be competitive with LightGBM on the leaderboard and was my strongest single model in cross validation. During early experiments adding it to my previous-best ensemble improved LB scores by ~0.001, but I'd need to do some ablation testing to say how much it helped in the final ensemble.

# Hyperparameter tuning
The winning solution's hyperparameters were selected by running Optuna a half dozen or so times per model type with 5-fold cross validation and only a single random seed, then re-checking each of the "optimal" configurations with 10-fold cross validation and 3 seeds. CV was [GroupKFoldShuffle](https://github.com/scikit-learn/scikit-learn/issues/20520#issuecomment-1868029324) partitioned on the ruleset names.

A few things I found relatively noteworthy about the winning config are outlined below.
* TabM's piecewise linear embeddings for continuous numerical features seemed to help tremendously. I suspect this is the main reason it was able to beat a relatively traditional MLP.
* The optimal config for TabM was slightly outside the search range described in the original TabM research paper. My best results were achieved with relatively wide hidden layers, small batch sizes, and low learning rates.
* Setting LightGBM's `boosting` parameter to `dart` was critical to getting it working well enough to be useful. This improved my LightGBM-only scores by roughly 0.004 CV, 0.003 LB, but came at the expense of a 5-10x increase in training runtimes. My hyperparameter searches were run both with and without dart.

# Ensembling
One detail which isn't clear in the overview that I figure I should probably mention is that each of the model blocks show above is actually an ensemble of 20 models, the models trained during 10-fold cross validation repeated for 2 random seeds. CatBoost & TabM rely on early stopping to work well, so the individual models can't really be trained on all the data - ensembles was the best way to make use of all of it. I generally repeated my CV runs for 3 seeds but only used models from 2 of them in order to conserve memory in the inference notebook (of the 30 GB RAM, 18 are used for tree searches and 10 are used for the models, so I can't really scale up to a third seed). The 2 seeds used were selected arbitrarily, not tuned in any particular way.

# "Trust CV" vs. "Trust LB"
During the last week or so of the competition, I pursued two strategies for creating final ensembles in parallel.
1. "Trust LB" - I tried selecting candidate configurations proposed by optuna that randomly happened to work well on the leaderboard when trained with "version 4" of my extra data (~half as much as the winning solution) and tuned the ensemble weights to the leaderboard. The candidate models used while tuning this approach to the leaderboard did not make full use of the feature selection & data augmentation approaches described above because most of my hardware was focused on approach #2...
2. "Trust CV" - I tried selecting candidate model hyperparam configurations proposed by Optuna by double checking them with 10-fold CV run for multiple seeds and more thoroughly tuned the data augmentation, feature selection, and extra data weight choices to my CV. This process was done with "version 6" of my extra training dataset, the one described above. The choices made during this process were not tested on the leaderboard until the last 12 hours of the competition or late submissions after the competition ended.

Surprisingly, "Trust CV" beat "Trust LB" on both the public and private leaderboards. I really did not expect that in light of how much the public LB seemed to hate ensembles with weights tuned to my CV, how little the final 2x scaling in the amount of extra training data seemed to help (in early experiments it made my LB scores worse!), and how weakly correlated my CV scores previously seemed to be correlated with the public LB. The "Trust CV" strategy was mostly intended to avoid falling victim to a major shakeup in case the public and private leaderboards were poorly correlated with one another, I did not pursue it with the expectation that I'd be able to beat the public LB score of an ensemble tuned to the public LB.

| Strategy | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| "Trust LB" | 0.3554 | 0.4142 | 0.4190
| "Trust CV" | 0.3442 | 0.4137 | 0.4178

# Things I tried that didn't work very well
* Using unsplitted agent strings as categorical features
* TF-IDF & LSA
* MLPs
* XGBoost
* Various deep learning based approaches for generating text embeddings as extra features
* Training language models to predict how balanced each game is based on the lud rules
* OpenFE
* The extra features I saw people using in public notebooks
* The model stacking approach I saw people using in public notebooks (OOF predictions as features)
* Importance based feature selection
* Iterative forwards and backwards feature selection
* Training models to predict average elo ratings for MCTS-based agents in which each part of the MCTS config is set a particular way (e.g. average elo of agents which use random playouts for a particular game) and using those estimated elo ratings as supplemental features.
* Training tabular classification models to identify horrendously imbalanced and drawish games that always end a particular way (agent 1 win, agent 2 win, or draw) and incorporating the resulting predictions into an ensemble.
* Adding random noise to some of the features for data augmentation purposes
* Enforcing monotonicity constraints with CatBoost
* Using CatBoost's finetuning functionality to "improve" the predictions from other models (similar to [this](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/543725) winning solution to a past competition)
* Using ensembles of multiple MCTS algorithms to produce the starting position evals. At best, this seems equivalent to letting a single algorithm run longer per ruleset.

I think it is plausible that some of the items above could work well with additional effort. I frequently move on when early results do not seem promising without making a super determined effort to get things working well.

# Avenues for further improvement (things I missed)
I completely overlooked the flip/inversion data augmentation trick that was used by pretty much all other top teams! Looking at their writeups, I suspect this is the main reason other people were able to score almost as well as me without running tree searches on the game starting positions. If the statistics in @goldenlock's [writeup](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549617) are correct, then I believe a hybrid solution could likely score in the ballpark of 0.407 on the private leaderboard (0.01 better than my 1st place solution 🤯).

# Links
* [Inference notebook](https://www.kaggle.com/code/jsday96/mcts-multi-isotonic-ensemble-2?scriptVersionId=210801408)
* [Extra training data](https://www.kaggle.com/datasets/jsday96/mcts-extra-training-data)
* [Starting position analysis, training, test, and data generation code](https://github.com/jday96314/MCTS)
