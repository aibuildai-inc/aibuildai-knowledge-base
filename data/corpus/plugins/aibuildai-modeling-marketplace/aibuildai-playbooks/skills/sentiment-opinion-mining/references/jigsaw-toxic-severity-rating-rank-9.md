# [9th solution] Ensemble of 5 external dataset transformers

Competition: jigsaw-toxic-severity-rating
Rank: #9
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306187

First of all huge thanks to the organizers of the competition and to all the participants. I've learned a lot and strengthened my faith in the "Trust your CV" principle 🙂

#### High level description

List of models used in the ensemble:
1. CV: 0.7356, LB: 0.786 - roberta-base trained on CCC 2017 dataset using custom attention head, cross-entropy and margin ranking losses, a multi-staged prediction (first predict toxicity without knowing the 6 labels, then aggregate that prediction and 6 labels into the final prediction).
2. CV: 0.7450, LB: 0.747 - roberta-base trained on CCC 2017 dataset using a weighted average of margin ranking and binary cross-entropy loss.
3. CV: 0.7403, LB: 0.790 - roberta-base regression model trained on Ruddit dataset.
4. CV: 0.7448, LB: 0.751 - unitary/unbiased-toxic-roberta fine-tuned on Ruddit dataset with 2-layer DNN as a regressor.
5. CV: 0.7429, LB: never submitted - roberta-base regression trained on Wiki Talk Labels dataset.

The ensembling has CV 0.7605, LB 0.779. All those models were ensembled using rankdata + mean.

#### Validation strategy

All of the models were validated on `validation_data.csv` aggregated using majority voting.
First I used to define a final CV score as `CV * 0.65 + LB * 0.35`. Those weights represented my personal confidence in the score value. At some point, I've noticed a huge flakiness in the public LB score, small model change (0.001 score difference on CV) led to the 0.05-0.1 change in the LB score. I hypothesized that a small number of unique comments are used either in `more_toxic` or in `less_toxic` roles. More about it here https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/305301 . So I gave up on LB and stuck to the local CV score only.

#### Ensembling strategy

At first, I wanted to ensemble 14 models (trained on UBTC, Offenseval, C3, Hate Speech Measurement) instead of 5. But it in the ranking task more models do not necessarily give better score. The score distribution differs a lot across the models, the toxicity definition differs a lot across the datasets (Ruddit and Offenseval for example). To get maximum out of ensembling I've generated all the subsets of 14 models and run the validation on the results blended using rankdata + mean. This way I've identified that the highest score is reached by the subset of the 5 aforementioned models.

I've also experimented a lot with various ensmebling strategies: taking mean across the closest scores only, writing custom comparator function and sorting the comments based on voting, ranking + artificial distribution generation (Poisson, Normal), etc. Unfortunately, everything worked slightly worse than rankdata + mean on CV.

#### General transformer tricks

The following tricks were used for training all the models:

1. Per-layer LR, lower for backbone layers (2e-5), higher for DNN or final attention head (up to 1e-3).
2. High and dynamic validation frequency: typically the model is trained for 1-2 epochs with validation at least every 10% batches. The higher the accuracy, the higher the validation frequency.
3. AdamW optimizer and cosine annealing with warmup scheduler, warmup typically takes 1 validation cycle.

#### Things which didn't work

- MLM pretraining
- Augmentations / TTA. Although I've only explored some basic ones, they seem to change the sense or the toxicity of the comment significantly, and thus worsen the score. Moreover, I've started playing with them at the very end of the competition so probably that's just a lack of time.
- unitary/toxic-bert backbone. It gives superior performance, but it has been trained on the whole CCC 2017 dataset and that's why some portion of validation_data.csv comments have been exposed to it before. I've considered it a leaky one and switched to roberta-base and unitary/unbiased-toxic-roberta.
