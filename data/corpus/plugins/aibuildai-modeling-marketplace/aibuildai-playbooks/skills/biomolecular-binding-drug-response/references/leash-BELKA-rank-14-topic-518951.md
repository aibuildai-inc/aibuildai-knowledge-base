# 14th Place Solution (7th in Public): All We Need is Frequent Checkpointing

Competition: leash-BELKA
Rank: #14
Source: https://www.kaggle.com/c/leash-BELKA/discussion/518951

Congratulations to all the winners and thanks to the organizers for hosting this interesting competition.
As we all know, shared targets and nonshared targets have completely different tendencies, so we built different pipelines.


# Nonshared Target Part ([yu4u](https://www.kaggle.com/ren4yu))
The most important observation in the nonshared target part is that the validation score of the nonshared target overfits very quickly, even within just one epoch of training.
Therefore, we changed the validation interval to 0.01 epoch, and as a result, successfully obtained a checkpoint that does not overfit.

## CV Strategy
CV strategy to simulate the nonshared targets is also important.
We used a 5-fold CV strategy, where BB1, BB2, and BB3 (with the BB2 building blocks removed) building blocks were split into 5 folds. Then we removed the other data so that the training data contains only the training building blocks and the validation data contains only the validation building blocks.
We released the notebook that generates the CV folds [here](https://www.kaggle.com/code/ren4yu/leash-split-for-noshare/).

## Model
[ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)

## Training
- 5-fold CV.
- AdamW optimizer with LR=1e-3 (fixed), weight_decay=1e-5, batch_size=512.
- Train for one epoch and validate at each 0.01 epoch interval.

## Selecting the Best Checkpoints
- We found some folds have different TP distributions from whole training data, so we used only fold0 and fold2 that have similar TP distributions to the whole training data.
- In our training, we do not complete even one epoch of learning, so there is data that is not used in a single training run. By training with multiple seeds (11 seeds for each fold), we utilize all the data.
- Finally, we excluded seeds with extremely low CV scores and used an ensemble of 19 models for the final results.
- In selecting the best checkpoints, we used two strategies: (1) a single checkpoint is selected based on the average score of the three targets, and (2) select three checkpoints based on the scores of each of the three targets. In the private LB score, the former strategy was better.


# Shared Target Part ([monnu](https://www.kaggle.com/fuumin621))
The main strategy was to improve the [public 1DCNN model](https://www.kaggle.com/code/ahmedelfazouan/belka-1dcnn-starter-with-all-data) by concatenating features from ECFP.
As an option, we also trained models with additional features from ChemBERTa.

## CV Strategy
- We used 5-fold Stratified Kfold splits.
- We released the notebook that generates the CV folds for share [here](https://www.kaggle.com/code/fuumin621/leash-split-for-share/notebook).

## Preprocess
- ECFP: Used rdkit. r = 4, bit = 2048 or 3072
- 1DCNN: Encoded SMILES strings into numerical values and converted them into fixed-length vectors
- chemberta_feature (optional): Used the ChemBERTa model to infer the SMILES strings and used the 384-dimensional output as features.

## Model
- SMILES are passed through an embedding layer, followed by 4 layers of 1D convolution
- ECFP is passed through an FC layer to 128 dim
- The outputs above are concatenated and passed through an FC layer to output binary classification scores for 3 targets
- Optionally, the 384-dimensional output of ChemBERTa can be passed through an FC layer and then concatenated

## Training
- 5-fold CV.
- AdamW optimizer with LR=1e-3, weight_decay=0.05, batch_size=4096.
- num_epochs=25

## Score
The scores of the trained models are as follows:
| Model Name | ECFP | chemberta_feature | CV | PublicLB(mask noshare)|
|------------|------|-------------------|----------|----------|
| exp031     | r=4, bit=2048 | No  | 0.6589 | 0.352 |
| exp032     | r=4, bit=2048 | Yes | 0.65934 | 0.351 |
| exp039     | r=4, bit=3072 | No  | 0.66001 | 0.350 |
| average ensemble     | - | - | - | 0.352 |

In the end, we submitted the average ensemble.
