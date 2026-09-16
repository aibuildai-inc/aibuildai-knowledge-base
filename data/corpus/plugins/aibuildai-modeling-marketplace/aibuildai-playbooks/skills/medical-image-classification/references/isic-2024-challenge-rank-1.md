# 1st Place Solution

Competition: isic-2024-challenge
Rank: #1
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/533196

Congratulations to all the winners and participants! I can hardly believe I’m writing a post for the “1st Place Solution.” I’m incredibly grateful to the organizers for the competition itself, and to the kagglers, especially @greysky for his public notebook, which I used as the starting point for my experiments ([ISIC 2024 | Only Tabular Data](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data)) about three weeks ago, shortly after the LMSYS competition ended. 

### Solution Overview
My solution, like that of most participants, was based on ensembling various implementations of GBDT models along with image models. For the image models, I used two architectures: **EVA02-small** (eva02_small_patch14_336.mim_in22k_ft_in1k) and **EdgeNeXt** (edgenext_base.in21k_ft_in1k). A significant amount of time was also spent generating synthetic data and attempting to incorporate data from previous competitions into the current pipeline.

### Cross-Validation Strategy
I used a simple 5-fold Stratified Group KFold without any specific tuning. For performance evaluation, I applied a strategy once described by @daniel89 for the [Mercedes-Benz Greener Manufacturing competition](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing). Just as he did, I ran CV 10 times with different seeds, calculated the t-statistic (using scipy.stats.ttest_rel), and used the p-value to guide my decisions. To address the multiple comparisons issue, I tested only the most significant changes using this approach. If CV showed any significant improvement, I tested it on the Public leaderboard, and if it improved there as well (which it almost always did), the changes were added to the final solution.
However, towards the end of the competition, when I was stuck around a 0.185-0.186 score (0.173 on Private), I deviated from this rule (it was disheartening to see myself drop from 23rd place down) and started testing even small hypotheses, lowering the p-value threshold to 0.2 and mainly relying on the Public leaderboard. This led to my final solution performing better on the Public leaderboard, but slightly worse on the Private leaderboard.

### GBDT Models
#### Parameters and Ensembling 
I used CatBoost, LGBM, and XGBoost. Each model was trained on a GPU using Group K-Fold (5-folds) 10 times with different seeds for the model and data splitting. This resulted in a total of 150 models. To be honest, this didn’t provide any significant improvement compared to the base setup with just 45 models, but since the models trained fairly quickly (the total training time was under 20 minutes), I decided to increase the ensemble size, even if the gains were minimal.
For each model, the predictions were ranked (.rank(pct=True)) and averaged with equal weights. All models were trained with default parameters from @daniel89’s public notebook, including undersampling and oversampling. The only exception was CatBoost, which was trained for 1000 steps with early stopping based on the validation set (od_wait = 100). For CatBoost, I used the following parameters:
```python
1. 'learning_rate': 0.026
2. 'l2_leaf_reg': 18
3. 'random_strength': 4.7
4. 'depth': 6
5. 'bagging_temperature': 0.874
6. 'border_count': 256
7. 'grow_policy': 'Lossguide'
8. 'min_data_in_leaf': 38	
```

The parameters were initially selected using Optuna, based solely on the tabular data. Unfortunately, I didn’t have time to retune them after introducing features based on CV-models. I also didn’t optimize the parameters for the other models.

#### Feature Engineering
Most techniques adopted from [here](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data)
Additionally, I’ve added:
- Total Area of lesion  per patient and per patient & anatom_site_general

While most of the features describe absolute lesion parameters, I aimed to add additional relative information (e.g., describing how abnormal a specific lesion is for the patient). To achieve this, I selected the top features based on CatBoost’s feature importance and calculated the Local Outlier Factor score for each patient’s lesion. This resulted in a significant improvement in my CV score (from 0.18149 to 0.18185) and was reflected on the leaderboard as well.

#### Didn’t work
Another attempt involved clustering the moles using the most important features and calculating the Z-score for each one within the cluster. This slightly improved the CV and public leaderboard scores but didn’t result in significant improvement on the private leaderboard.

### Vision-models
- The augmentations were taken from [a previous competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412). I used EVA02 small and EdgeNeXt base as the models. They demonstrated a good balance between metrics and inference time.
- To address the significant class imbalance, the examples from different classes in the training batches were balanced at a 1:1 ratio. Both architectures were trained with different seeds following the previously described 5-fold Stratified Group KFold scheme with early stopping based on the validation set (200 epochs, early stopping tolerance = 10 validation checks). In practice, most models rarely trained beyond 100 epochs.
- Additionally, since validating the model on the entire validation dataset takes considerably more time than training one epoch with weighted sampling, in most experiments, validation was performed every 5 epochs initially, then every 4 epochs, and so on, reducing the frequency of checks until validation was done every epoch by the 50th epoch. All the resulting models were used for inference on the test data. The obtained OOF predictions were used to train GBDT models.

#### Integration of Model Predictions into the GBDT Model
- The best-performing approach involved using standardized model predictions, where the standardization was applied independently for each model’s predictions. The resulting feature varied within the range of p1: -0.44 and p99: 4.99. During inference, predictions from models of the same type were averaged.
- However, since models tend to slightly overfit to the test dataset due to early stopping based on validation data, normally distributed noise with a standard deviation of 0.1 was added to the model predictions when training the GBDT. I tested noise values of 0.02, 0.05, 0.08, and 0.12 via leaderboard probing (one of few things tested without cv).
- Additionally, the ratio of each prediction to the average prediction for all of a patient’s moles was calculated, which consistently improved CV and leaderboard performance. Noise was added to these features in the same way as described above.

#### Didn’t work
- I attempted to more frequently select hard examples from the negative class when forming batches, where the probability of selection increased based on the difficulty. The difficulty was assessed using LogLoss from the OOF predictions.
- I also experimented with pretraining on data from previous competitions and [other sources](https://github.com/ImageMarkup/isic-cli), but this did not yield significant improvements. 
- Averaging model predictions for several variations of augmented samples also did not produce any positive results.


#### Using Data from Previous Competitions
Due to the extremely small number of positive examples, it is expected that the model would struggle to recognize borderline cases of skin lesions. To address this, before integrating the models trained on the images from this competition, I trained a 3-class classification model (bkl/melanoma/nevus) using EVA02 small on data obtained from [repo](https://github.com/ImageMarkup/isic-cli). I then applied the following rule based on diagnosis_pr:
```python
nevus -> nevus
melanoma -> melanoma
basal cell carcinoma -> bkl
seborrheic keratosis -> bkl
solar lentigo -> bkl
lentigo NOS -> bkl
```
All remaining samples were marked as benign_malignant == 'benign' but with diagnosis_pr != 'bkl' were labeled as ‘nevus’. 

Adding the predictions from this model to the models based only on tabular data significantly improved both the CV and leaderboard scores. CV: 0.1756 -> 0.1760, public LB: 0.180 -> 0.182, private LB: 0.163 -> 0.165.

At the same time, in the final model, adding these features slightly improved the CV: 0.18185 -> 0.18195. However, there was also a slight improvement in the public and private leaderboard scores.

#### Synthetic Data (or where most of the time was spent)

I was particularly interested in the potential of using synthetic positive examples to improve the model’s performance, and this idea was one of the main reasons I decided to participate in the competition. A similar approach was implemented in [Derm-T2IM](https://arxiv.org/html/2401.05159v1/#bib.bib9)
The process for generating synthetic data is outlined below.
.png?generation=1725919714475856&alt=media)

Below, you can compare real photographs of malignant lesions, examples generated by the Derm-T2IM model, and models trained on competition data. 
.png?generation=1725919770830018&alt=media)

The average metrics at the individual model level demonstrate the effectiveness of the synthetic data.


It’s evident that the CV scores for models trained on synthetic data are consistently better. The leaderboard (public and private) results are slightly better, and on average, models trained on synthetic data perform better. 




For example, an ensemble of models trained on synthetic data shows slightly better results on the Private LB (0.140 vs 0.142) and marginally better on the Public LB (within the range of 0.157). However, unfortunately, the addition of models trained on synthetic data did not improve the final ensemble, so they were not included in the final solution.
If anyone is interested in continuing experiments in this direction, I’m attaching one of my [datasets with synthetic mole images](https://www.kaggle.com/datasets/ilya9711nov/isic-2024-synthetic/data).


P.S. All the source code for models training can be found [here](https://github.com/ilyanovo/isic-2024/tree/main)
P.P.S. [Submission code](https://www.kaggle.com/ilya9711nov/first-place-submission)
