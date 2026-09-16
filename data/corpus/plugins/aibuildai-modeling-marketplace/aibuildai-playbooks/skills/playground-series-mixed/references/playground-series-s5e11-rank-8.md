# Rank8 approach - trust the CV score

Competition: playground-series-s5e11
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s5e11/writeups/rank8-approach-trust-the-cv-score

Hello all,

I am elated to present my solution writeup for the Playground Season 5 Episode 11 competition. This was a very tumultuous month for me, with a lot of serious health issues in the first 2 weeks, but I am happy to be fit and back to competitive distance running now and have performed well in my opinion in the competition as well, based on the available time at my disposal. I would like to extend sincere thanks to my close family members and friends for being supportive during my difficult days at the start of the month. 

I wish to extend sincere thanks to Kaggle for this interesting challenge and my fellow participants for their generous sharing through the month. My solution is a by-product of a number of public ideas and my private inputs as well. Let's delve into the details as below- 

## Assignment details 

- Goal - To predict the chance of a subject customer repaying the loan 
- Model type - binary tabular classifier 
- Eval-metric - ROC-AUC score
- Competitors = 3724
- Public LB rank = 12
- Private LB rank = 8

## Overall architecture

As illustrated in the figure below, my overall solution architecture relied on 3 major pillars -
- Diverse features from ideas discussed in past playground solutions and my past pipelines 
- Models with wide and diverse parameters to elicit variety in the results 
- A robust ensemble using CV scores as base, with 3-4 methods to elicit as much signal boost as feasible post the single model framework

I tried a lot of ideas here and a lot of them failed at various stages. We will consider these separately in a later section.



## Feature Engineering 

This was the core element in this competition. I found from the initial days that this dataset was large, quite stable and was moderately sensitive to an ensemble, but extremely sensitive to the features chosen. 1-2 good features would augment the CV scores massively, while a gamut of moderate features would not augment the CV scores over my private baseline score of 0.92630. In some cases, adding certain features would even reduce the CV scores considerably. My initial experiments and some relevant public opinions in the forum goaded me to try out as many feature ideas as possible within my available time this month. My key feature ideas included -

- category twins of numeric columns 
- arithmetic operations involving 2-3 numeric columns at a time 
- n-gram target encoder interactions with category features 
- original data as rows 
- target encoded original data columns as added columns 
- feature encoders - group-by features, with 1 feature as a grouper and another high cardinal feature as an aggregator
- global count encoder across all category columns
- leaving low cardinal category columns alone without target encoding 
- rounded columns as per public ideas and kernels 
- digit extraction from numeric columns
- concocted digit components from multiple numeric columns

I created more than 1000 features using all of these ideas and used multiple feature subsets through the month from my feature store. Using such a master feature store is handy and often saves time and effort while iterating through experiment ideas. Organizing the associated code with a GitHub repository also helps with version controls. 

My single models often used between 65-800 features. CV scores for associated models are presented subsequently.

## Single models 

I used a lot of single models with varying parameters through the month and a lot of ideas performed, while a lot of them failed too. Let's discuss them in details here, with CV scores across component models -

| Model type  | Key ideas and thoughts | CV score range | LB score range | Key failures and success points | 
| --- | --- | --- | --- |
| XgBoost | - Offered the best single model option <br> - Smaller feature sets with minimal target encoding and n-grams performed the best <br> - Lowering the learning rate was effective, I used learning_rate of 0.0010 - 0.0030 and the CV scores improved <br> - depth of 6+ was counter-productive <br> - increasing reg_lambda was productive with higher features in the model | 0.92630 - 0.92778| 0.92610 - 0.92770 | - pseudo-labels performed well individually but failed with a blend <br> - using pre-training and fine-tuning type of an arrangement completely failed <br>  - original as columns performed better than original as rows <br> - structures like init_model and base_preds failed <br> -pseudo models for the remaining 11 original dataset columns using other features failed |
| LightGBM | - Offered a good competitive single model option <br> - Smaller feature sets with minimal target encoding and n-grams performed the best <br> - Learning rates of 0.0050 - 0.0070 were the best in this model type <br> - Max-leaves was a crucial parameter here  | 0.92640 - 0.92774| 0.92620 - 0.92772 | - pseudo-labels performed well individually but failed with a blend <br> - using pre-training and fine-tuning type of an arrangement completely failed <br> - structures like init_model and base_preds failed <br> - goss variant was a great diversity option here|
| Catboost| - Offered a limited single model performance <br> - Moderate feature sets with 100-150 columns with minimal target encoding and n-grams performed the best <br> - Learning rates of 0.01 were the best in this model type <br> - Other parameters did not yield a significant difference in CV performance  | 0.92630 - 0.92748| 0.92624 - 0.92740 | - pseudo-labels totally failed here <br> - overall model type was a failure individually, but I retained a few single models here for diversity |
| TABM| - Very good performance, often rivalled the XgBoost model performance on the leaderboard <br> - Round features boosted this model the best <br> - tabm-normal was the best architecture among the options <br> - Model training time was higher than the gbdt counterparts but using a bigger GPU like an A100/ A6000Ada reduced the time a lot  | 0.92665 - 0.92721 | 0.92698 - 0.92765 | - provided a decent amount of diversity in the ensemble and provided a good boost to the leaderboard scores individually  |
| REALMLP| - Performed akin to the TABM counterpart on the leaderboard with a lower CV score <br> - Round features and bigrams boosted this model the best  <br> - Model training time was higher than the gbdt counterparts but using a bigger GPU like an A100/ A6000Ada reduced the time a lot  | 0.92650 - 0.92695 | 0.92690 - 0.92769 | - provided a decent amount of diversity in the ensemble and provided a good boost to the leaderboard scores individually  |

All of these models were trained with the below CV scheme - 
`StratifiedKFold(5, random_state = 42, shuffle = True)`

## Model blending 

This was a crucial step in this competition as this was not the most effective step unlike other playground competitions. Choosing a robust blend strategy was crucial to maximize my gains here. My model blending strategy was divided into 3 sections - 

|Section| Strategy | CV  | LB | 
|----| --- | --- | --- | 
|1| Stacking with autogluon  | 0.92781-0.92820 | 0.92776 - 0.92787|
|2| Blending with logistic regression  | 0.92781-0.92784 | 0.92776 - 0.92788|
|3| Simple average of multiple ensemble approaches and past submission files  | 0.92778-0.92806 | 0.92778 - 0.92788|

All in all, a good single model was key here, a lot of the ensemble candidates elicited better CV and worse LB scores compared to good single models. This was unusual for a Kaggle competition of this type. 

## Training GPUs used

This was a relatively fast training pipeline, so I resorted to the below GPUs across models 

| Model type | GPU |
| --- | --- |
| XgBoost | L4 Colab, A100 Colab (80GB)  |
| LightGBM| L4 Colab, A100 Colab (80GB)  |
| CatBoost| A6000Ada  |
| TABM| A100 (80GB) |
| REALMLP| A100 (80GB) |

## Key lessons learnt 

- Single models are sometimes more effective than a conventional blending and stacking approach
- Rely on the CV
- A good feature store, a functioning GitHub repo and organized folder structure is gold
- Colab extension on VSCode is great!

## Concluding remarks 

Hearty congratulations to the winners and best wishes for the holiday season ahead! Wishing you the best for the last Playground episode for 2025 as well!
