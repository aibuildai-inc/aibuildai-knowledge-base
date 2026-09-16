# Rank 38 approach

Competition: playground-series-s6e3
Rank: #38
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/rank-38-approach

Hello all,

I am elated to present my solution writeup for the Playground Season 6 Episode 3 competition. I wish to extend sincere thanks to Kaggle for this interesting challenge and my fellow participants for their generous sharing through the month. 

My solution is a by-product of a number of public ideas and my private inputs as well. Let's delve into the details as below- 

## Assignment details 

- Goal - To predict the probability of customer churn from tabular features
- Model type - binary tabular classifier 
- Eval-metric - ROC-AUC score
- Competitors = 4117
- Public LB rank = 17
- Private LB rank = 38

## Overall architecture

As illustrated in the figure above, my overall solution architecture relied on 3 major pillars -
- Diverse features from ideas discussed in past playground solutions and my past pipelines 
- Models with wide and diverse parameters to elicit variety in the results 
- A robust ensemble using CV scores as base, with 3 stage model methods to elicit as much signal boost as feasible post the single model framework

I tried a lot of ideas here and a lot of them failed at various stages. We will consider these separately in a later section.

## Feature Engineering 

This was the core element in this competition. I found from the initial days that this dataset was large, quite stable and was moderately sensitive to an ensemble, but extremely sensitive to the features chosen. 1-2 good features would augment the CV scores massively, while a gamut of moderate features would not augment the CV scores over my private baseline leaderboard scores of 0.91355 - 0.91380. In some cases, adding certain features would even reduce the CV scores considerably. My initial experiments and some relevant public opinions in the forum goaded me to try out as many feature ideas as possible within my available time this month. 

My key feature ideas included -

- category twins of numeric columns 
- arithmetic operations involving 1-2-3 numeric columns at a time 
- n-gram target encoder interactions with category features 
- original data as rows 
- target encoded original data columns as added columns 
- feature encoders - group-by features, with 1 feature as a grouper and another high cardinal feature as an aggregator
- global count encoder across all category columns
- leaving low cardinal category columns alone without target encoding 
- rounded columns as per public ideas and kernels 

I created more than 600 features using all of these ideas and used multiple feature subsets through the month from my feature store. Using such a master feature store is handy and often saves time and effort while iterating through experiment ideas. Organizing the associated code with a GitHub repository also helps with version controls. 

My single models often used between 30-350 features. CV scores for associated models are presented subsequently.

## Single models 

I used a lot of single models with varying parameters through the month and a lot of ideas performed well, while a lot of them failed too. <br>
Let's discuss them in details here, with CV scores across component models -

| Model type  | Key ideas and thoughts | CV score range | LB score range | Key failures and success points | 
| --- | --- | --- | --- |
| XgBoost | - Offered the best single model option <br> - Moderately sized feature sets with target encoding and n-grams performed the best <br> - Lowering the learning rate was effective, I used learning_rate of 0.0010 - 0.0030 and the CV scores improved <br> - depth of 6-8 was productive <br> - increasing reg_lambda was productive with higher features in the model | 0.91557 - 0.91937| 0.91660 - 0.91690 | - pseudo-labels performed well individually but failed with a blend <br> - using pre-training and fine-tuning type of an arrangement completely failed <br>  - not using the original data was highly beneficial |
| LightGBM | - Offered a good competitive single model option <br> - Smaller feature sets with high target encoding and n-grams performed the best <br> - Learning rates of 0.0010 - 0.0050 were the best in this model type <br> - Max-leaves was a crucial parameter here  | 0.91684 - 0.919278| 0.91657 - 0.91688 | - pseudo-labels performed well individually but failed with a blend <br> - using pre-training and fine-tuning type of an arrangement completely failed <br> - structures like init_model and base_preds failed <br> - goss variant was a great diversity option here|
| Catboost| - Offered a limited single model performance <br> - Moderate feature sets with 100-150 columns with minimal target encoding and n-grams performed the best <br> - Learning rates of 0.005-0.008 were the best in this model type <br> - Other parameters did not yield a significant difference in CV performance  | 0.916153 - 0.919012| 0.91654 - 0.91685  | - pseudo-labels totally failed here <br> - overall model type was a failure individually, but I retained a few single models here for diversity |
| TABM| - Very good diversity based performance, but performed average on cv-lb <br> - Round features boosted this model the best <br> - tabm-normal was the best architecture among the options <br> - Model training time was higher than the gbdt counterparts but using a bigger GPU like an A100/ A6000 reduced the time a lot  | 0.91803-0.918753 | 0.91652 - 0.91689  | - provided a decent amount of diversity in the ensemble but did not provide the best boost to the leaderboard scores individually  |
| REALMLP| - Performed akin to the TABM counterpart on the leaderboard with a slightly lower CV score <br> - Round features and bigrams boosted this model the best  <br> - Model training time was much lower than TABM with limited epochs but higher than the gbdt counterparts <br> - Using a bigger GPU like an A100/ A6000Ada reduced the training time a lot  | 0.917511947- 0.918772143| 0.91655 - 0.91697 | - provided a decent amount of diversity in the ensemble and provided a moderate boost to the leaderboard scores individually  |
| OTHERS| - Included a lot of other models like GNN, logistic regression, TabICL, FTTransformer, AutoInt, Simple MLP, Yggdrassil gbdt learner with moderate levels of individual performance and a decent diversity benefit to the leaderboard | 0.9161587 - 0.91826879 | 0.91655 - 0.91683 | - provided a decent amount of diversity in the ensemble and provided a moderate boost to the leaderboard scores individually  |

All of these models were trained with the below CV scheme - 
`StratifiedKFold(15, random_state = 42, shuffle = True)`

## Model blending 

This was a crucial step in this competition as this was not the most effective step unlike other playground competitions. Choosing a robust blend strategy was crucial to maximize my gains here. My model blending strategy was divided into 3 sections - 

|Section| Strategy | CV  | LB | 
|----| --- | --- | --- | 
|1| Stacking with autogluon  | 0.9197735 - 0.91978465 | 0.91714-0.91730 |
|2| Blending with hill climber   |  0.91978254 - 0.91979354 |0.91718-0.91732 |
|3| Simple average of multiple ensemble approaches and past submission files  | 0.9197980  | 0.91735|

All in all, a good single model was key here, a lot of the ensemble candidates elicited better CV and worse LB scores compared to good single models. This was quite usual for a Kaggle competition of this type. 

I trained in total, about 220 single models and blended 5 times in total, resulting in a total of more than 330 models in the full pipeline. 

## Training GPUs used

This was a moderately fast training pipeline, so I resorted to the below GPUs across models 

| Model type | GPU |
| --- | --- |
| XgBoost | L4 Colab, A100 Colab (80GB)  |
| LightGBM| L4 Colab, A100 Colab (80GB)  |
| CatBoost| A6000Ada  |
| TABM| A100 (80GB) |
| REALMLP| A100 (80GB) |
| OTHERS| A6000Ada (48GB) |

## Key lessons learnt 

- Single models are sometimes more effective than a conventional blending and stacking approach
- Rely on the CV and don't fall in the blind blending bandwagon
- A good feature store, a functioning GitHub repo and organized folder structure is gold
- Colab extension on VSCode is great! 
- Claude code + Codex combination is great! Making a code plan with Claude and executing this with Codex is quite useful
- Using other LLMs within Claude Code is a good way to manage token expenses, but using the correct model with appropriate effort is necessary to manage the workload efficiently
- Claude remote-control is the best thing Anthropic could do for me!!
- Manual code writing is quite useful in feature extraction (this is a trust issue from my end) but not so useful in training and submission - LLMs do a better job at making a good CV pipeline!

## Concluding remarks 

Hearty congratulations to the winners and best wishes for the subsequent episodes and featured competitions ahead!! <br>
Happy Kaggling and regards!
