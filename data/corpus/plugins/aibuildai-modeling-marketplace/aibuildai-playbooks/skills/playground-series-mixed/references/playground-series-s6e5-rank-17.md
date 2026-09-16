# Rank17 approach - diverse models and blend

Competition: playground-series-s6e5
Rank: #17
Source: https://www.kaggle.com/c/playground-series-s6e5/writeups/rank17-approach-diverse-models-and-blend

Hello all,

I am elated to present my solution writeup for the Playground Season 6 Episode 5 competition. I wish to extend sincere thanks to Kaggle for this interesting challenge and my fellow participants for their generous sharing through the month...

My solution is a by-product of a number of public ideas and my private inputs as well. Let's delve into the details as below- 

## Assignment details 

- Goal - To predict the probability of next lap F1 pit-stop by a race driver
- Model type - 2-class tabular classifier 
- Eval-metric - ROC-AUC score
- Competitor teams = 3023
- Public LB rank = 17
- Private LB rank = 17

## Overall architecture

As illustrated in the figure above, my overall solution architecture relied on 4 major pillars -
- Diverse features from ideas discussed in past playground solutions and my past pipelines 
- Special features designed considering the nature of the problem
- Models with wide and diverse parameters to elicit variety in the results 
- A robust ensemble using CV scores as base, with 3 stage model methods to elicit as much signal boost as feasible post the single model framework

I am currently fully engaged in 2 featured competitions and am aiming for the best result in both of them, considering this, I could manage to devote a couple of days for this competition. I am glad my strategy paid off as I invested 2-3 days this month and secured my rank this time!

## Feature Engineering 

This was the core element in this competition. I found from the initial days that this dataset was large, quite stable and was highly sensitive to an ensemble, but extremely sensitive to the features chosen. 1-2 good features would augment the CV scores massively, while a gamut of moderate features would not augment the CV scores over my private baseline leaderboard scores. 

My initial experiments and some relevant public opinions in the forum goaded me to try out as many feature ideas as possible within my available time this month. 

**My key feature ideas included -**

- category twins of numeric columns 
- arithmetic operations involving 1-2-3 numeric columns at a time 
- n-gram target encoder interactions with category features 
- original data as rows 
- target encoded original data columns as added columns 
- feature encoders - group-by features, with 1 feature as a grouper and another high cardinal feature as an aggregator
- global count encoder across all category columns
- leaving low cardinal category columns alone without target encoding 
- rounded columns as per public ideas and kernels 

**Specific features for this competition -**
- driver details across all years (ignored leakage)
- driver details for current year races excluding the current one 
- driver details for past years' races 
- driver details for only future year races 
- details across all races for the same year (ignored leakage)
- details across all races for the past years (ignored leakage)
- details across all races for the future years (ignored leakage)

I created more than 800 features using all of these ideas and used multiple feature subsets through the month from my feature store. Using such a master feature store is handy and often saves time and effort while iterating through experiment ideas. Organizing the associated code with a GitHub repository also helps with version controls. 

My single models often used between 30-700 features, considering resource availability and training time. Considering the time available, I preferred to train a lot of *diverse models* over a *lot of models*. Perhaps training more models with more diverse parameters (outsourcing to a Claw agent) would have landed us in a far higher position! 

## Single models 

I used a lot of single models with varying parameters through the month and a lot of ideas performed well, while a lot of them failed too. Failed single models added diversity to the ensemble, so all model candidates were chosen for the ensemble...<br>

Let's discuss them in details here, with CV scores across component models -

| Model type  | Key ideas and thoughts | 
| --- | ---  | 
| XgBoost, LightGBM, Catboost | - Most reliable single models with highest level of stability and good performance across parameters <br> - I used classifier and regressor options and both were equally effective |
| TABM | - Good diverse candidate, but took longer hours to train <br> - Performance was almost equal to gbdt models but slightly lower <br> - TABM consistently outperformed gbdts in 1-2 folds, and this added value to the ensemble | 
| REALMLP | - Good diverse candidate, and trained relatively quickly  <br> - Performance was lower individually compared to gbdt candidates across all folds, but this model added a lot of diversity as seen in CDF plots  |
| TABR | - similar to TABM, added diversity to the ensemble and was retained in the final ensemble | 
| CuML Random Forest| - individual performance was poor but added diversity to the ensemble | 
| Linear models| - individual performance was way below gbdt, but added a bit of diversity to the ensemble | 
| GNN| - individual performance was way below gbdt, but added a bit of diversity to the ensemble | 
| FT-Transformer| - individual performance was way below gbdt, but added a bit of diversity to the ensemble | 
|Yggdrassil | - again used for ensemble diversity despite poor individual performance | 
| Histogram Gradient Boosting Classifier | - trained on basic features for ensemble diversity only | 
| TabICL | - trained a gamut of candidates for ensemble diversity, training speed is good, but individual performance was poor <br> - this candidate added a lot of ensemble diversity | 


<br>All of these models were trained with the below CV scheme - 
`StratifiedKFold(5, random_state = 42, shuffle = True)`

<br>I stratified by year and target variable to ensure a double stratified folding scheme herewith.

## Model blending 

Choosing a robust blend strategy was crucial to maximize my gains here. My model blending strategy was divided into 3 sections - 

- Stacking with autogluon
- Blending with hill climber/ logistic regression/ ridge regression 
- Final blending by Arun to maximize the leaderboard for 1 candidate, using past submissions 

I trained in total, about 130 single models and stacked 2 times in total, resulting in a total of more than 200 models in the full pipeline. 

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

- Rely on the CV and don't fall in the blind blending bandwagon
- A good feature store, a functioning GitHub repo and organized folder structure is gold
- Codex was a better and more reliable agent partner than Claude this time around...
- Manual code writing is quite useful in feature extraction but not so useful in training and submission - LLMs do a better job at making a good CV pipeline!

## Concluding remarks 

Hearty congratulations to the winners and best wishes for the subsequent episodes and featured competitions ahead!! <br>

Happy Kaggling and regards!
Ravi and Arun
