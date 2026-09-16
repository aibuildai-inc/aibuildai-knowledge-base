# 8th Place Solution

Competition: march-machine-learning-mania-2024
Rank: #8
Source: https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/493041

Thanks to kaggle and everyone involved for hosting such an exciting competition. March Madness was the first competition I ever participated in back in 2019, so I'm delighted to have won a prize this time around. However, this year I spent more than half of my available time understanding a new metric and couldn't allocate enough time to write the code. Therefore, please understand that my solution is a combination of public notebooks.

Update(April 18, 2024): I have released the code other than the submission Notebook. And, I have added the results of late submission.

## Code
Submission notebook is [here](https://www.kaggle.com/code/flat831/ncaa2024-sub-8th-place-code). The notebook reads from [this dataset](https://www.kaggle.com/datasets/flat831/ncaa-2024/data). Each data in the dataset is created from the following code.
* EloRating_mens_10.csv & EloRating_womens_10.csv: https://www.kaggle.com/code/flat831/elo-rating
* features_mens.csv: https://www.kaggle.com/code/flat831/create-feature-mens
* features_womens.csv: https://www.kaggle.com/code/flat831/create-feature-womens
* train_mens.csv: https://www.kaggle.com/code/flat831/create-train-mens
* train_womens.csv: https://www.kaggle.com/code/flat831/create-train-womens
(You need to execute in the order of  elo-rating -> create-feature-XX -> create-train-XX.)

## Model
weighted ensemble the following two models:
* [rustyb's ≒ radder's XGB Model](https://www.kaggle.com/code/rustyb/paris-madness-2023) (weight = 0.8)
* [theoviel's Logistic Regression Model](https://www.kaggle.com/code/theoviel/it-s-that-time-of-the-year-again) (weight = 0.2)

The Logistic Regression Model used 538's ratings in the past, but I couldn't use them this year. Instead, I implemented Elo Rating in R as a substitute. These two models have produced good results in past competitions and capture different tendencies, so I think ensemble is effective.

## Simulation
I'm using [lennarthaupts' notebook](https://www.kaggle.com/code/lennarthaupts/simulate-n-brackets) as is. (n_brackets = 100000)

## Manual Overrides
For the 1st submission, I've set it up so that UConn wins 100% in the men's bracket and South Carolina wins 100% in the women's bracket. Similarly, for the 2nd submission, I've set it up so that Houston wins 100% in the men's bracket and South Carolina wins 100% in the women's bracket.

As mentioned in the [jacklichtenstein's 2nd solution](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/492761), I think this strategy is very effective in this year's format. Also, given that last year's results were chaotic, I thought many participants might hesitate to use overrides this year. This is a kind of metagame; the more participants use overrides, the less value there is in using overrides.

## Late Submission
The result of the Late Submission is as follows:

| Model | Override | Score | Note |
| --- | --- | --- | --- |
| LR:XGB = 2:8 | UConn&SC | 0.05604 | 8th place model |
| LR:XGB = 2:8 | None | 0.05985 | Equivalent to 69th place |
| LR:XGB = 2:8 | UConn | 0.05683 | Equivalent to 13th place |
| LR:XGB = 2:8 | SC | 0.05906 | Equivalent to 45th place |
| LR:XGB = 10:0 | None | 0.05761 | Equivalent to 23th place |
| LR:XGB = 0:10 | None | 0.06106 | Equivalent to 139th place |
| LR:XGB = 10:0 | UConn&SC | 0.05482 | Equivalent to 3rd place |
| LR:XGB = 10:0 | UConn | 0.05545 | Equivalent to 4th place |
| LR:XGB = 10:0 | SC | 0.05698 | Equivalent to 14th place |

* At least for this year's results, the LR model has better performance.(I decided on the ensemble rate by gut feeling due to lack of time, but I should have thoroughly validated it.)
* In my model, the override effect of UConn had a very significant impact.
