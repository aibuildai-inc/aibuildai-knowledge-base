# #6 solution - Ensembling was the key

Competition: playground-series-s5e11
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s5e11/writeups/6-solution-ensembling-was-the-key

It was a good competition, even though the winner seemed clear after the first week. Congratulations to @mahoganybuttstrings and the rest of top scorers. I am impressed that @angelosmar1 and @pirhosseinlou placed so high with only a couple of submissions.

My thanks to @yekenot and @masayakawamata for excellent NN notebooks. Both of them produced very diverse models with good CV scores, but they would not ensemble with the rest of my models. I am fairly certain that I left a lot on the table by not finding a way to include them into my ensemble, but that's how it goes sometimes.

We always talk about model diversity giving rise to good ensembles, and in my case that was achieved by treating all features as categoricals. We had a couple of features with cardinality > 100,000, but I converted that into a manageable number by rounding up `loan_amount` to the nearest 10, and `annual_income` to the nearest 100. This was the number of categories after that modification, and it worked just fine with Keras factorization machines (FM):

```
annual_income 1778
debt_to_income_ratio 572
credit_score 406
loan_amount 3683
interest_rate 1498
gender 3
marital_status 4
education_level 5
employment_status 5
loan_purpose 8
grade_subgrade 30
```

That scored ~0.926, and adding bi-gram interactions pushed it to 0.9265. On their own these were not the best models I had, but they meshed really well with GBM models. You can see from the attached hill climbing plot that these FM predictions pushed the CV score from 0.9275 to ~0.928 after a single round of hill climbing. Both Keras and CatBoost stacking were overfitting on these large ensembles, even though in previous AUC competitions they were the best. Eventually I got Keras to work after some heavy regularization, but at that point its scores were slightly behind hill climbing.

Here is a list of some representative models that went into the final ensemble.

| Model | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| XGBoost num | 0.927548 | 0.92742 | 0.92862 |
| CatBoost cat | 0.927551 | 0.92743 | 0.92862 |
| XGBoost cat | 0.927516 | 0.92738 | 0.92854 |
| LightGBM num | 0.927171 | 0.92714 | 0.92839 |
| CatBoost num | 0.927202 | 0.92725 | 0.92837 |
| LAMA AutoInt | 0.926842 | 0.92702 | 0.92829 |
| LAMA DenseLight | 0.926779 | 0.92705 | 0.92824 |
| RealMLP | 0.926831 | 0.92709 | 0.92823 |
| Keras FM | 0.926564 | 0.92653 | 0.92778 |
| PyT-trompt | 0.926300 | 0.92605 | 0.92755 |
| TabM | 0.926061 | 0.92636 | 0.92751 |
| TabR | 0.925350 | 0.92528 | 0.92645 |

As you can see from the table, I tried almost 10 different NN architectures, as I was convinced that would be needed to make up for the lack of feature engineering in my models. Unfortunately, I couldn't get either `RealMLP` or `PyT-trompt` models to work within the ensemble. They contributed to better CV scores, but public LB scores (and later private LB) went down when they were added to the group.

I couldn't get AutoGluon to produce anything useful, possibly because I started relatively late and used only the original data without any feature engineering.

I am curious if anyone else had trouble with overfitting at the ensembling stage. Normally that's not an issue with a dataset of this size and with models that were consistent in terms of fold splits, but here I was often getting 0.9289 CV scores that scored much lower on public LB.

In the end I picked my best and 4th best submissions, and that always feels good.
