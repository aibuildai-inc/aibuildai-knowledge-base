# 1# Winning solution

Competition: playground-series-s3e14
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e14/discussion/410627

Hi,

I would like to express my deepest gratitude to the organizers, winners and the community who made this competition.

# **Introduction:**
The competition attracted a record number of participants.
The community was extremely active. The discussions were fruitful creating a collaborative environment that fostered learning and rapid score improvement. The number of gold/silver notebooks and discussions speaks for itself.

It was not an easy journey for me. It took me almost 5 months of polishing tabular competition pipelines. Few times I was first on the public leaderboard and did not survive shakeups. Few more times I chose the wrong submissions. It burned my fingers but it made me understand: "if the approach is solid you don't need to submit 5 times a day.


# **The solution:**

## **Context:**

From the very beginning of the competition, I found out that the models will be limited in performance by looking at one of the most important features. I [posted](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/409417) the discussion here, showing how much variance was synthesized from the original dataset. No matter how good the ensemble was going to be, it would not give a winning edge I got.

## **Early discovery:**

At a very early stage of model validation, I found that the model struggled with some obvious predictions.
Most of the common error was `yield` == **1945.53061**. You will see it in the [attached work](https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e14-2023-first-place-winning-solution).
I noticed that the values for `fruitset` and `fruitmass` **0.335339	0.233554** which were the same for the train and origin were always `1945.53061` no matter what. So I assigned `1945.53061` to those predictions and saw how oof behaved. As you can assume they did fantastic.

Then I found a couple more examples, you will see them in the first phase of correction in the notebook.

## **Automating the process:**

I figured out that the manual search was not scalable, so I came up with the whole script which ran through common `fruitset` and `fruitmass` values for both origin, train and test. **2073** total samples in test.
But what values should have I assigned? Well, easy enough, I ran through all unique `yields` **776** iteratively and saw which one were better. Though it improved the oof, it did not perform well on the leaderboard, so I assigned the value from the original dataset and it was a nuke bomb (further I continue on this analogy).


## **Strategy:**
I decided to play slow. The phase 2 postprocessing included waterfall of different submissions in terms of risk.
I named them according to this [classification](https://en.wikipedia.org/wiki/International_Nuclear_Event_Scale) where **anomaly** was just correction of **1945.53061**.
Then you can go deeper with these lines of code:

```python
if len(dsp_train) > 2 and len(dsp_test) > 1:
	if not dsp_origin.empty:
		orig_value = dsp_origin['yield'].values[0]
		tr_idx = train.loc[train.fruitset.eq(txt[0]) & train.fruitmass.eq(txt[1]), 'pred'].index.tolist()
		te_idx = test.loc[test.fruitset.eq(txt[0]) & test.fruitmass.eq(txt[1]), 'pred'].index.tolist()
```
		
`len(dsp_train) > 2 and len(dsp_test) > 1`: means there should be at least more than **two common records** in **train**, **one** in **test** and **one** in **origin**.

**Serious Incident** : > 4 > 3
**Zone With consequences**:  > 3 > 2
**Dangerous Zone**:  > 2 > 1
**Ground Zero**:  > 1 > 0 (not submitted)

The risk was great to fail terribly on the private leaderboard that's why you will see _suffix **_private_600** in the submissions screenshot later on.


## **@adaubas helped the strategy:**

One submission and the second place, what a shot, right! It was @adaubas who generously shared the config and the [work](https://www.kaggle.com/code/adaubas/ps-s3e14-stacking-leastabsolutedeviation-reg) later on.

How did it help? Well, it could have seemed natural that I jumped to the second place right after the discussion once the config got into the trends.
So I even posted the teaser with my close result config with almost the same parameters (sneaky me...).

## **Stacking Masterclass:**

1st level models included 8 different oofs. They could be found [**in this dataset**](https://www.kaggle.com/datasets/sergiosaharovskiy/s3e14-oofs).

Here is the model pipeline and works of those who helped the winning solution:


[high resolution](https://monosnap.com/file/8D0U5butFrJnQhKEio4kCUfv4LZGkp) or `ctrl + scroll up` to zoom in.

## **LAD vs scipy.minimize**:
It was nice to learn about LAD from @adaubas.
What I found out is that I used the same backend with very similar results.
But for the sake of code simpicity and slightly better performance I chose LAD.
Here is the comparison:



## **Validation:**






# **Outro:**

## **Tribute to @naganohikaru**:
The effect of the above postprocessing was immense. Various model blendings gave floating point improvement when at the same time the postprocessing above gave an integer sometimes.
You saw me jumping on the leaderboard. I waited the next move of @naganohikaru and send the bomb understanding the risk.
 
I stopped at the **Dangerous Zone** since I did not see any improvement from my opponent anymore, mitigating the already high risks on the private leaderboard.

## **Taking the risks with submissions**:
I chose two best public scoring submissions (they happened to be my best private). The motivation was simple, I realized that the portion of Alex models in the final stack was not big, and the only way I could win was to take the full risk. Otherwise, I would compete against hundreds of the same variety of ensembles.

## **Acknowledgements**:

p1	     @zhukovoleksiy [notebook link](https://www.kaggle.com/code/zhukovoleksiy/ps-s3e14-simple-eda-ensemble)
p2, p5   @paddykb       [notebook link](https://www.kaggle.com/code/paddykb/ps-s3e14-flaml-bfi-be-bop-a-blueberry-do-dah) 
p3, p6   @yzokulu       [notebook link](https://www.kaggle.com/code/yzokulu/ps3e14-simple-lightgbm-regressor-with-cv) 
p4	     @tetsutani     [notebook link](https://www.kaggle.com/code/tetsutani/ps3e14-eda-various-models-ensemble-baseline) 
p7/p8/p9 @adaubas       [notebook link](https://www.kaggle.com/code/adaubas/ps-s3e14-stacking-leastabsolutedeviation-reg) 
@mattop [post_processing](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/40732)

## **Icing on the cake**:
What would be my last move if my opponent above found some miracle in the end?

You are right - the **ground zero** submission, here you go:


## [**The winning solution code**](https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e14-2023-first-place-winning-solution).
##  [**Dataset**](https://www.kaggle.com/datasets/sergiosaharovskiy/s3e14-oofs).
