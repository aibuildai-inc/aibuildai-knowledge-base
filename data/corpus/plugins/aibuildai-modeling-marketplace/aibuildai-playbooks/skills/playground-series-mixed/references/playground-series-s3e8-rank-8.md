# 8th place solution (Flying Over the 1st place again)

Competition: playground-series-s3e8
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s3e8/discussion/392860

**Intro**:
Another week and another salute to the winners of the competition! 
This was painful to write, but @jcaliz cheered me up since the solution was not trivial.

**Preface**:
I was interested in gemology during my university days and was considering getting a GIA (Gemological Institute of America) certification to appraise precious stones. I even had plans to open my own pawn shop to sort diamonds not pandas dataframes. It's amazing how our interests and career paths can change over time. While my current focus is not in gemology at all, the past experiences and interests can still be valuable assets.

**Part 1. Fighting outliers**:
Take a look at these 3 illustrations where I tried to clip the outliers based on their reasonable price range (red boxes shows the areas where the `train` and `original` datasets are awfully different).







Once I fixed the values of outliers I got extreme boost on CV, but poor results on LB...
In order to understand why that happened, I decided to look at the best predictions and voila! It happened to be that all models already squashed the outliers so there was no reason to fix it.



**Part 2. The models were good already:**
I looked at public works written by @tetsutani and @eamonntweedy since their modeling part was somewhat similar to mine. One thing I found in common is that the models were good and generalized well from the very beginning. Though I did not really like the squashed predictions, so I tried to help the model myself. I ventured to imagine myself a certified Appraiser and start labeling the data manually.

**Part 3. Visualizations and perfect stone:**
No self-promotion intended, but I started looking into [When 4Cs Concept becomes an Art](https://www.kaggle.com/competitions/playground-series-s3e8/discussion/389465) topic more and more. I chose 'Ideal" cut stones of "D" color (best colorless), 'Flawless'  within a specified carat range from the train. The idea was to find the max possible value for a specified group of stones and compare it to the test predictions, if for some reason test predictions were higher - assign Q3 + (1.5 * IQR) from train to the test predictions. It gave +0.01 LB for 20 examples. When I decided to automate it a bit:

```python
def postprocess_clip_sub(best_sub: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Clips test preds max values based on the train statistics. 
    
        Args:
            best_sub: best ensembled submission at the moment.
            alpha: tunable parameter for calibrating the upper bound for predictions.
            
        Returns:
            best_sub: modified df
    """
    # LB +0.011 (574.92723 - 574.9163)
    # LB +0.2 tuned alpha
    
    sub = best_sub.copy()
    train = pd.read_csv('data/train.csv').drop(columns='id')
    test = pd.read_csv('data/test.csv').drop(columns='id')
    test_sub = pd.concat([test, sub], axis=1)
    
    bins = np.arange(0.19, 4.5, 0.035)
    train['carat_bin'] = pd.cut(train.carat, bins=bins, labels=list(range(len(bins)-1)))
    test_sub['carat_bin'] = pd.cut(test_sub.carat, bins=bins, labels=list(range(len(bins)-1)))
    for cl in tqdm(train.clarity.unique()):
        
        for cb in sorted(train.carat_bin.unique()):
            price_frame = train.query('cut == ["Ideal", "Premium"] & clarity == [@cl] & color == ["D", "E"]').query('carat_bin == @cb')
            
            if price_frame.shape[0] > 5:
                Q1 = np.percentile(price_frame.price, 25, interpolation='midpoint')
                Q3 = np.percentile(price_frame.price, 75, interpolation='midpoint')
                IQR = Q3 - Q1
                lower_bound = Q3 - (1.5 * IQR)
                upper_bound = Q3 + (1.5 * IQR)

                idx = test_sub.loc[test_sub.carat_bin.eq(cb) & test_sub.clarity.eq(cl)].id
                sub.loc[sub.id.isin(idx), 'price']  = sub.loc[sub.id.isin(idx), 'price'].clip(0, upper_bound*alpha)
    return sub 
```

**This algorithm gave me straight +0.2.**

**Part 4. Jumping to the first position on the public LB:**
Once I clipped my max predicted values, I came down to the min ones.
And this is how I found myself +1.1 right away.
The algorithm:
* iterate thru `carat` in train with sliding window with range (0.15)
* inside the range iterate thru `cut`
* inside the 'cut' iterate thru `clarity`
* inside the 'clarity' iterate thru `color`
* filter the train frame by this group of parameters and if there were more than 5 records found look for the same in the test predictions
* if the predictions were for some reason lower than the min possible train, assign min train to those records.

This basically did impossible with any average model pushing it straight to the **Top 10**.

**Part 5. Why it is painful.**
Winning a Kaggle competition can bring a sense of accomplishment and validation, similar to the rush of euphoria that comes with using drugs.

However, Kaggle competitions can also be incredibly challenging and frustrating. Participants may spend countless hours tweaking their models, only to find that they still fall short of the competition leaders. This can lead to feelings of disappointment and sadness, similar to the lows experienced by drug addicts during withdrawal.

In both cases, the highs and lows can create a cycle of addiction. Participants may become obsessed with winning Kaggle competitions, constantly seeking the next challenge to achieve that feeling of euphoria again. Similarly, drug addicts often chase the initial rush of euphoria, leading to a cycle of addiction and seeking out the next high. Be careful! Be save!

**Outro**:
Night changes many thoughts. it is a bit messy but I hope you found this write-up useful.
