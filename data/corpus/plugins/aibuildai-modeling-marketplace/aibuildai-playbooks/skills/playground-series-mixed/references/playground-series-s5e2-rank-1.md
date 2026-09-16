# 1st Place - Single Model - Feature Engineering

Competition: playground-series-s5e2
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s5e2/discussion/565539

# Single Model Wins!
My favorite solution in a Kaggle competition is a powerful single model versus a large ensemble. I'm excited that a single model with creative feature engineering wins this competition! Although this was a weird competition with weird data, this was one of my favorite competitions because this was a tricky puzzle to solve that required lots of creative features! 

# Weird Competition Data
This competition's data was weird and unnatural as explained [here][1]. The techniques that were successful in this competition are not what we would need if we were predicting real backpack prices. However it is important to note that every technique used here is used in other real world models. So it is beneficial to learn these techniques.

# Final Solution
My final solution is a single model trained with 500 features using 1xA100 GPU 80GB. However a single model with only 138 features trained with Kaggle's 1xT4 GPU 16GB also wins first place. I publish this simple Kaggle T4 GPU solution [here][2].

# Feature Engineering
The key to success in this competition was running as many experiments as possible trying as many different feature engineering ideas as possible. To perform experiments as fast as possible, I used [RAPIDS cuDF-Pandas][4] as shown in my starter notebook [here][3]. In one month, I trained over 300 XGBoost models and tried thousands of different feature engineering ideas! My final solution keeps the best ideas. Below I list some of my favorite ideas from my final solution.

# Groupby(COL1)[COL2].agg(STAT)
Basic groupby stats are explained in my starter discussion [here][7]. We pick a column `COL1`, then pick a column `COL2`, then pick a `STAT` like `"mean"`, `"std"`, `"count"`, `"min"`, `"max"`, `"nunique"`, `"skew"` etc etc. (If `COL2` is a target column, we use nested folds to prevent leakage). Below are more advanced features!

# Groupby(COL1)['Price'].agg(HISTOGRAM BINS)
I had fun inventing this technique. I have never seen it being used before. When we `groupby(COL1)['Price']` we have a set of number for each group. 

Below we display histogram for the group `Weight Capacity = 21.067673`. We can count the number of elements in each (equally spaced) bucket and create a new engineered feature with this bucket count to return to the `groupby` operation! Below we display 7 buckets, but we can treat the number of buckets as a hyperparameter.

    result = X_train2.groupby("Weight Capacity (kg)")["Price"].apply(make_histogram)
    X_valid2 = X_valid2.merge(result, on="Weight Capacity (kg)", how="left")



# Groupby(COL1)['Price'].agg(QUANTILES)
After `groupby`, we can compute the quantiles for `QUANTILES = [5,10,40,45,55,60,90,95]` and return the 8 values to create 8 new columns.

    for k in QUANTILES:
        result = X_train2.groupby('Weight Capacity (kg)').\
            agg({'Price': lambda x: x.quantile(k/100)})

# All NANs as Single Base-2 Column
We can create a new column from all the NANs over multiple columns. This is a powerful column which we can subsequently use for `groupby` aggregations or combinations with other columns!

    train["NaNs"] = np.float32(0)
    for i,c in enumerate(CATS):
        train["NaNs"] += train[c].isna()*2**i

# Put Numerical Column into Bins
The most powerful column in this competition is `Weight Capacity`. We can create more powerful columns based on this column by binning this column with rounding!

    for k in range(7,10):
        n = f"round{k}"
        train[n] = train["Weight Capacity (kg)"].round(k)

# Extract Float32 as Digits
The most powerful column in this competition is `Weight Capacity`. We can create more powerful columns based on this column by extracting digits! This technique seems weird but it is often used in real life to extract info from a product ID where individual digits within a product ID convey info about a product such as brand, color, etc. (idea from @jordanbarker [here][6])

    for k in range(1,10):
        train[f'digit{k}'] = ((train['Weight Capacity (kg)'] * 10**k) % 10).fillna(-1).astype("int8")

# Combination of Categorical Columns
There are 8 categorical columns in this dataset (excluding numerical column `Weight Capacity`). We can create 28 more categorical columns by combining all combinations of categorical columns. First we label encode the original categorical column into integers with `-1` being NAN. Then we combine the integers:

    for i,c1 in enumerate(CATS[:-1]):
         for j,c2 in enumerate(CATS[i+1:]):
            n = f"{c1}_{c2}"
            m1 = train[c1].max()+1
            m2 = train[c2].max()+1
            train[n] = ((train[c1]+1 + (train[c2]+1)/(m2+1))*(m2+1)).astype("int8")

# Use Original Dataset which Synthetic Data is Created From
The following feature seems weird, but it is based on the idea that a product's price is based on `manufacture suggested retail price`. We can treat the original dataset that this competition was created from as the manufacture suggested retail. And this competition's data as the individual stores' price. Therefore we can help predictions by giving each row knowledge of the MSRP:

    tmp = orig.groupby("Weight Capacity (kg)").Price.mean()
    tmp.name = "orig_price"
    train = train.merge(tmp, on="Weight Capacity (kg)", how="left")

# Division Features
After creating new columns with `groupby(COL1)[COL2].agg(STAT)`, we can can then combine these new columns to make even more new columns! For example

    # COUNT PER NUNIQUE
    X_train['TE1_wc_count_per_nunique'] = X_train['TE1_wc_count']/X_train['TE1_wc_nunique']
    # STD PER COUNT
    X_train['TE1_wc_std_per_count'] = X_train['TE1_wc_std']/X_train['TE1_wc_count']

# Final Submission Code
I publish a simplified version of my single model code [here][2]! 

[1]: https://www.kaggle.com/competitions/playground-series-s5e2/discussion/564056
[2]: https://www.kaggle.com/code/cdeotte/first-place-single-model-lb-38-81
[3]: https://www.kaggle.com/code/cdeotte/feature-engineering-with-rapids-lb-38-847
[4]: https://rapids.ai/cudf-pandas/
[5]: https://www.kaggle.com/code/cdeotte/fast-gpu-hill-climbing-starter-cv-0-94-lb-0-94
[6]: https://www.kaggle.com/competitions/playground-series-s5e2/discussion/563743#3128871
[7]: https://www.kaggle.com/competitions/playground-series-s5e2/discussion/563743
