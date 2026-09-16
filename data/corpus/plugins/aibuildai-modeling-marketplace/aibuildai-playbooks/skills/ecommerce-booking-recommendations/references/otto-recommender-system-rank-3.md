# 3rd Place - Using Only Rules Achieves LB 0.590!

Competition: otto-recommender-system
Rank: #3
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/383013

# Team G & B & D & T
It was a pleasure to work with @titericz @benediktschifferer @theoviel . We each made individual models and then ensembled all our work together by adding the ranks of each of our predictions per user target type. Below I describe my individual single model. My teammates will describe their work in their own discussion posts. You can read about Theo's LB 0.6029 model [here][5]! You can read about Benny's LB 0.601 model [here][8]!

Most of my solution was made public in my notebook [here][3] and discussion [here][2] during the competition. There were only 3 significant ideas missing from my public work. Let's discuss how to boost my public work to LB 0.601 single model. (UPDATE: All code published to GitHub [here][6])

# How To Score LB 0.601 Single Model
It was explained [here][1] and [here][2] that the best approach was "candidate rerank" model. My public notebook [here][3] shows how to achieve LB 0.575. And my discussion post [here][2] explains how to improve my public notebook by adding a GBT ranker model. Below are the three missing pieces labeled (1), (2), and (3) to achieve LB 0.601

# (1) Choosing Candidates
To build a "candidate rerank", we need candidates. Where do we get candidates? The easiest way is to get them from my public notebook. In the `def suggest_buys(df)` and `def suggest_clicks(df)` function, the last lines are

    top_aids = [aid for aid, ct in aids_counter.most_common(20)] 
    return top_aids

To generate 50 candidates, we change 20 to 50 as in

    top_aids = [aid for aid, ct in aids_counter.most_common(50)] 
    return top_aids

# (2) Choosing Interaction Features
We now have 50 candidates per user from our public notebook above. Next we need to make features for our reranker model. The strongest and easiest way to make features is extract our co-visit counts by changing the last two lines of my public notebook to the following:

    top_counts = [ct for aid, ct in aids_counter.most_common(50)] 
    return top_counts

When we merge these counts to our candidates, we now have an interaction feature for each user item pair. To make more interaction features, we can extract the counts for each co-visit matrix individually. For example, imagine that we have 3 co-visit matrices named covisit2, covisit3, and covist4. Then one by one, we extract each covisit's counts:

    EXTRACT = ['covisit2']
    aids_counter = Counter()
    if 'covisit2' in EXTRACT:
        aids = list(itertools.chain(*[covisit2[aid] for aid in unique_aids if aid in covisit2]))
        for a in aids: aids_counter[a] += 1
    if 'covisit3' in EXTRACT:
        aids = list(itertools.chain(*[covisit3[aid] for aid in unique_aids if aid in covisit3]))
        for a in aids: aids_counter[a] += 1
    if 'covisit4' in EXTRACT:
        aids = list(itertools.chain(*[covisit4[aid] for aid in unique_aids if aid in covisit4]))
        for a in aids: aids_counter[a] += 1
    top_counts = [ct for aid, ct in aids_counter.most_common(50)] 
    return top_counts

# Reranker Boost CV and LB +0.011
First we use the technique above to generate candidates and the technique above to extract covisit counts. Next we add some simple item and user features like counting the number of times an item is click cart or order. When we apply the XGB reranker described [here][2], our CV and LB will boost by `+0.011`. For example, the public notebook will boost to LB 0.586.

# Using Rules Only (without reranker) Scores LB 0.590
To score over LB 0.600, we create more co-visit matrices to boost the original notebook's LB score. My public notebook uses 3 covisit matrices and achieves LB 0.575. If we make 17 more covisit matrices, we can boost my public "rules only" notebook to LB 0.590 (new notebook published [here][4]). Then when we extract the covisit counts explained above, the XGB reranker will boost +0.011 and achieve LB 0.601

# (3) Twenty Covisit Matrices
Below are a description of my 20 covisit matrices. These covisit matrices are the secret sauce enabling my single XGB ranker model to achieve LB 0.601. The following variable names are from my new LB 0.590 notebook posted [here][4]. (Example code showing how to compute covisit matrices on GPU is [here][3])
* **top_20** - this covisit matrix is in my original notebook
* **top_20b** - all covisit pair counts are consecutive items. See code below.
    `df['k'] = np.arange(len(df))`
    `df = df.merge(df, on=['session'])`
    `df = df.loc[ (df.k_y - df.k_x).abs()==1 ]`
* **top_20c** - all covisit pair counts are `(df.k_y - df.k_x).abs()<=2`
* **top_20d** - all covisit pairs are carts/orders and forward at most 3 consecutive
    `df = df.loc[df['type'].isin(['carts','orders'])]`
    `df = df.merge(df, on=['session'])`
    `df = df.loc[ (df.k_y - df.k_x > 0) & (df.k_y - df.k_x <= 3) ]`
* **top_20e** - all covisit pairs are `(df.k_y - df.k_x).abs()<=3` and have time decay with
    `df['wgt'] = (1/2)**( (df.ts_x - df.ts_y).abs() /60/60)`
* **top_20f** - same as above but `(df.k_y - df.k_x).abs()<=6`
* **top_20_orders** - this covisit matrix is in my original notebook
* **top_20_buy2buy** - this covisit matrix is in my original notebook
* **top_20_buy2buy2** - use most recent 3 weeks data and only carts/orders. Apply time decay shown above.
* **top_20_test** - use most recent 3 weeks data. Only forward in time pairs. Use clicks/carts/orders to carts/orders. Add time decay
    `df = df.loc[df.ts >= LAST_3_WEEKS ]`
    `df2 = df.loc[df['type'].isin(['carts','orders'])]`
    `df = df.merge(df2, on=['session'])`
    `df = df.loc[ df.ts_y - df.ts_x > 0 ]`
    `df['wgt'] = (1/2)**( (df.ts_x - df.ts_y).abs() /60/60)`
* **top_20_test2** - use most recent 2 weeks data with time decay.
* **top_20_buy** - Limit to forward 2 hours. Use clicks/carts/orders to carts/orders. Apply time decay.
* **top_20_new** - Find cold start users in train. Pairs using only their first history item. Use clicks/carts/orders to carts/orders.
    `df['x'] = df.groupby('session').ts.transform('min')`
    `df = df.loc[df.x > train.ts.min() + TWO_WEEKS ]`
    `df['n'] = df.groupby('session').cumcount()`
    `df2 = df.loc[df['n']==0]`
    `df3 = df.loc[df['type'].isin(['carts','orders'])]`
    `df = df2.merge(df3, on='session')`
* **top_20_new2** - Find cold start users in train. Pairs using only their first history item. Use clicks/carts/orders to clicks/carts/orders. Apply time decay.
* **top_40_day** - Use only last week data. Forward in time. Clicks/carts/orders to carts/orders. Time decay
* **top_40_day2** - Use only last week data. Time decay
* **top_40_less** - Train users with less than 6 history and test users with less than 3
    `df = df.loc[df[COUNT]<THRESHOLD]`
    `df = df.merge(df, on='session')`
* **top_40_more** - Train users with more than 6 history and test users with more than 3
* **top_40_less2** - Use item pairs with first item before 2pm. Clicks/carts/orders to carts/orders. Time decay
    `df2 = df.loc[df[HOUR]<14]`
    `df = df2.merge(df, on='session')`
* **top_40_more2** - Use item pairs with first item after 2pm. Clicks/carts/orders to carts/orders. Time decay

# Fast Covisit Experiments With RAPIDS cuDF
To find the above 20 covisit matrices, i computed hundreds of covisit matrices and then computed local CV score. To make covisit matrices quickly, I used RAPIDS cuDF to make each covisit matrix on GPU in under 1 minute. Code to make covisit matrix is shown [here][3]. Matrices were made using Nvidia 4xV100 32GB GPUs.

# UPDATE: GitHub Code!
I published all 261 jupyter notebooks in my GitHub [here][6]. Specially we can review all the code used to generate co-visititation matices. And we can see the pipeline for building, training, inferring a GBT reranker model. Our team's final Kaggle inference submit notebook is [here][7]. My notebook to generate 100 candidates for reranker is [here][4]. By itself it scores 49th place LB 0.590!

[1]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/364721
[2]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210
[3]: https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
[4]: https://www.kaggle.com/cdeotte/rules-only-model-achieves-lb-590
[5]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/382975
[6]: https://github.com/cdeotte/Kaggle-OTTO-Comp
[7]: https://www.kaggle.com/code/cdeotte/3rd-place-team-g-b-d-t-0-604
[8]: https://www.kaggle.com/competitions/otto-recommender-system/discussion/386497
