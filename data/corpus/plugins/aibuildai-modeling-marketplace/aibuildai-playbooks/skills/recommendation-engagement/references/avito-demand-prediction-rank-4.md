# 4th Place Solution

Competition: avito-demand-prediction
Rank: #4
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/59881

**Edit/update**: please also see [@jtrotman][1]'s writeup in the responses below! :) 

Thank you @ Kaggle and Avito - this competition was just awesome. There were so many interesting facets of the data to explore, and this was a rare competition where I wish I had joined earlier and spent even more time with it instead of getting sick of the data by the end. And a huge, huge thank you to my team [@neongen][2], [@gphilippis][3], and [@jtrotman][4]. This was a great and inspiring team that I feel very lucky to have been a part of.

Our solution is a lgbm stacker trained on a bunch of good base models and a chunk of our strongest features. Nonlinear stacking and including features for stacking definitely helped. My teammates can share more about that framework and all of the different models that went into the stack (lgbs, nns with lstm, sparse nns, weaker models like ridge), but my part of this thread will focus most on my biggest contribution to the result - a single lgbm model that scores .2175 public /.2213 private and the features that go into this model.

**Hyperparameters**

Hyperparameter tuning was not my main focus, but I found that very deep trees with a low learning rate worked quite well. Here are the final parameters:

    lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'num_leaves': 400, 
                'colsample_bytree': .45
              }

**Features**

Now to get to the good part and my focus in this competition - features. I ended up with about 800 tabular features (original + engineered), along with 100k tf-idf features for description and title + param_1. Here are the tf-idf settings:

    TfidfVectorizer(stop_words=stopwords.words('russian'), 
                             lowercase=True, ngram_range=(1, 3),
                             max_features=50000,
                             sublinear_tf=True)

I'd summarize my tabular features by saying that they try to extract as much information as possible using price and the text fields. At first I only used very simple image features like color channels and size, and the image features I added at the end only gave about a .0003 boost.

My core feature ideas:

**Price statistics**: 

It seemed clear early on that price information should have a very import impact on the target. So I wanted to extract price distribution information / statistics on pretty much every basis of aggregation I could come up with (e.g. by category, by category and city, by user_id, by image_top_1, etc.). For a bunch of different aggregates, I took these stats across all records  (with duplicate item_ids dropped):

 - 20th percentile
 - median
 - max
 - standard deviation
 - skew

Then computed some relative price numbers for each post record with respect to the corresponding aggregate -- this measures how expensive or cheap the item is relative to the grouping
- row / 20th percentile, row / median, row / max 

These computations are at the core of my FE, including for the ideas just below. 

**Granular textual aggregates on price**: 

I believe my most important discovery in this competition was that applying my price statistics features to very specific text groupings worked extremely well, and these features are at the core of my model. Taking price / median price for each post's title alone gave me a big boost. I hypothesize that this may be a way to partially reverse engineer keyword search rankings (e.g. search sorted by price) and that search rankings heavily impact views and therefore deal_probability. Image that you're an avito user looking browsing for items - you search for a few specific keywords (e.g. "red bicycle"), sort by price, and start looking at many of the cheapest options. I've extended the idea to add more value in a few different ways that try to reduce the number of distinct groupings while still being granular enough to capture this level of signal -- I run price statistics on all of the following:

 - title_noun aggregates: I extract all nouns from each title, normalized them, removed duplicates and sorted them alphabetically, and then used the resulting key as a basis for aggregation
 - title_noun_adjs aggregates: same as above, but adding adjectives as well 
 - title_cluster: using title tf-idf features, I run SVD with 500 components and form 30,000 k-means clusters on these components. k-means is very slow, so I used mini-batch k-means and limited it to 500 components / 30k clusters even though I think more of both may have performed better.
 - text_cluster: same as above, but based on a concatenation of title, description, and the param fields. 

**User semantics**:

When you build ML models, it's easy to get stuck in a row-based world and lose sight of the many interesting relationships between the different rows. Normal aggregate feature engineering like column statistics help with this, but I got to a point beyond that where I realized there was still more to be done with the text fields to help capture user characteristics.
    
Coming off of talkingdata where it saw some success, I had the idea to use matrix factorization for this. So for users in train/test, I concatenated their text fields across all rows (title, desc, params), ran tf-idf -&gt; 300 component SVD, adding these components as features. A performance tip here is to use hashingvectorizer like below rather than straight tf-idf, since it is significantly more memory efficient (at the cost of a bit of accuracy). Set n_features higher than the number of features you really want, since there will be collisions.

    print('Applying hash vectorizer then tf-idf to text')
    
    print('Hash Vectorizer')
    hv = HashingVectorizer(stop_words=stopwords.words('russian'), 
                           lowercase=True, ngram_range=(1, 3),
                           n_features=200000)
    hv_feats = hv.fit_transform(user_text_df['all_text'])
    
    print('TF-idf transformer')
    tfidf_user_text = TfidfTransformer() 
    tfidf_user_text_feats = tfidf_user_text.fit_transform(hv_feats)

Another nice way to capture user-based information from text is with aggregations on meta-textual features like percentage of caps in title, etc. I took a bunch of meta-textual features and computed the same aggregate statistics that I did for price, across each user_id. For example, one of my top 30 features was the rather unfortunately named (by my conventions) 

    "pct_caps_description_pct_user_id_median:pct_caps_description"

This means taking the specific row's percentage of caps in description and dividing it by that user's median percentage of caps in the description. Maybe I should have called it "user shoutiness relative to their norm" instead.  

**Image Features**

For a big part of the competition I thought the images were mostly a red herring. My worldview was very focused on exploiting the possibilities of textual information and thought its relationship with search rankings had a much more dominant effect than the images would. It was very interesting to hear that people were seeing significant gains from good image features, and I think this is the area where I would try to improve my model more if given more time. I used a few very simple features (color channels, size dimensions), and some nice features prepared by my teammates (they could explain in more detail) that gave about a .0003 boost to my model as the final feature addition --

 - Image blurness
 - Color histograms - SVD components and some statistical features
 - NIMA - activations, score, stds, and score + std

Aside from features mentioned, I used a bunch of other miscellaneous features similar to those seen in kernels - simple meta-textual features, lat/long + lat/long clusters, avg days a user's post is active and similar stats from the periods data.  I did clean and lemmatize title and description before applying tf-idf and some of the other text processing mentioned above, but I believe it made a pretty minor difference. 
 
In addition to lgb I used these features in a few other models, e.g. a neural net that scores .2197 public. Architecture is very similar to what you can see in kernels - biGRU applied to title and description, category embeddings, numerics concatenated and passed through 2 dense layers. It was difficult to close the gap between lgb and NN with this feature set, I think largely because there were many null values as a result of aggregate stats on unique occurrences, and these aggregate stats were among the key features to the lgb. I tried some different imputation strategies, but nothing worked all that well (although I will say that imputing price in a smart way, whether by a groupby-average on some category combinations or by a dedicated price prediction model, improved the NN score by about .001). 

**Some Comments on Workflow**

I want to end on a few comments about workflow that I hope may be useful. What works well in one competition might not be ideal for another, but maybe it's helpful :) I've already written about some of these approaches here: https://www.kaggle.com/c/avito-demand-prediction/discussion/56986#330023

If you do a lot of feature engineering (especially with categorical aggregates), you are likely creating what amounts to a miniature relational database. You don't have to go as heavyweight as creating an actual database, but I found it helpful to create a bunch of feather files for storing aggregate features then merge all the ones I wanted back into the main data to build models. For example I had many feature files like "global_city_features.ftr" that I would merge back into the training dataframe using city as a key. Storing features in a relational manner saves you processing time (run FE once, not every time you model), saves you disk space, and gives you a lot of flexibility for choosing which features to include in a model run (just don't merge them in if you don't want that set).

I used greedy forward feature selection - engineer a new set of related features, add them to my current best model, see if validation result improves. If it doesn't improve, leave them out of the model. This is a fast and fairly reliable (if not optimal) way of selecting for good features to include in a model.

I noticed early on that there was very little variance in RMSE between validation folds (&lt;.0002), so I felt pretty safe testing for feature contributions to my model by evaluating on only one fold at a time. This helped speed up iteration on feature sets.

I've already written a lot and my teammates will have additions to make as well so I'll stop there. If you're not sick of my blabbering and are interested, I'll likely write a blog post about my ideas / approach and share a link here once it's up. I hope this was a useful read, and happy kaggling!
  


  [1]: https://www.kaggle.com/jtrotman
  [2]: https://www.kaggle.com/neongen
  [3]: https://www.kaggle.com/gphilippis
  [4]: https://www.kaggle.com/jtrotman
