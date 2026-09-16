# Solution Sharing

Competition: facebook-v-predicting-check-ins
Rank: #10
Source: https://www.kaggle.com/c/facebook-v-predicting-check-ins/discussion/22078#126222

Thank @Jared  for creating this topic, and thank everyone here for sharing your method.

My laptop had a hardware failure in the last week and I lost two sets of predictions that I could have use for my ensemble model. But anyway, here is my model of my best submission:

**Base models**

1. xgboost model 1 on 25*80 grids

    features: x, y, accuracy, hour of a day, day of a week, week of a year, year

    exponentially decaying sample weights

    expanded grid border in training (or border augmentation)

    3 grid offsets

    base model public LB score: 0.597 with 1 grid offset, 0.601 with 3 grid offsets combined


2. xgboost model 2
    
    all the same as model 1, except that it used "sin", "cos" transforms on the day of week feature

    3 grid offsets

    base model public LB score: 0.598 with 1 grid offset, unknown with 3 grid offsets as I initially planned to have 5 grid offsets if my laptop didn't fail me.


3. KNN from the [kaggle script](https://www.kaggle.com/lscoelho/facebook-v-predicting-check-ins/modified-grid-knn/run/288176)

    base model public LB score: 0.583

**Ensemble**

Soft voting of top 10 predictions from each model.

Final score: 0.607

I think the diversity between xgb model and KNN model played an important role in improving the score of the final ensemble model. I would also consider adding random forest, extra trees, and even neural networks to the final ensemble because they worked as good as KNN in my early testing of various models.

Although with more careful parameter tuning my model could still get a better score, I don't think it would get anywhere close to 0.620, so I'm really looking forward to @tvdwiele and @Markus sharing their awesome methods.
