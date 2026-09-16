# 4th Place – Residual XGBoost + Meta NN + Hill Climb Opt

Competition: playground-series-s5e10
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s5e10/writeups/4th-place-residual-xgboost-meta-nn-hill-clim

**First**, I would like to express my sincere thanks to the Kaggle team and the Playground Series organizers for this valuable learning experience.
A big thank you as well to all participants for sharing ideas and notebooks.

I’m especially grateful to @cdeotte for the helpful information he always provides, and also to Masaya Kawamata, Mikhail Naumov, Kirderf Aliffa Agnur, and others who published their great notebooks — they were very inspiring and useful references during this competition.

I would also like to sincerely thank my teammate @ravi20076 for the great teamwork, thank you so much 

**Approach**

Our approach in this competition was to collect and learn from shared community information.
I started with notebooks by @cdeotte and valuable comments in discussions, especially from @broccoli_beef.

**XGBoost**

I experimented with several models — XGBoost, LightGBM, and CatBoost — and found that XGBoost gave the best results.
So, I decided to focus mainly on XGBoost, training residual models and using Optuna for fine-tuning.

My journey started with a CV of 0.05598 after the first submission.
After applying the residual training technique inspired by @cdeotte , my CV improved slightly to 0.05594.
Then, I introduced feature engineering, which helped reduce my CV to 0.05590.
Using Optuna for parameter optimization brought it down further to 0.05588 after using

```
BEST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'max_bin': 518,
    'learning_rate': 0.018518760913887423,
    'max_depth': 7,
    'min_child_weight': 6,
    'subsample': 0.8043445574984056,
    'colsample_bytree': 0.632194196921175,
    'colsample_bylevel': 0.8258554264939971,
    'colsample_bynode': 0.8456510777534194,
    'reg_alpha': 0.15615008876432407,
    'reg_lambda': 0.9717727187629448,
    'gamma': 0.0048414273522507795,
    'max_delta_step': 0,
    'scale_pos_weight': 0.8129498984084946,
    'random_state': 42
}
``` 
@metamodels  params , thank you

**Used OOF and prediction as new features**

Next, I used OOF predictions as new features.
I trained 14 XGBoost models, all using the same base features (my own delta features) plus additional features oof who shared by other participants.

**Neural Network**
Finally, I trained a Neural Network meta-blender +14 from our XGB and +7 oofs  and  using 17 different seeds  , reaching CV = 0.0558347.

When I moved to the Neural Network meta-model, I again included these external OOF features along with my stacked model outputs.

**Hill Climbing**

Then, I used hill climbing (with tolerance = 1e-9, inspired by @cdeotte method [hill-climbing](https://www.kaggle.com/code/cdeotte/gpu-hill-climbing-cv-0-05930)) to fine-tune the weights, which improved my CV to 0.055821.

In the final stage, I applied blending between this 2 results  my best models

This consistent alignment between CV and lb score gave me strong confidence in my final solution.

At the very end, after exploring @AnthonyTherrien’s notebook, I also experimented with hill-climb weighting (1.2 to hill-climb and 0.6 to nn ) , which slightly enhanced the final private score.

**Conclusion**

This competition was an incredible learning journey.
By experimenting with different approaches — from feature engineering and residual training to model stacking, neural network blending, and hill climbing — I was able to gradually improve my CV and achieve a solid alignment with the private leaderboard.

I’m truly grateful to the Kaggle community for the openness and collaboration that make these challenges so valuable.
Finishing 4th place was very rewarding, but the real value came from the knowledge gained and the chance
