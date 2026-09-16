# #8 private 6 public approach- Simple ensemble and probing

Competition: playground-series-s3e19
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s3e19/discussion/428368

Hello all,

I wish to extend sincere thanks to the Kaggle team for this episode. It was indeed a great experience with lots of takeaways and learning that we can use outside of Kaggle too. I also wish to thank others participants for their generous contributions with special mention of the below users-

**Acknowledgement**-
1. @paddykb - his R-GAM notebook formed the base of my work for this challenge, his post-processing and levelling trick was key to my score in this challenge
2. @ymatioun - his insinuation about post-processing helped me reduce my public LB score from 25.8 - 6 in few minutes 
3. @onurkoc83 and team - their contributions to the assignment also formed a base for my work in the challenge
4. @oscarm524 - I was inspired to diversify my models in the assignment with the [relevelled kernel](https://www.kaggle.com/code/paddykb/ps-s3e19-relevel-oscar-s-notebook) by @paddykb 
5. @siukeitin - I took the macro-economic variable component from his [discussion post](https://www.kaggle.com/competitions/playground-series-s3e19/discussion/423725)

**Features used**-
1. Date features - I reused most of the features from my [date feature class](https://www.kaggle.com/competitions/playground-series-s3e19/discussion/423645). This included trigonometric features for relevant date-parts (sine-cosine features) an weekday indicator. I had mentioned in my [EDA and baseline work](https://www.kaggle.com/code/ravi20076/playgrounds3e19-eda-baseline) that this is an important feature 
2. I used a binary column for Dec26-Dec31 across countries 
3. I used GDP as mentioned in the above section 
4. I also used a holiday indicator as mentioned in my class 
5. I used a binary indicator for Japan between May6-9. I believe another high scoring solution has used it too (I could not mark the post here though). I found this to help me slightly in my CV results. 

**Model structure**-
1. I used a 2 stage model approach to keep the process diversified and ensure I try and blend different and complementary models to capture the signal in the data efficiently. Base models included-
a. LightGBM
b. XGBoost
c. CatBoost
d. GAM
2. I tuned the standard ML models (tree models) akin to my baseline work, with Optuna. Additionally, I calculated CV scores across 2-periods, from Jan-Mar and Apr-Dec to try and replicate the public private LB split. I think several public notebooks also used this strategy. 
3. I used a custom eval metric to stay within the requirements of the organizer's metric directives. Rather than using the MAPE metric, I built my own function/ class for the evaluation and embedded it with the tree models. This helped me tune and infer in line with the competition. One may refer to the below links to understand how to do so. In my opinion, developing a custom Loss Function would have also helped me more.
a. https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
b. https://m-e-gorelli.medium.com/pass-a-custom-evaluation-metric-to-lightgbm-65ef062415ad
c. https://stackoverflow.com/questions/69137780/provide-additional-custom-metric-to-lightgbm-for-early-stopping
d. https://mljar.com/blog/catboost-custom-eval-metric/
4. I relied on the [R-GAM](https://www.kaggle.com/code/paddykb/ps-s3e19-gammy-sales) notebook almost entirely (barring the surprise). I built a python equivalent of this notebook privately, but did not feel the need to use it as the results were almost the same. I adjusted a few weights across countries to probe and check the impact on the LB. The impact was small and sometimes worked for me too. Reference cell to edit from the original work is as below-
```
pred_probe = round(pred * case_when(
            country == 'Argentina' ~ 3.372,
            country == 'Spain' ~ 1.600,
            country == 'Japan' ~ 1.394,
            country == 'Estonia' ~ 1.651,
            country == 'Canada' ~ 0.850)))
```
5. I tuned my tree models just once (the day before) to submit my final models. I relied on features rather than tuning this time

**Model blending and post-processing**-
1. I blended my tree models using Optuna as mentioned above. I also fiddled with the weights manually at submission to probe and tune with the public LB score. I used post-processing here in line with the [GAM work](https://www.kaggle.com/code/paddykb/ps-s3e19-gammy-sales)
2. I blended the Optuna results with the GAM work to engender my final structure
3. I post-processed my predictions with [rounding](https://www.kaggle.com/competitions/playground-series-s3e19/discussion/425973)
4. I found that adding 1 to some of my submissions improved my public score a bit. I took this risk and it paid off. This was [elicited yesterday](https://www.kaggle.com/competitions/playground-series-s3e19/discussion/428123) as well. 
5. My final model submission was heavily weighted towards GAM and moderately weighted towards the blend of tree models. My model weights and final blending relied on the public LB score.

**Improvements**-
1. I should have looked into Ridge Regression. I myself posted about its efficacy in the past challenge and did not use it herewith. I think this could have boosted my score a lot
2. I should have paid more attention to holidays. I treated them commonly across countries while other top approaches treated them individually and attributed more effort in this regard
3. As mentioned above, a custom objective for the tree models perhaps would have helped my model process.

**My takeaways**-
1. Simple approaches have a lot of merit in time series at least. One needs to build good features and rely on them. 
2. I need to be more alert to the CV and LB relation. In this challenge, the public LB elicited insights about the final results too, especially after relevelling
3. Using the metric directed by the organizer in our models in full to develop and infer have more benefits than using the provided metrics. I shall strive and use this in other areas of work as well. 
4. Using past experience from TPS-Sep22 as a base for this challenge was helpful. 

Happy learning and best regards! 
See you in Episode 20 and good luck for other competitions too!
