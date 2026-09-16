# 41st place solution

Competition: ieee-fraud-detection
Rank: #41
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111341

Thank you to my teammates Rob Mulla @robikscube , @raddar , and Yang @ljjsfe .

### Initial findings

It starts with Rob Mulla and Yang independently submitting stuff when the competition first opened. Yang had independently achieved 0.927 private LB. Then when CHAMPS ended, Rob and I decided to pursue the IEEE competition more fiercely.

I had noted that the D-features leaked `user_id` privately to Rob. Just using the D-features alone, however wasn't enough, and so I saw many users flip from `isFraud=1` back to `isFraud=0`. I reached out to Raddar because he is the king of reverse-engineering (I knew of his strength from ELO and Home Credit post-processing). After I explained to him the D-features, he knew exactly how to exploit this and was on-board to team. The simple problem we were missing was also to enforce the user should have the same `card1-6, P_emaildomain, addr1-2`, etc.

### Raddar post processing

After raddar did that, he also began tinkering to enforce other rules, like C features should be monotonous. By simply post-processing people with `isFraud=1` and people with `isFraud=0` to 0.93 and 0.000001 respectively in the test set, he brought our private LB 0.927 -&gt; 0.9297 (And Rob and I were very happy to have raddar on our team when we saw our public LB jump so high)

Yang reached out to join us on September 21, and we accepted given his strong performance on Home Credit.

### LB Probing

I then began to LB probe based on the `user_id` raddar had created. Basically, if the `user_id` was only in test and not in train, and if it had a high prediction like &gt;= 0.15, I changed him all to 1's. If LB improved, I kept it; if it worsened, I changed him to all 0's. I continued this every day until the competition ended; this ended up netting us a decent amount of AUC

We coded up some features related to `user_id` (like number transactions, max TransactionAmt, mean isFraud, etc.) and chose them if they improved the CV.

### Adversarial Validation, Pseudolabels, and train on all data blindly

We began to get scared of train/test difference in distribution since many were saying there would be a shakeup (and it was evident that many features had different distribution in test). Therefore we fit a feature 1 at a time to predict train vs private test (this technique is called adversarial validation). From now on, we had 2 branches: one model that kept all of the features, and one model that only kept features that had &lt;= 60% ROC AUC in adversarial validation.

Additionally, after having so many LB probes, we began to train on full train set + use the LB probes as pseudolabels from the test set. The number of iterations we used was roughly (number of iterations on 5th fold) * 1.1

### Corey post processing

In the last few days, I modified `corey_user_id` (different from `raddar_user_id`) to be just based on the D intercept and then group by `card1-6, ProductCD, and P_emaildomain`. Whenever I tried any LB probes based on my ID, it worsened LB. So I made a modification: IF in test set there was a `raddar_user_id` with `isFraud=1`, then for the `corey_user_id` mapped to that row, change all predictions to `isFraud=0.99`. Do the same if `raddar_user_id` had `isFraud=0`, except change predictions to 0.0002. Why? Because I had more confidence in `raddar_user_id` (he was grouping by more features, and was enforcing C monotonicity) and my `corey_user_id` was more shoddy. However, doing this post processing gave us a jump of like 20 places.

### Final submissions

In the end, we have 2 submission:
1. The first is a giant model LGBM trained on Rob's, my, raddar's, and Yang's features such that ROC AUC in adversarial validation is &lt;= 60%. We bagged it thrice, trained blindly on full train data + pseudolabels from `raddar_user_id` and `corey_user_id` and then did post processing similar to how I described it above. This model gave 0.955 public LB, and 0.9339 private LB
2. The second is a 60% * (50/50 blend between Rob's old model that used all features and Yang's model that used solely his features, with post processing) + 40% * (first submission). This had 0.9586 public LB and 0.9363 private LB.

### Post-competition thoughts

We thought there would be more of a shakeup, so we are sad not to achieve gold (though we did rise 18 LB spots). There are some problems we could have pursued: for example, using D3, D4, D5, or V307 to make our `user_id`'s better. Additionally, basically Rob was the only one training LGBM, so we had no big diversity from other models/feature sets apart from Yang's early model. Next time, we will work harder ;)
