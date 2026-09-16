# 16th place solution

Competition: prostate-cancer-grade-assessment
Rank: #15
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169163

**Thank you very much to organizers, participants and my teammates** @ryunosukeishizaki @rinnqd for such competition.

We can name our solution as **"zero public LB to hero private LB"**.  In public we could even get a bronze and in private we are in top 20 teams. It's not a lucky submission because we have a lot of them and success points are real.


1. **Removing noise** (marks, duplicates) based on [this Zac Dannelly](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/151323) and [this Leonie](https://www.kaggle.com/iamleonie/panda-eda-visualizations-suspicious-data), and also my own manual clean up

2. Training efficientnet-b0, b2, b4 and mixnet-xl on **different tiles sizes** 36x256x256 (level 1) =&gt; 49x256x256 (level 1) =&gt; 64x256x256 (level 1) without regularization and with high (dropout 0.4)

3. **Combining cleaning dataset training and raw data**

4. Blending based on local **CV weights** w = w / np.sum(w)



5.**Trust your local CV** and train stable models!
