# 19th Place Solution

Competition: child-mind-institute-problematic-internet-use
Rank: #19
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552513

**Thoughts**

I have heavy feelings after it is finalized. I mean, obviously I'm not smart enough to compete in serious competitions. Solving a difficult task on clean data, with the huge amount of famous skilled folks in the top LB, is pretty challenging even for strong professionals. The only chance to get gold for me is highly shakeable games like this one. I realized it after a few of my first competitions in the past. And by the background's violet color in my profile picture, I am symbolizing that, honestly, I have no hope for reaching Master tier.

I would be happy to surrender and not try again, but here is the quintessence of forming addiction. A person will leave the game if he can't get close to the desired result for a long time, he will also quit the game if he gets what he wants and be satisfied with it. But if the environment leads the game to the fact that a person **almost** achieves what he wants, a person will try again and again in this vicious cycle, comforted by the anticipation of success. This is how it works, and I would really like to reduce the time spent on Kaggle, but unfortunately it seems to me that I will become even more addicted to it.

**Baseline**

I joined lately but have watched this competition since the very beginning. On the one side, I was lucky enough to choose the right strategy; on the other side, I didn't bring my effort to victory. I described my plan for solution development in one discussion [here](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/discussion/551850#3073586); you can find my post. The basis of solution is my public [notebook](https://www.kaggle.com/code/yekenot/cmi-piu-deeptables-nn-cv-0-482), Version 60, CV 0.482 | Public 0.448 | Private 0.459, around 60th place on Private LB, so it is a high silver medal just from fork and submit. I've not planned something grand, just tried to explore data by the model's response.

According to the plan, I've worked with main data only, reduced by missing target, with a tuned threshold ratio (more than 30% missing) for dropping high-NaN columns. So it has 21 features in total after dropping: 6 categorical and 15 numerical. I've tried to fill remaining numerical NaNs with `SimpleImputer(strategy='median')`, linear interpolation, and `KNNImputer`, but the best results are achieved by `IterativeImputer(max_iter=19)` - this was something new to me. `MinMaxScaler()` was used for scaling. Categorical NaNs were filled automatically by the model's internal option `(SimpleImputer(strategy='constant'))`. I will not describe all the parameters of the model; you can explore it in the notebook. It's tuned DeepTables NN, pretty simple as you can see, with moderate dropout rates and an LR scheduler, nothing fancy here. But it worked very well, as in my other experiments, and produced an even better score after applying the global maximum search for the threshold optimization procedure. This may be considered a good single model but not the best, I think.

**Ensemble**

For the 19th-place ensemble, I just forked this NN's stuff and added a couple of GBDT models, untuned CatBoost and LightGBM(only adjusted `"reg_alpha": 3.1`), with early stopping 100 rounds. Same folds, all data preparation the same as in the NN's pipeline. Global maximum optimization for the thresholds was included. As a result I received 3 models with close CV performance. The CV of GBDT is a little higher as expected, and this could be a better choice to fit an untuned CatBoost as a single model instead of NN, although I haven't checked this yet. For the ensemble, I applied the `majority_vote` function from other notebooks to the rounded predictions of these 3 models to get Public LB 0.458 | Private LB 0.469 | >2000 places jump due to shake and 19th-place terrible silver medal in two steps from the gold. That's all I can say. Thanks for your attention.

Sincerely yours,

Forever Expert
