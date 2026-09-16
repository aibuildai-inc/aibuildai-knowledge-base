# 22nd Place Solution

Competition: playground-series-s6e3
Rank: #22
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/22nd-place-solution

**Introduction**

So far my current rank is my best result in the PlayGround Series, so let it be a brief overview of my approach at least.

I saw many people challenging themselves to compete with several submissions a month or do Vibe Coding only. My personal adventure is competing with the only strongest single model on my side. Despite the obvious advantages of boosted trees algorithms, I find it not as fun to develop as neural nets because NNs are still considered weaker players and need to be enhanced.

Well, after the rise of the RealMLP model, as declared the best solo model in the previous month's competition, I asked myself how far away could I ride on its only power? I prefer simplicity and compactness, which is why it's not a story about building hundreds models and utilizing all possible knowledge about tabular learning - the reason I'll probably never win on LB. But it's okay, because having a craft, elegant, highly optimized, low-cost, and top-performance solution is actually a win, isn't it?

**Baseline**

I've started with the Trompt model to touch data to see what initial features affect NN's productivity and what engineered features improve it. Although it's novel and exotic in nature, Trompt NN is a good baseline because it requires very little hyperparameter tuning from data to data (PlayGround Comps) and it's very fast to train, receiving decent scores on few epochs. The reason I didn't dive into RealMLP instantly is because I suspected it could demand dozens of epochs to converge with hours of runtime - definitely not an option when validation of each engineered feature is on plan. Surprisingly, RealMLP was good with only 3 epochs for several minutes of runtime. This twist is a main reason why my RealMLP's experimenting notebook has over 1000 run versions.

**Feature engineering**

Relying on my experience with NNs, I was pretty aware of which features I could boost. After setting mandatory steps such as a adding categorized numerical features set, I was going through AI assistant suggestions and public notebook research. What do you need to keep in mind during feature engineering with NNs? Not all features that significantly improve the performance of GBDTs are even compatible with NNs, not just redundant. Moreover, in each group of features, such as arithmetic interaction or digit extraction, you need to evaluate feature interactions with one another in the group, adjusting changes in a parallel manner. Finally, I came up with 22 new features, which boosted both CV and LB scores. The most interesting and valuable part is using `KBinsDiscretizer` with `'quantile'` and `'kmeans'` strategies stacked together. I'm not used original data at all and might say I don't benefit from categorical combinations or target encoding, presenting these in a public notebook just as a demo.

**Model tuning**

It seemed to be a quite boring idea to tune dozens of parameters manually, especially when Optuna's search might be enabled, but I've loved it very much. I didn't actually understand initially how RealMLP was built and how it processes data internally. Touching every single parameter allows me to intuitively reconstruct its machinery in my mind, revealing a path to potential accumulation of all possible predictive power. It was really fascinating as RealMLP gifted me a huge amount of discovery feelings.

**Conclusion**

So, this is it. The only serious thing I did for this competition is obsessively calibrate one single model to bring it to the level of the best public solo models. Considering that XGBoost with Ridge meta feature is technically not a solo model, until the competition ended, RealMLP was the best public single model again. The less serious thing I did is that I simply ensembled via Hill Climbing algorithm variants of my public Trompt, RealMLP (by pytabkit) models, and other public TabM, RealMLP (from scratch), XGBoost, LightGBM, and CatBoost with overall CV 0.91979, Public 0.91730, and Private 0.91835 scores to earn my final rank.
