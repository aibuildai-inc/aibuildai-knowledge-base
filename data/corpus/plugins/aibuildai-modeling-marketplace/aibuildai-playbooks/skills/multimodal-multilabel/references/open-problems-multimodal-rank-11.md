# 12th-Place Solution

Competition: open-problems-multimodal
Rank: #11
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/366455

First, congratulations to the winners, especially @senkin and @tmp for leading throughout and @shujisuzuki65 for his big jump from public to private LB. I'm eager to hear from both teams about their techniques. Overall, it was an interesting competition that gave me a greater appreciation for the challenges that bioinformaticians face. 

I don't know if others feel the same, but it seemed like a very long competition to me. I ran out of gas about midway and didn't really work on it much the last 4 weeks or so, which means I didn't use the raw counts at all. My solution, therefore, is fairly simple. 

**CV setup:** I assigned each batch (unique user/day) to a separate fold, so I had 9 folds for citeseq and 12 for multiome. This was expensive, but had the virtue that it wasn't optimized towards either new donors (public LB) or new days (private LB). LB scores tracked local CV scores very closely. Even small gains in local CV almost always led to similar gains on the LB. This CV scheme was probably the reason that I fared well on the private LB (46->12).

**Data transformations:** I tried a lot of ways to denoise and transform the data, but most of them failed. In the end, I just used PCA and tSVD of the original data. 

**Feature engineering:** None for Multiome. For Citeseq, I trained 140 shallow LGB models (one for each target) using the full set of data. The goal here was not to use the models themselves, but to see which features were important for each target. I used the top 100-200 features per target in the later, deeper modeling.

**Modeling:** For Citeseq, I trained both single-target (140 separate models) and multi-target NN models (using Fastai). I also trained LGB and CatBoost models for each target. Altogether, I trained over 20 sets of models using different variations of the PCA data combined with selected features from the feature engineering. Individually, the models had local CV scores in the range 0.8995 - 0.9017. For multiome, I trained 3 multi-target NNs on the tSVD-reduced targets, and one CatBoost model.

**Ensembling:** I blended the models together using a very simple optimized weighting scheme where the only possible weights were 0,1,2, or 3. I tried other ensembling techniques that had higher CV scores, but they performed worse on the LB. I was afraid this might be due to some hidden leakage between folds, so I stuck to the simpler weighting scheme. This led to local CV scores of 0.9039 for Citseq and 0.669 for Multiome.

**Thoughts about trends in data:** I was intrigued by @AmbroseM's posts arguing that the data is a time series. There are undoubtedly trends over the 7 days of training data, but I was concerned about whether these trends would continue to day 10. I don't know enough about the biology, but it seems likely that cell behavior has both long-term trends (aging) and short-term trends based on things like diet, exercise, illness, etc. I decided that the trends visible in the 7 days of training data could very easily be short-term trends that would reverse themselves after 7 days, or could be just coincidental due to technical aspects of the data collection. Based on @AmbroseM's post here (https://www.kaggle.com/competitions/open-problems-multimodal/discussion/366395), it seems like the trends did, in fact, continue to the test set. I would be very interested to hear from some cell scientists about what these trends might signify, whether they're cyclical in nature, and if yes, how long each cycle is typically.
