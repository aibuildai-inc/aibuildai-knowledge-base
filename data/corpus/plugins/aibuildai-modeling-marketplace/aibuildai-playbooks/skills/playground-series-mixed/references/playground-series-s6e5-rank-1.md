# 1st Place: By the skin of my teeth

Competition: playground-series-s6e5
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth

Well, that was the most thrilling finish to a Kaggle competition that I've experienced so far - I won by the barest of margins (0.00001) with a (literally) last minute submission! I was expecting a good rank, but to end up 1st was a most pleasant surprise.

Before I go on, I'd like to thank those who generously shared their code and/or insights, including (but not limited to) :

@yekenot,  @ern711, @masayakawamata, @siukeitin, @ravi20076, @amanatar, @quatlify, @yunsuxiaozi, @joypig 

I'd summarize my efforts in the competition as follows:

<b> Model Diversity </b>: As usual, I tried many diverse models and approaches, including XGB, LGBM, CatBoost, RealMLP, TabM, HGB, RF, YDF, FT-Transformer, MLP-PLR, and many more. I took advantage of AutoML libraries like Autogluon, Light AutoML, FLAML and PyTabKit, and also used diverse metrics, though it was AUC in the vast majority of notebooks. Also the usual tricks like swapping features, hyperparameters etc. between well performing models.

<b> Feature Engineering </b> I created several notebooks with "Heavy FE" of the kind seen in excellent notebooks by @cdeotte, @masayakawamata, @mahoganybuttstrings and others in past playground episodes, with 200-400 features; and also reused excellent features from @yekenot and others. I also tried AutoFE libraries like FeatureTools (which produced a poor AUC) and AutoFeat (which timed out after 12 hours without producing anything useful). I haven't looked into why the AutoFE tools failed this time - was it the nature of the data, or Claude's code?

<b> Ensemblers </b> I tried many, including Ridge, Logistic Regression (LR) with Logits, XGB, LGB, CatBoost, Simple MLP, RealMLP, Autogluon. Autogluon was the best overall, followed closely by LR with Logits. 

<b> Breakthrough </b> While I made steady progress for the first 3 weeks or so, after reaching ~0.95470 on the public LB, it felt like I was spinning my wheels for a while. Going back to the drawing board, I revisited the relationship between the original dataset & the competition data. Even though adding the original data was boosting AUC, @siukeitin and others had pointed out significant differences between the datasets. In particular, @ravi20076 showed the adversarial AUC for each variable, and "Driver" stood out as being the most different. It made sense to try some models without using "Driver" - the "Driver_Dropped" models boosted the score a little, as did some which didn't use the original data. I also played around with modulating the impact of the original via sample weights - weights between 0.5 and 1 seemed to produce pretty much the same score, while dropping it to 0.25 reduced CV and LB scores.

<b> Final Push </b> I made an effort to add many more models towards the end, though I couldn't reach my goal of 200 OOFs, ending up with 186. I'd been careful not to do any LB probing, and given how hard it had been to move beyond 0.95470, was wondering whether some of those ahead of me had perhaps overfit the public LB a bit. In the last two days, I took a judicious risk - that of adding some L2 OOFs to the OOF pool. This carries a risk of leakage, so I decided to keep one submission free of this. The 186 OOFs included 4 L2 OOFs and scored 0.95487 (public), 0.95503 (private) with LR Logits; while the 182 L1 OOFs combined for 0.95480 and 0.95495. A blend of two strong ensemblers often does better than both - I decided to try 50-50 and 80-20 blends of Autogluon and LR-Logits submissions that scored 0.95486 and 0.95487 on the public LB - the competition ended while I was submitting the 80-20 blend - but the 50-50 blend, which I'd submitted a minute ago, ended up scoring 0.95488 and 0.95503, landing me the 1st place by the barest of margins. I wouldn't have been surprised to end up at 3 or even 2 (nor shocked by a small fall down the rankings), but to end up pipping @cdeotte by 0.00001 in literally the last minute was quite the thrilling surprise 😀

I used Claude throughout the month and it was a big help, though there were several occasions where I had to talk it out of quitting too soon. And it kept regenerating entire notebooks when only a few lines needed to be modified, causing me to be locked out too soon on many an occsion.

I'd like to end with big congratulations to  @cdeotte, @kagglersergio, @unseenuser, @milanfx, @mahog, @masayakawamata, @mikhailnaumov & all others who finished strong and/or learned new things. All the best for June, and Happy Kaggling!
