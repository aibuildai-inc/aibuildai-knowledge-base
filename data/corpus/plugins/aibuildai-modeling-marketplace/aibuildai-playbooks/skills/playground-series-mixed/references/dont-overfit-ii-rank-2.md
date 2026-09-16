# 2nd Place Solution

Competition: dont-overfit-ii
Rank: #2
Source: https://www.kaggle.com/c/dont-overfit-ii/discussion/91683#latest-535361

Thank you Kaggle for a fun competition and thank you Kagglers for fun and stimulating discussions/kernels. My final model is a classification hyperplane that uses 15 features:

    pred = sigmoid( -0.0661x_16 + 0.178x_33 + -0.0526x_45 + -0.0736x_63 + 0.171x_65 + -0.105x_73 + -0.118x_91 + 0-0.0446x_108 + -0.0984x117 + 0.0462x164 + -0.0513x_189 + 0.112x_199 + -0.0704x_209 + -0.120x_217 + -0.0455x_239 )

This hyperplane was found by performing logistic regression with L1-penalty on the 2225 observations in the public test dataset combined with the training dataset using LB probes. It achieves CV 0.914, LB 0.890, and Private score 0.869. The solution would be more accurate if I had two more weeks to allow the LB-probe regression iteration to converge.

To find coefficient `a_33` (the coefficient in front of `x_33`) you execute and submit the following code:

    var = 33
    test = pd.read_csv('test.csv')
    sub = pd.read_csv('sample_submission.csv')
    sub['target'] = test[str(var)]
    sub.to_csv('submission'+str(var)+'.csv',index=False)

[image]

Then the value of `a_33` is computed by

     a_33 = LB_SCORE_33 − 0.500 = 0.671 − 0.500 = 0.171

In the past 10 days, I was only able to compute 50 coefficients. The remaining 250 coefficients have been set to zero. To prevent overfitting, I (1) modify `a_k` based on the training data and (2) set all `a_k` with `abs(a_k)&lt;0.04` equal to `a_k=0` because it was shown [here][1] that we cannot be confident that a variable is **useful** in this situation.

A full explanation of my LB-probing techinque with code and simulation is posted [here][2]. I look forward to reading other Kaggler's solutions. Please share.

UPDATE: A pictorial / mathematical explanation why `a_k = AUC - 0.5` is posted [here][3].

[1]: https://www.kaggle.com/cdeotte/can-we-trust-cv-and-lb
[2]: https://www.kaggle.com/cdeotte/lb-probing-strategies-0-890-2nd-place/
[3]:https://www.kaggle.com/c/dont-overfit-ii/discussion/92565
