# #13 Solution, true story: tries and fails

Competition: PLAsTiCC-2018
Rank: #13
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75134

Thank you for this challenge, lots of fun and learning, data are good and clean, answers from organizers are timely. Very well organized. 
Here is our write up.

**What worked:** 
Our final solution was based on LGBM model inspired by Oliver kernel, feature design and selection, and augmentation.

**Features:** 

1. Magnitudes. Both aggregated and per band, but without k-correction till the last day. As we were not quite sure how to interpret detected flag exactly, we decided to try both: for detected == 1 and for all data. Surprisingly, magnitude features for detected == 1 worked better on LB (not on CV though). I attribute it to the outliers removal with detected == 1 and also class 99. Indeed, to detect unknown -- you have to be 100% sure, so the detected points may be important for 99 discovery. Known objects are a different story: we know them and the confidence level of error can be lower. 
 
2. Parametric curve fittings. We used models of Bazin (https://arxiv.org/pdf/0904.1066.pdf) and Karpenka (https://arxiv.org/abs/1208.1264), Gaussian fits to define peak-width, and the decay slope fits of supernova light curves in the log scale. Fitting loss was one of the most important feature in all the parametric curves.

3. Basic features from cesium package and statistics. Those based on ratios, std, and skew made it to the final. 

4. From the Bazin equation we derived the position of the max for the incomplete light curves and derived magnitudes from fits as well. They helped for some bands. We also tried a range of other parameters, m15 and m-10 parameters, peak width at different levels -- but they did not help. 

5. We checked all features from feets. CAR tau from feets was particularly helpful, but it takes ages to calculate. So, Mithrillion optimized a code to make it 10 times faster! If astronomers are interested, I think we can put it in a kernel.

6. Autocorrelation lag 0, also estimated peak widths made it to the final. 

7. Finally, on the last 2 days I was pointed to the proper calculation of k-correction by Kyle (thank you). I used the calculator from here (http://kcor.sai.msu.ru/getthecode/) and the colors as suggested here (https://arxiv.org/pdf/1410.8139.pdf). It's only valid for relatively small red shifts, but the majority of data lie in this region. 
The k-correction of magnitudes per band did not help, but along the way we calculated colors: both from detected magnitudes and Bazin fitted magnitudes --&gt; that's how we got from 0.836 to 0.815 on a single model 

Feature selection: we tried both RFE from sklearn and manual tuning while looking at eli5 importance and deleting highly correlated features. Manual approach and eli5 importance worked out much better than RFE.
  

**Battling over fitting:**

1. To overcome over-fitting we first augmented train 10x times using flux variation within normal distribution of the corresponding flux_err, similar for photoz values (except for galactic classes). We also introduced small time shift to bring translation symmetry. It was meant for NN mostly, that we did not use at the end.

2. We twisted parameters of classifier, especially deep tree and max bin gave a change.

3. We realized that we have ~30% of ddf samples in train and only 1% in test. So we are likely to over fit to the ddf samples. First, we down-sampled ddf in the current augmentation and re-selected features checking eli5 importance for wide field drilling samples only (ddf = 0). Secondly, we changed augmentation, adding 30x train from ddf = 0 samples and leaving ddf samples only in the original train fold. That helped. Unfortunately, this idea came to us in the last 48 hours, so we did what we could manage within that time, and feature selection -- optimization still could have been better. 

With 30x augmentation CV 0.5 LB 0.815, gap 0.315

Not much ensembling, 50/50 blend of two LGBM models with slightly modified features and parameters was the final blend

**What did not work:**

1. Autoencoder. Although it has shown quite nice curves reconstruction, unfortunately, non of it's features made it to the final. We tried to include autoencoder loss as well -- did not help.

2. More parametric fitting. Although we tried adding models with double-peak (https://arxiv.org/abs/1208.1264 ), and add tau per band as well, nothing really improved LB compared to just our very first Bazin parametric fit.  

3. Gaussian Processes augmentation. We tried to use GP to create more non-ddf samples for augmentation, but got a bit of an issues with detected and flux error, so far this did not work out, but the idea is good I think. Mithrillion gave more comments under Kyle's solution.

4. PU classification to identify class 99. We tried to detect negative labels (99 labels that are not in train) for 99 class prediction using a PU classifier (https://arxiv.org/pdf/1605.06955.pdf ). It did not work out...

5. Class 99 probing. We tried to find class 99 in test using LB probing. We noticed that the percent of high probability 99 objects is in the far z region. Indeed, apart from themselves, people tend to know better what is closer than what is far. Also, our visible Universe keeps expanding as the light from far away objects keeps reaching the Earth, so we expect more unknown from far, I think. We tried to find 99 among photoz &gt; 2.5 and high probability of 99 class in Oliver and Scirpus methods. Guess what ? ... Bingo! It did not work out.

**What we did right:**

1. We teamed up! Teams have power, totally recommend, it's more fun and more learning and better results.

2. We did not give up when we fell to place 19 just a three days before the end. The competition is not over till the last submit! It's in the last two-three days that we actually realized our mistakes, fixed what we could and managed to improve from 0.87 to 0.815 on a single model getting to place 12. Then it was just a bad luck on private...  

3. We asked questions on forum. And people answered. Special thanks to Kyle, CPMP and organizers.

4. We read, we learned, we tried, we discussed, we tried again and we learned more...

**What we did wrong:**

1. We did not utilize our submissions properly in the beginning of competition. When we had plenty of them -- that's when we should have tried more features on LB as well and more experimenting with parameters earlier. Last days every submission counts and you cannot just twist parameters while also checking on LB.

2. We did not bring ddf percent close to the test data until the last 48 hours. We should have looked at data more and try in earlier. In general, I think 99% of answers are in the data, so if you feel stuck -- look at them again, look at them more.

3. Strategy. We first did things that should work, like parametric fits from papers and magnitudes, and we left fun experiments to the end. At the end everything has speed up,  and we did not had time to properly utilize those experiments. Maybe we should have done it earlier when the time was not pressing. It's my third kaggle, so I still do not know what the best strategy is. More comments from Grandmasters here would be helpful.

4. We did not plan submissions ahead. Trying to generate 5 submissions in evening to utilize all of them did not work out. I think it's nice to plan what we want to probe on LB and prepare ahead, so to have a full use of submissions. 

5. Trying to find a black cat in a dark room. We spent too much time trying to find "the secret", identify class 99 and build a classifier to find it. Instead, we could do more traditional things that bring incremental improvement, like better feature selection and parameters tuning. We left it to the last few days, but then its not enough time and submissions left.

*The hardest thing of all is to find a black cat in a dark room. Especially if there is no cat. Confucius

**What I still do not understand:**

We noticed a considerable variations in folds loss with LGBM, it is stratified, it is still big for wdf only training. Anyone knows, why is that?

NN classifier did not want to improve anymore after around 0.94 even with augmentation. Why is that? We tried to twist parameters.

Feature selection: what are the best feature selection methods people use here? I've seen comment about boruta, but found it for random forest only, does it exist for lgbm and xgboost? What people use here?

It looks from other solutions that we did things right, so what was the most important that we missed to cross 0.8 margin on a single model (we had 0.815)? And why there was such a shake on private? 

**Personal outcome:** It was a hard work and lot's of learning. The last few days were especially crazy for me. I was literally holding a baby with one hand and programming with the other, trying to concentrate and compete for gold against people who can use both hands, lol. No much sleep, no much food, no time for anything else. 

Still, it was fun and learning experience! Thanks to kaggle, organizers and most of all -- to my best-ever-teammates!!!

PS. Dear kaggle people, please consider: 
- adding emotions (happy, sad, thinking, and a little penguin dancing). 
- consider adding a sign: 
KAGGLE IS ADDICTIVE ! ENTER AT YOUR OWN RISK !!!
