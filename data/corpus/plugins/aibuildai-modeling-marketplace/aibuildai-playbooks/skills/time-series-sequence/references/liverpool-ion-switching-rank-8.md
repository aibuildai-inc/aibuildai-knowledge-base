# 8th Place Solution

Competition: liverpool-ion-switching
Rank: #8
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153802

Congratulations to the winners and especially to Gilles &amp; Kha Vo &amp; Zidmie for an excellent effort leading the public lb and private non-leak scores. Despite the simulated nature of the data it was fun trying to piece together all of the aspects of the problem.

**Initial Approach**

Drift was removed from the raw data in the generally accepted manner using a sine curve as identified by Chris Deotte, Markus and Eunho.

A simple regression of the (cleaned of drift) signal against the target produced a reasonably good fit and thus I adopted the idea of defining the residuals from this regression (reg0) as *noise* which I could then attempt to model subsequently.

I also adopt the designation of the batches into five separate models depending on their nature as defined by Chris Deotte in his early notebook “One Feature Model”. I denote these separate models as m1s, m1f, m3, m5 and m10 depending on the maximum number of channels open in each batch and a further subdivision of the seemingly binary sections of the target data into ‘slow’ and ‘fast’.

(reg0)	signal = 1.2336 x target – 2.7337 + *noise*		m1s,m1f,m3,m5
		signal = 1.2336 x target – 2.7337x2 + *noise*	m10

Visual inspection of the *noise* over time suggested two regions of excessive *noise*, a large patch in m3 and a smaller one in m1s. These regions were excluded from all further analysis.

Even though the regression coefficients for the *noise* regression of signal against target were approximately the same for all models, the standard deviations of the residuals were different for each model suggesting that analysing the five models separately would probably be easier. The intercept for the m10 was almost exactly twice that of the others – see later.

***Noise* Modelling**

A set of simple features were first created for each model, initially for input into lightgbm. These were the signal and its leads and lags up to 14 time periods. In addition, I calculated the rolling 10-period min and max for both lead and lag for each observation as well as the rolling min and max for a 20-period lead and lag ahead of or behind the first 10 periods.
For an observation of the signal Xt, the features here are Xt-1…Xt-14, Xt+1…Xt+14, min(Xt-1..Xt-10), min(Xt-11..Xt-30), min(Xt+1..Xt+10), min(Xt+11..Xt+30), max(Xt-1..Xt-10), max(Xt-11..Xt-30), max(Xt+1..Xt+10), max(Xt+11..Xt+30).

I first used these features to estimate the signal itself in a lightgbm model. This estimated signal then formed the basis of another set of features in a similar manner to those I first calculated. Finally, I added a number of features based on the rolling min and max of the signal relative to each observation, e.g., min(Xt-1-Xt..Xt-10-Xt).

These features then were used in lightgbm to estimate the *noise* for both test and oof. These *noise* estimates could then be reversed back using the original regression (reg0) to estimate the target and calculate F1 metrics. Note that I found that only including 3 observation leads and lags for the signal (rather than 14) produced the best results here.


**Further Cleaning of the Signal**

Frequency analysis of the *noise* showed up clear contamination with a regular signal at about 50Hz along with higher frequencies with lower amplitudes. Attempting to remove this led me to find that each 10 second minibatch had a slightly different frequency (and phase) around 50Hz. My lack of expertise in signal processing led me to perform an exhaustive search for the best fit for each minibatch. Here I noticed some differences between models. Models m1s, m3 and m5 all had easy to spot signals in 10 second minibatches. Model m1f seems to have 1 second minibatches. Model m10, however, has a more complicated structure with one or two peaks for every 10 second minibatch. More on this later.
Removing the 50Hz contamination is reasonably straightforward for the train data but fortunately my *noise* estimates from the previous lgb also showed the same contamination (though with smaller amplitudes for m5 and m10). I was thus able to estimate the contamination for the test data and remove it. 

With my newly cleaned signal I was then able to repeat all of the previous steps. I obtained a new *noise* estimate (which produced much better results in cv and on the public lb).
Further frequency analysis showed up harmonics of the 50Hz still contaminating the signal. The most prominent were at (50x23)Hz, (50x25)Hz and (50x29)Hz. For each of these harmonics I removed it from the signal and repeated all the steps so far, seeing whether it improved the cv each time. For m1s and m1f just the original 50Hz contamination removal was as good as it got. Model m3 had 2 cycles, m10 had 4 cycles. The m5 model kept going in this fashion for 7 cycles. (Note the seventh cycle for m5 was not a harmonic – maybe something else was contaminating the signal?)

I thus ended up with what I hope are fully clean signals for each model.


**Different Types of Modelling**

As well as using the lgb modelling, I also experimented with neural nets using Keras. Ultimately a simple 2-layer dense net trained with same features I built for the lgb model proved to be the best I could do.

**Funky Loss Objectives**

Noticing that the exact error doesn’t really matter when it is small or large but mainly when it is close to the boundary between the discrete channel numbers, I experimented with different loss objectives in both lgb and keras. Both of these showed better results on cv and public lb using a cubic error term compared to plain mean squared error. A quartic error term was also ok giving similar results to a cubic term.

**Model m10 and Data Augmentation**

There were a couple of clues in preceding sections. The intercept in the original regression (reg0) was twice as large for m10 compared to the others. Also, the 50Hz contamination showed single or double peaks for m10 only. Eventually I realised that the m10 model was comprised of the sum of two separate m5 models. The double peaks appear to be an interference pattern from two peaks from the m5 models. In addition, the variance of the *noise* for an m10 model is almost exactly twice that of an m5 model.

This meant that I could augment the m10 data by creating artificial data using the m5 data that we have and training on that. I created new data that looked like the m10 data by combining each 10 second minibatch of m5 data with the other 10 second minibatches for a total of 45 minibatches of new data. The hope that the data was similar enough to use for training was backed up by a small but worthwhile improvement in cv.

**Ensembling and Optimization**

For model m10 I used an ensemble of approaches utilising cubic and squared objectives, lgb and keras modelling. [This ensemble was then optimised slightly to obtain the maximum f1_macro score in cv.] For models m5, m3, m1f and m1s I used the single model that was best in cv. The rank order of my best m10 models in cv seemed to show little in common with the rank order on the public lb so it was hard to choose my final submissions. In the end I opted for a weighted ensemble of four m10 models that was good in cv (and fortunately ok on the leaderboards).
This ensemble scored 0.94443 in cv overall and gave 0.94685 on public lb, 0.94539 on private lb, good enough for 8th place.
