# 17th Place. Lessons learned.

Competition: career-con-2019
Rank: #17
Source: https://www.kaggle.com/c/career-con-2019/discussion/89259#latest-515629

I will join the crowd of thankful guys who applaud to  Markus F, Thomas Rohwer, and Nanashi. My success is mostly based on their findings.

Before going into technical details I would like to say once again that this competition was about finding leaks rather than helping robots to navigate better.
Even if the train and test could not be linked through endpoints the results of the predictions would be useless anyway.
Most of us were using vibrations to generate features. But the vibrations are all about a combination of frequencies. Frequencies depend on the robot’s speed. Robots speed depends on the internal logic. The logic is based on the predictions. Dead loop.

If the dataset would have a feature for speed (with different scenarios on each surface) or a visual sensor for the surface it would be more valuable for the robots.


**Now technical details:**
Of course, the linkage hack pushed CV and LB up as nothing else. 
This picture gave me a clue where to search for hidden treasure.

[Y to W plot for multiple series]

I was fighting to fine tune my own linkage solution for a week when Markus published his clear and incredible effective code.
I over-complicated my solution with additional weights for XY and W, Z coordinates (assuming  they have different scale of changes). And also I was matching trends on endpoints in order to filter out “hit the wall” cases. I assumed that the probability to see a change of gradient for XY line in between of series is quite low. The pairs with different gradients should belong to different sequences even if their Euclidean distance says they are matching. As a result my linkage gave very clean results but the sequences were short (2-4 series in a row) and only 1k of pairs were identified (out of 3k).


I used linkage conservatively. The test series linked to train series got surfaces from y_train. The remaining test series got predictions from RandomForest model. This gave me a boost from 0.72 to 0.91 on LB. Above that I cleaned test predictions. A surface surrounded by other surfaces was updated to match neighbors (like ABA &gt; AAA and ABCA &gt; AAAA). It gave one additional percent on LB. 

Also I added a number of features that are "forward" and "backward" moving averages calculated for 5 series forward and 5 series back along a linked chain (or to the boundary of the sequence). They all jumped up on the importance list. The base features for MA were the ones that were high on importance but orientation X, Y were ignored.


From the very first days of the competition I added [librosa.mfcc](https://librosa.github.io/librosa/) features ([code is here](https://www.kaggle.com/c/career-con-2019/discussion/87258)). A simple one parameter optimization gave me the best combination of the signal frequency and number of features. It did not increase LB but I had strong feeling that it is the most correct way to solve the problem for robots (though not to win). 

Later I added a number of features from scipy package for each series
1.	Detrending &gt;  signal.detrend 
2.	Number of peaks and total peaks height &gt; signal.find_peaks
3.	signal.welch
4.	Sum of the absolute of the detrended and scalled signals per quarters of each series.
5.	STD of the above four numbers to represent the measure of the signal stability inside series.

All new features were cross validated on the simplest Random Forest model with GroupKfolds  (original groups are used rather than the new groups from linkage). The features that increased the CV (by the way it was always in a range of 0.5+/-0.03) were added to the model.

Group 27 was split onto 14 groups to have one series per group.
      target.loc[target['group_id']==27, 'group_id']=target['series_id']+76-27

Final model was using StratifiedKFold with 75 folds just because it gave slightly better result on LB than GroupKfold.

KNeighborsClassifier and BaggingClassifier also gave comparable results and I hoped to do a blending but the result did not look promising.

It is interesting that looking back on my submissions sent before I started to use the linkage I see that the BaggingClassifier worked best of all on private LB as well as LGBM while RandomForest lost 2 percents. While CV and public LB were below RF numbers.

On public LB I got 0.92 while only 0.76 on private side though moved 5 places up.

**What did not work.**
- CV: GroupKshift, GroupShuffleSplit, RepeatedStratifiedKFold
Transformations: PCA, Fake additional data with slightly randomized features to fight categories disparity, GaussRanking, Fast Fourier Transformations (FFT) (worse than MFCC), a few other features from scipy package.
- Models: GradientBoostClassifier, AdaBoost, KerasNN, LGBM (even optimized), CatBoost
- Validations: Major Voting for blending of folds against argmax for the sum of predicted probabilities.
- Recursive feature elimination and recursive feature combinations. I had to use GroupKFold here instead of  StratifiedKFold  but spent too much time with no valuable result and decided not to invest more in this technique. 
- Recently installed Nvidia card was useless in this competition because of small dataset and simple models.

**Useful tips:**
“Pickle everything”. I saved hours and hours of my time by just saving intermediate results from all feature transformations. The next time the same code was running it loaded the saved file instead of recalculating everything from scratch. I had separated files for Quaternion, MFCC and scipy features that I merged before sending to a model. At the end of the competition the folder with all those versions reached 10Gb.

Think about physics and data meaning before doing all those feature transformations. For example: an attempt to filter out a noise can throw away the important signal because the noise is the signal by itself. Robots have wheels. The wheels have rubber tires with threads. Those threads create base signal (low frequency vibration) while the surface introduces additional high frequency as noises (except soft carpet that decreases the vibration). Plus the motors of the robot have own frequency. It is hard to predict what you will lose by applying highpass/lowpass filters.
I left Santander competition because it had no sense to optimize models without knowing what each feature means.

Optimize features one by one and remember that Bayesian optimization cannot handle integer values properly.

PS: Many thanks to Kaggle for hosting this awesome competition. I learned a lot during the rage for the top rates.
