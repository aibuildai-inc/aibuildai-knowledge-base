# Overview of 1st place solution

Competition: PLAsTiCC-2018
Rank: #1
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033

EDIT: code is now available on my github page at https://github.com/kboone/avocado

First of all, thanks to everyone who participated in this competition! I learned a lot doing it, and I have enjoyed all of the discussions that I had with you. Here is an overview of my model that took 1st place in this competition. I will be releasing the code with a full writeup shortly.

I am an astronomer studying supernova cosmology, so my work mainly focused on trying to tell the different supernova types apart. This ended up working out well because everything else was fairly easy to tell apart. Here is a summary of my solution:

- Augmented the training set by degrading the well-observed lightcurves in the training set to match the properties of the test set.
- Use Gaussian processes to predict the lightcurves.
- Measured 200 features on the raw data and Gaussian process predictions.
- Trained a single LGBM model with 5-fold cross-validation.

I first use Gaussian process (GP) regression to extract features. I trained a GP on each object using a Matern Kernel with a fixed length scale in the wavelength direction and a variable length scale in the time direction. My machine could do ~10 fits per second so it took around 3 days of computation time to do all the fits. Gaussian processes produce very nice models for well-sampled lightcurves, and are able to get a nice model even when the measurements are in different bands. They also handle measurements with large uncertainties very gracefully. For poorly sampled lightcurves, the GP fits the available data well, but doesn't always do great for extrapolation. Here is an example of what comes out:

![gp example][1]

Using the GP predictions, I calculated lots of different features. The distinguishing features of supernovae are their peak brightnesses and the width of their lightcurves, so I put several measures of those into the model. For poorly sampled lightcurves, the GP doesn't always give great results, so I added features that let the model know how well the GP is doing. This basically boils down to counting the number of observations in different windows around maximum light. I also added features related to the signal-to-noise in each band, and some simple peak detection and counting to help with the non-supernova classes.

Now the training set is very different from the test set. To deal with this, I took every lightcurve in the training set and degraded it up to 40 times to get something that looks like the less well-sampled lightcurves in the test set. The degradation includes:

- Modifying the brightness of galactic objects.
- Modifying the redshift of extragalactic objects (including dilating time and changing the brightness).
- Adding in large "gaps" like the ones in the real data that show up because of the time of year.
- Choosing a new photo-z and photo-z error for the observation based on a model of how the spec-zs in the data turn into photo-zs.
- Simulating the detection to choose which objects would be included in the dataset that we were given.

This degradation was all tuned to the training/test datasets, and no external data was used. After this procedure, I end up with a training set of ~270000 objects that is much more representative of the test set than the original training set. I trained a LightGBM model on this training set using 5-fold cross-validation and making sure that I kept the up to 40 degradations of each object in the same folds. After tuning this model, I get a CV of around 0.4 on the original training set. Here is the confusion matrix:

![confusion matrix][2]

There are some interesting differences compared to [CPMP's confusion matrix][3], such as the fact that I have much lower accuracy on class 6 objects than CPMP. This appears to be due to the fact that my degradation procedure produces more low signal-to-noise class 65 than class 6 objects. It will be interesting to see if my method really does reproduce the test set better.

I played a lot with how to identify the class 99 targets. I found that the tree based models that I used are not very good for outlier detection. My best results came from choosing a flat score to give to the class 99 objects, and then using that in the soft-max to get the final probabilities. Using this, I got what I consider my best real score of 0.726 on the public leaderboard.

After trying to improve this score for a long time and getting nowhere, with a week to go I realized that I could figure out what the class 99 objects look like/don't look like by probing the leaderboard. This defeats the purpose of the class 99 prediction, and unfortunately ends up doing much better than any real estimate of the class 99 objects. I contacted the organizers about it, and was told that it was within the Kaggle rules to do this. In the end, I found that my best prediction of the class 99 objects was a weighted average of the predictions for classes 42, 52, 62 and 95. This trick boosted my final score to 0.670 on the public leaderboard. It will be interesting to see what other competitors did here.

Overall, I really enjoyed this competition and I learned a lot! I am working on cleaning up my code and making it friendly for others to play with. I think that there is quite a bit of room to improve the tuning of my model, and I did not try doing any kind of ensembling or using classifiers other than LGBM. Thanks to everyone who participated, especially @cpmpml who initiated a lot of great discussions and @ogrellier whose kernel I started with for my work!

For any astronomers who participated, I'll be at AAS in a couple weeks. I'd love to meet up and discuss the competition!


  [1]: https://storage.googleapis.com/kaggle-forum-message-attachments/440847/10890/gp_example.png
  [2]: https://storage.googleapis.com/kaggle-forum-message-attachments/438639/10876/confusion_matrix.png
  [3]: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/74564
