# 2nd-Place Solution Notes

Competition: PLAsTiCC-2018
Rank: #2
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75059

First, thanks to Kaggle and the PLAsTiCC organizers for presenting such an interesting challenge, and congratulations to all the top teams, especially Kyle with his amazing performance.

I haven't looked at all the other solution write-ups yet but there were clearly a lot of different ways to tackle this problem effectively. For us, NNs worked much better than LGB models. Our final ensemble included 9 models, 7 of which were NNs that scored as low as 0.75x individually. The two LGB models were much weaker, each scoring about 0.90x, but they did provide a bit of diversity to the ensemble.

My teammate Mike built the NN models so I'll let him describe them separately. What follows is some more general notes about our overall solution:

1. The 'Gap'. There was quite a bit of chatter about lowering the gap between CV and LB scores, most of which we ignored because our LB scores were tracking our CV scores very consistently, with a gap of 0.43-0.44. Our final solution, which scored 0.694 on the LB had a CV score of 0.264 (0.430 gap). In hindsight, it may have been a mistake not to look more closely at the gap. We really didn't pay much attention to the differences between the train and test sets until the final few days of the competition.

2. 'Class_99'. Especially in the final week, we did quite a bit of LB probing to come up with a better way to predict Class_99. In the end we settled on the following alorithm:
	a) Set Class_99 to 1-top_prediction per row
	b) Move class_99 closer to 0.14 (for extra-galactic objects) and 0.014 (for galactic objects) with the following 	   formula: class_99 = (2*class_99 + 0.14) / 3
This improved our scores modestly (about 0.005).   

3. Ensemble. The final ensemble was a very shallow (max_depth=2) LGB model that was very effective. 

4. 'Detected' Flag. I still don't understand what this means or how it was calculated, but it's clearly very significant. As Kyle pointed out in a post, it's roughly set when the absolute flux value is 5 times greater than the flux error (the documentation says +- 3 sigmas, which doesn't make much sense to me), but there are many exceptions to this rule. The only thing I can think of is that the flag was set before some random jitter was applied to the flux values. If anyone knows more about this, please tell me because I devoted several days trying to reverse-engineer this feature with no success. That said, I did find that this ratio -- absolute flux/flux_error -- to be very significant in my models. One of the LGB models generates features for 4 different sets of observations -- all, detected only, flux error ratio &gt;3, flux error ratio &gt; 4. Because the NN model was able to continuously process the flux and flux error values in parallel, this relationship did not need to be manually specified, one reason I suspect that the NN model performed so well.

5. Hostgal_specz pseudo-labeling. One thing that worked for both the NN and LGB models was to build a separate model to predict the hostgal_specz values (using both the training set and the test set objects that have this value), and then using oof predictions for these values in our models.

6. Flux adjustments. For the LGB model, I found that adjusting the the flux values with the simple formula flux *= hostgal_photoz improved the results. All efforts to normalize flux values by object and/or passband did not help. In the NN models, on the other hand, Mike found a very effective normalization strategy.

7. MJD adjustments. Adjustments to MJD based on redshift and wavelength did not help the LGB model.

8. Augmentation. We tried a variety of different augmentation methods. Augmenting the training set with random variations helped the NN models but did not help the LGB models. We also tried augmentation with pseudo-labeled test objects but this did not help. As I mentioned above, we weren't really focused on the differences between the train and test sets. In the final days of the competition, I realized that we should be augmenting in such a way that the train distribution would more closely match the test distribution, but we didn't have time to try this idea.
