# 5th place solution - NeurIPS ARIEL 2024

Competition: ariel-data-challenge-2024
Rank: #5
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543760

First of all, I would like to thank the organizers for this amazing competition and my teammate Youri. It was a pleasure to work with you on this competition for the last two weeks. Congratulations on your Grandmaster title!

## Tldr, give me the Code
Data Prep Notebook: https://www.kaggle.com/code/ilu000/neurips-ariel24-data-prep-5th-place-solution
Data Augmentation Notebook: https://www.kaggle.com/code/ilu000/ariel24-data-augmentation-5th-place-solution
Final Train Dataset: https://www.kaggle.com/datasets/ilu000/neurips-ariel24-5th-place-solution-data
Final Models: https://www.kaggle.com/datasets/ilu000/ariel24-models
Train Notebook: https://www.kaggle.com/code/ilu000/neurips-ariel24-5th-place-solution-train
Inference Notebook: https://www.kaggle.com/code/ilu000/neurips-ariel24-5th-place-solution-inference

## Data preprocessing

As with most competitions with synthetic data, when starting this competition I focused on data exploration and modeling the aggregated light curves. Data preprocessing in our final solution was mainly done as described in https://www.kaggle.com/code/ilu000/ariel24-data-prep-fixed-dt and as shared by the hosts. Minor improvements to speed up the process by using GPU and parallel processing were helped to get the runtime down to a about 1h for the full test dataset. As test has different hot and dead pixels, using interpolation to fill the gaps was helpful to generalize to the distribution shift.

## Modeling the light curve



As many have figured out, artifical drift over time as a polynomial of up to an order of 4 or 5 has been added to the data. After proper data preprocessing, detecting and removing this drift was the key to get to a decent score. Detecting transit points, fitting this polynomial and using directly the mean predictions with a proper sigma (tuned by star on LB score) gives an public LB score of around 0.546 and private of 0.578. Excluding the regions around the transit points makes this a bit more stable. By the end of the competition, this type of solution was well known and was the highest scoring public solution which would have ended in the bronze region with some tuning. Up to this point, local validation could be done by splitting by star and had great correlation with the public LB score.

## Wavelength features

All planets in the training dataset featured weak to strong wavelength dependent absorption caused by the molecules in the atmosphere. Actually looking at train labels, we can overlay them with absorption spectra of known molecules and see that train is composed of a mixture of H2O, CO2 and CH4.



When we teamed up, we had two different ways to deal with this. I was using a model-less approach to extract absorption from a sliding window over the wavelength. With some smoothing and binning, public LB score that can be achieved with this approach and a constant sigma is about 0.590 and still correlates well with validation as it doesn't overfit on specific destributions. 2D Gaussian + Markov chain Monte Carlo (MCMC) as shared in the forums and what apparently is the state of the art when modelling the light curves in literature helped to get slightly better results, but the runtime for MCMC prevented us for ever using it in any kernel submission.

When adding any kind of model on top of this, local validation score skyrockets by 0.100 easily as the model can fit to the three molecules in the training set. But public LB didn't follow, indicating that the test set has different molecules present.

Youri did use a model-based approach and cleverly constrained it by modelling 5 gaussians that fitted approximately the absorption peaks of H2O, CO2 and CH4. Then, splitted the signal with these gaussians and used the aggregates as features for a Linear Regression model. He also subtracted the mean absorption first to further constrain the model. This approach worked very well on local validation while still able to carry over most to LB. From these results, it was quite clear that models are very powerful to fit the training data, but it was hard for us to fully generalize to the test set.

As we expected and close to the end of the competition, the hosts confirmed that there are new molecules in the test set that were not present in the training set. We experimented with different splits of the training data and even experimented with clusters by dominant molecule, but public LB stayed to be the best way to validate.

## Sigma estimation

Estimating the sigma was a crucial part of the competition. We tried different ways to estimate the sigma, and some were harder to validate locally. What did work amazingly locally and on LB was the strong correlation from pred.std(-1) (standard deviation over wavelength for each planet) vs. the error. Scaling the mean sigmas by this factor improved local validation and LB score by ~0.030. Additionally, we used a small fully connected NN to predict the sigma and blended these predictions to gain a few more points on LB.

## Synthetic data

Generating more data to extend the training set to more molecules pushed our LB score significantly in the last days. With Taurex3, we generated absortion spectra for 9 more molecules (all_molecules = [(H2O, CH4, CO2), CO, NH3, HCN, C2H2, SO2, C2H4, H2S, PH3, TiO]). 



We had to make assumptions regarding planet temperature which we took from literature, but still we couldn't get ideal matches between Taurex3 and what we see in the training data for known molecules such as H2O. 



We used the training data and augmented it with these absorption spectra in two ways. First, we added it on top of the training labels and modified the signal accordingly (using the transit zone from our pipeline). Second, we used clean/mean signals from the training data and added the absorption spectra to it, again modifying the signal accordingly. Then, we retrained our models on this augmented data and improved our LB score by ~0.020 while local validation score slightly decreased as it is harder to fit more molecules. Interestingly, only adding a few molecules improves both, local and LB score, but adding more molecules only improves LB score.



I am confident that with more LB probing for the right molecules, we could have improved our solution even further. But, we are happy with the 5th place and are looking forward to read about the solutions of the top teams.

## What did not work

- Linear combinations of absorption spectra. Still not sure why this didn't work, probably never got the settings quite right and the models just learned it better.
- Scale sigma by absolute absorption. We saw a slight correlation in the training data, but it didn't work on the test data.
- Use a network to predict the absorption. It might work now with more training data that better fits what we have in test, but in the end time was running out to test it again.
- many more
