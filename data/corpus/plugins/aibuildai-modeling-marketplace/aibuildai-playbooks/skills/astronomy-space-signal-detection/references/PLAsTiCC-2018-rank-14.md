# 14th place solution

Competition: PLAsTiCC-2018
Rank: #14
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75054

Code and full writeup (see the pdf in the github repo).  https://github.com/btrotta/kaggle-plasticc

**Summary**

My solution is implemented in Python and uses LightGBM gradient boosted classification tree models. It scores 0.84070 on the private leaderboard, and runs in around 6 hours on a 24 Gb laptop (including calculating features, training, and prediction). It uses only elementary operations to calculate the features: there's no curve fitting or optimisation, which helps keep the runtime down. Apart from the hints revealed in the forum discussions, my original insights that gave the most improvement in score are: 

- Bayesian approach to removing noise from the flux measurements. Replace the flux with 
```
 (flux / flux_err**2 + flux_mean / flux_std ** 2) / (1 / flux_err**2 + 1 / flux_std **2)
```
where `flux_mean` and `flux_std` are the mean and standard deviation for the given object and passband. (Detailed justification for this calculation is in the pdf).
- Adding features based on scaled flux values. As well as a lot of features caclulated on the raw flux (after removing noise), I also calculated a scaled version of the flux by dividing by the maximum absolute value for each object and passband. I think this helps because it captures the shape of the curve, normalising for magnitude.
- Adding features to capture the behaviour around the peak.  Since these peaks can occur at any time during the observation period, we need a way to extract the data from these peaks. I did this by finding the "most extreme" minimum and maximum times for each object, defined as the time with detection=1 when one of the object's passbands differs the most from its median value. Then, for each passband, I retrieve the closest flux value before and after this peak time. I do this for both the raw (but de-noised) flux and the max-scaled flux. Also, I calculate the "duration" of the peak in each passband, defined as the period of continuous detections around the peak, and the time difference between the overall peak and the individual passband peaks.
- Understanding how to optimise the metric (including for class 99). (Basically you can just optimise normal log loss, then multiply the estimated probabilities by W/class frequency.)
