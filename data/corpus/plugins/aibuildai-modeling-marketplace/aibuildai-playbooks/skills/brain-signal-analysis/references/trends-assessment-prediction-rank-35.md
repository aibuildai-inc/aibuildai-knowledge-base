# 35th Place Solution- Magic PP - Site 2 correction

Competition: trends-assessment-prediction
Rank: #35
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162785

**Part 1**
First of all, Congratulations to my teammate @prateekagnihotri for his first silver medal and a lot of thanks for coming up with excellent stacking models. Congratulations to all the teams that did good on private lb. Also, credits should be given to @david1013 for the high scoring public kernel that was used in final blend. The best part was we did all these in the last 10 days, that too with just tabular data.  We wanted to explore fMRI, but due to limited time, we couldn't go there. 
1. **0.0003 public lb loost - Preprocessing for Known site 2 samples** 
We tried to perform a gaussian fit on IC and fnc features for both site1 and site2.The std was nearly same, so, it looked like we can approximate the site 1 features by just adding mean shift to site 2 features. 
```
Mean Shift = Mean(Site 1 samples) - Mean(Site 2 samples)
Generated Site 1 sample = Site 2 sample + Mean Shift
```
And then the prediction for known site 2 samples was done using the generated site 1 sample.

 2.**Calculating No of site 2 samples**
We simply solved this equation for all the features that were showing significant shift and plot the histogram with x values obtained from all features:
```
Mean(Unknown Samples) = x * Mean(Site1 samples) + (1-x) * Mean(Site2 samples)
Here, x is the percentage of site 1 samples in the unknown dataset.
```
It was clear that 20-40% of the dataset belongs to site 2. 

3.**0.0002 public lb loost - Postprocessing for Unknown site 2 samples** 
Our site classifier was ensemble of:
- LightGBM on all features
- Stacked Ensemble on all features with generated samples. The additional site 2 samples were generated from known site 1 using mean shifts.
- Bayesian Model on top 30 features. 

The individual probabilities were on different scale, specially for model 1, due to class imbalance. So, Ensemble was done by ranking the samples according to their probabilities and then taking the weighted mean of all the ranks. 
Top 500 ranks were assigned probability 1. Next 500 were assigned 0.9. Next 0.75.
Similarly at the bottom, samples were assigned 0, 0.1, 0.25.
All the rest samples were assigned probability 0.5

On private set, the last one actually worked better than blend but we didn't select that. 

With the probabilities, final prediction were calculated as:
`Prediction = p1 * site1_pred + p2 * site2_pred`
where site1_pred was the actual prediction by the model
and site2_pred was the prediction made assuming the sample comes from site2, it was done in manner similar to (1)


4.**Models** Blend Ensemble of 
1. Public LB with PP
2. 3-level stacking (43&gt;7&gt;1)
3. 3-level stacking (29&gt;5&gt;1)

Models used in Stack 2 &amp; 3: Ridge, Bayesian Ridge, Bagging Regressor, SVR, NuSVR, Kernel Ridge, Neural Networks, Lasso, E-Net, GLM

All three scored 0.1585 on public lb, blend ensemble gives **0.1584**

I am new to writing detailed solution. So, pardon me for any mistakes!!

**More Details about the stacking model and some more tricks that worked will be posted soon. **

**Full Detailed Solution with code will be uploaded soon**
