# 11th solution - very basic but may different methods

Competition: PLAsTiCC-2018
Rank: #11
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75174

Here is my brief summary that may be different from others:   

---   

## Feature Engineer   

* **Features inspired by Starter Kit**   
Not only construct features between adjacent passbands as starter kit showed, like between passband 0 and 1, 1 and 2. But also between passband 0 and 2, 0 and 3, and so on. Besides,  not only mean flux, but also try max flux and min flux. Finally, features like   
```
log((passband_1_max_flux) / (passband_3_max_flux))
```   
contribute a lot to score.   

* **Features I've tried that useful**  
1) 1st quartile and 3rd quartile flux per passband. And their gap value relative to full scale, like   
```
(passband_0_q3_flux - passband_0_q1_flux) / (passband_0_max_flux - passband_0_min_flux)
```   
2) 3rd quartile flux per passband multiply square of hostgal_photoz, like   
```
passband_0_q3_flux * (hostgal_photoz**2)   
(passband_0_q3_flux - passband_0_q1_flux) * (hostgal_photoz**2)
```   
Those features work well on my model.  

* **Features from cesium and feets packages**   
period_fast feature per passband from cesium pkg and CAR feature per passband from feets pkg, made some contribution to my model.

* **Features referred to papers**   
[Here is the very important paper I've read and referred to](https://arxiv.org/abs/1603.00882). In this paper, authors introduced several method to extract  light curve parameters for classification. I referred to the most effective method of them -- salt2 model, in part 3.2 of the paper. Fortunately, you can use [sncosmo api](https://sncosmo.readthedocs.io/en/v1.6.x/examples/plot_lc_fit.html#sphx-glr-examples-plot-lc-fit-py) to fit salt2 model and get model parameters as features. 

* **Features referred to github**   
Bazin function, this method is same with what [4th solution](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011) has metioned, but I found it by this [link](https://github.com/ramp-kits/supernovae/blob/master/PLAsTiCC_starting_kit.ipynb). I've tried modify this method, but the result shows no big different.

* **Features referred to a public kernel**   
[This public kernel is very impressive](https://www.kaggle.com/manugangler/optimal-feature-extraction-for-class-6), but only got a few upvotes. Instead of use it as hostgal_photoz == 0 ( author set in the kernel), I use it for ext-gal model with a set hostgal_photoz &gt; 0. It really improved my model.   

* **Features selection**   
In order to remove some inappropriate features, I do it simple, observing some less important features' distribution on train and test. Use [this method](https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/) I've used before. Since there're class99 in test set, it didn't work very well, but still contribute a little.

---   

## LGB algorithm   

* **Modify Objective Function**   
Mithrillion has public [a very impressive kernel](https://www.kaggle.com/mithrillion/know-your-objective). But you may find well but not as good as just setting sample weights. So I add sample weights to grad and hess, it works better than built-in objective. This result in my final model structure:   
```
0.55 * (lgb with customized objective) + 0.45 * (lgb with built-in objective)
```
separated by gal/ext-gal, which means the final model contains 4 lgbs.


* **Tune Parameters**   
I tuned parameters to avoid overfitting,  small max_depth: 3, small max_bin: 20,  large min_child_weight: 10, large min_data_in_leaf: 35. These parameters' setting work well both on CV and LB.

---   

## Final Score   
When my CV gets better, there are three things that decreasing the gap between CV and LB: 1) adding salt2 model parameters (improve CV 0.033, but LB 0.058); 2) Remove inappropriate features; 3) Tuning Parameters. Finally, I got CV 0.411, public LB 0.794, gap 0.383, private LB 0.822, gap 0.411.

---

## What I've tried but failed, you may try it better   

* **Wavelet Decomposition**   
Again in [this important paper](https://arxiv.org/abs/1603.00882), part 3.4. Wavelet Decomposition is model-independent compared to salt2 model, but also effective. Three steps: GP -&gt; Wavelet Decomposition -&gt; PCA. I've tried, but failed. You may do it better.

---  

## Useful papers I didn't tried    
Here are links to what I've found may useful    
1) [Machine Learning-based Brokers for Real-time Classification of the LSST Alert Stream](https://arxiv.org/abs/1801.07323)   
2) [Deep Recurrent Neural Networks for Supernovae Classification](https://arxiv.org/abs/1606.07442)  

---

## Thanks   
As a novice in kaggle before this competition, I really learn a lot from those who public their impressive kernels, like [ogrellier](https://www.kaggle.com/ogrellier), [mithrillion](https://www.kaggle.com/mithrillion),  [kyleboone](https://www.kaggle.com/kyleboone),  [meaninglesslives](https://www.kaggle.com/meaninglesslives) ( this is my first time to use keras practicing NN). And [CPMP](https://www.kaggle.com/cpmpml), he gives lots of truly useful tips in discussion. [Giba](https://www.kaggle.com/titericz), his class weights discussion help us a lot . [sionek](https://www.kaggle.com/sionek) released the most famous 'detected_mjd' feature.  It's strange that those useful and original kernel have more fork times than upvotes, those truly helpful discussions got only a little upvotes. If you think they are helpful, just upvote.   
Besides, there is an abnormal thing :),  [Silogram](https://www.kaggle.com/psilogram) didn't post his famous 'a few notes...' as he did in previous competitions I joined before, like 1) [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/discussion/58332); 2) [Google Analytics Customer Revenue Prediction](https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/67767) . We can learn a lot from his notes, as CPMP do in this competition.   
They are the real fundamental of kaggle community !!
   
---
  
## PS
I just began to learn python and data science about 2 years ago ( I learn mechanical engineer in college). I think I've got lots luck in this competition, I'm not good enough. When I received teaming up invitations, I regarded them as an honor and encouragement,  Thank you !
