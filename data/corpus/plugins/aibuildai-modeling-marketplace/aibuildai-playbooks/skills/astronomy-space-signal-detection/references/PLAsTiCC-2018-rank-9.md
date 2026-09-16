# 9th place solution

Competition: PLAsTiCC-2018
Rank: #9
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75316

Thanks to everybody for this competition. Besides enjoying it greatly it has been extremely instructive for me. Thanks also to my teammates [Siddhartha][1] and [Ynktk][2] for their great spirit, hard work, and ideas. Finally, a big thanks to those who have shared their kernels and their thoughts in the discussion board throughout the competition. The open flow of ideas enrichened the competition invaluably.

Our solution relied primarily on careful feature engineering and ensembling. Below I present an overview of it. 

**Stacking**
Our best submissions were obtained by stacking the predictions of different lgb, catboost, and nn models, and then taking a weighted average with our previous best submission. This last step improved the score by 0.02 with respect to submitting the meta-model predictions alone (for our last submissions).

As meta-model we initially used a shallow lgbm with depth 1 and learning rate 0.01. At the end of the competition we increased the depth to 3. 

**First-level models** 
My teammate Siddhartha focused mostly on a densely connected CNN model. You can read more about his ideas [here][3]. Ynktk created a variety of nn, xgb, lgb and catboost models. He will add some notes on these later in this thread. We also experimented a little with regularized greedy forests but they did not add much to the ensemble. 

Personally I focused on a single first-level lgb model. This was split into two: one was trained on galactic objects, and the other on extragalactic ones. Each training set used different features. This was the model with best lb score (0.863) of our ensemble as far as we know.

**Features** 
I engineered around 8000 features and used lgb importance to select the top 80 for the galactic set and the top 130 for the extragalactic set (approx). Some notes on this part:

 - First of all I would like to point to @manugangler's [kernel][5], where he provided a method for fitting light curves to microlensing events. The features obtained from there worked wonderfully for us, especially after taking passband-wise ratios and differences for the microamp and microbase features, respectively. Einstein time is the most important feature of my model.

 - I browsed and read some texts on astrophysics which gave me some ideas for feature engineering. Below I comment the most useful ones for my model. 

  - `distmod/log10(hostgal_photoz)`. If I understood correctly, this roughly measures how far from being a so-called 'standard candle' the object is (setting standard candles to be 'standard' supernovae). The actual formula is more complicated but for a tree-based approach this suffices. This is the 2nd or 3rd most important feature of my model.
  
  - `log_10( luminosity_mean )` where `luminosity = 4pi * (distance in parsecs) * flux` (I may be very confused and this may have nothing to do with luminosity). Some similar feats explained by Kyle Boone like magnitude `-2.5log_10(max flux in passband i) - distmod` and their ratios. There are several other feats. If someone is interested I can explain further.

 - I used different sets of time observations for feature agreggation. One set used all observations, the other only the ones marked as detected, and the other only the ones marked as undetected. After aggregation I took ratios passband-wise and also I created some other ratio variables such as `passband_std_undetected/mjd_diff_detected` and many others.

**Some notes**

- Siddhartha augmented the train set by randomly selecting from 30% to 70% time observations for each object id, and then creating new objects with the selected data. This improved his nn cv score by 0.01. It did not seem to improve my lgbm. Looking now at other solutions it may have been a good idea to spend some more time thinking on this idea.

- Class 99: we used Olivier's and Cpmp's method throughout the competition. The less than 5 submissions we used for probing this class were unsuccessful.

- We tried to model hostgal_specz but this did not work for us.

- Pseudolabeling also did not work.

- Our public lb top score was 0.792. However the best score obtained by one of our single first-level models was 0.863. Hence I believe one of the strong aspects of our solution is the diversity of models constructed, which may have been achieved thanks to the fact that each team member used mostly his own features and set up. These were quite different between the three of us. 

This is all. Thank you for reading! If you have any question I will be pleased to answer


  [1]: https://www.kaggle.com/meaninglesslives
  [2]: https://www.kaggle.com/naka2ka
  [3]: https://www.kaggle.com/meaninglesslives/a-slightly-better-nn-arch-and-some-tricks
  [5]: https://www.kaggle.com/manugangler/optimal-feature-extraction-for-class-6
