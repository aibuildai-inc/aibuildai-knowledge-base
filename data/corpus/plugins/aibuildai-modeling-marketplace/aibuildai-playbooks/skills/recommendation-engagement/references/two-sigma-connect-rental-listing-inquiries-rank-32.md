# Solution share (#32 place)

Competition: two-sigma-connect-rental-listing-inquiries
Rank: #32
Source: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32364

Hi everyone!

This competition has been a great experience. I really liked the dataset (apart from the leak :)), which can inspire so many ideas that one could work on it for months and still get new ones.

First of all, congrats to all gold medalists, and in particular to plantsgo and LinuX18 for making such great debuts, to Faron for his impressive sprint, and to KazAnova for having such a big influence on the competition.

## Feature engineering

My solution uses mostly ideas, which have been discussed so far. In particular, I have:

 1. Cleaned the display address and street address
 2. Joined duplicates in features, e.g.: 'laundry', 'laundry_in_building', 'laundry_in_unit', 'laundry_room', 'on-site_laundry', 'washer_in_unit‚ 'washer/dryer
 3. Added features:
   - price_over_rooms = price / (1 + 0.8 × num_bathrooms + num_rooms)
   - price × price_over_rooms
   - price × price × price_over_rooms
   - going on with the above didn’t help :)
   - added 'half-bathrooms' as a feature
 4. Rotated latitude/longitude – New York is not 'aligned' horizontally
 5. Used clustering algorithm (one-hot-encoded) from this kernel: https://www.kaggle.com/anki08/two-sigma-connect-rental-listing-inquiries/trying-feature-engineering

Apart from these basic features, I have concentrated my efforts on dealing with manager_id. My idea was to handle it in many different ways to ensure diversity of the base models used for stacking. To this end, I have used one-hot-encoding, likelihood estimation from the 'it_is_lit' kernel, mean-target-encoding (which is a particular case of the former), 'cv-statistics' as described here: https://www.kaggle.com/guoday/two-sigma-connect-rental-listing-inquiries/cv-statistics-better-parameters-and-explaination
but in the form given in comments, not the one in kernel. Namely, I have used median and std of 15 features (like price_over_tot) grouped by manager_id and interest_level. I have additionally used likelihood estimation over building_id, display_address, and street_address (summed together, and with slightly different parameters than in the original kernel). Finally, I have used the mean interest level, its std and number of listings of managers in a 24 hour interval (this was used on top of the 'cv-statistics' models). 

All the above (apart from one-hot-encoding) required a nested cv loop. I have used 10 folds for both inner and outer loops and the same (outer) folds for all methods and levels. (for the time features I have used a 10x9 scheme)

## Stacking

I had about 40 models in the first level of my stack - I have also used regression for all of them, and one-vs-all (instead of multiclass) classification for the one with one-hot-encoding.

My best 1-level model - the one with statistics of price_over_tot grouped by manager_id and interest_level - scores: 0.5065 (cv) / 0.51556 (private) / 0.51567 (public). 

My idea to ensure diversity in the models was to incorporate StackNet somehow into my stack. I have tried a few schemes, and ended up with using StackNet on the second level. I was feeding it with predictions of 5 first level models (regression + classification) chosen by their score and correlations. These 5 models were added as meta-features to StackNet separately (so 5 different StackNet runs) and together (additional 'all-in-one' StackNet). The all-in-one StackNet scored 0.50302 on the private leaderboard, which is already close to my final result of 0.50068. In my final approach I have used the 1 level StackNet models as my 2 level models. In this manner I had ~40 models in the first level of my stack, and ~50 in the second level (which also contained XGB regression on all 1-level models with MAE and RMSE). Therefore, my first two levels were a bit overblown, and for my final predictions I have used XGB on both 1 and 2 level models with recursive feature elimination (amounting to 0.001x improvement). The elimination was based on 'weight' of the features summed up over all folds. I have also tested this procedure with 'gain', but it seemed to perform worse. Finally, I have collected 5 best models obtained during the elimination procedure and averaged their output (0.0002 improvement). I have used averaged predictions of the models produced on the cv folds (the models trained on all the data were only slightly worse).

I have only used StackNet with the original feature set, because most of my feature sets were computed per fold and the original StackNet version didn't support that.


## Other interesting things I have tried (but didn't work)

1. Entity Embeddings of Categorical Variables, https://arxiv.org/abs/1604.06737, on manager_id. Didn't help but I really like the concept.
2. Learning to Generate Reviews and Discovering Sentiment, https://arxiv.org/abs/1704.01444. I have checked it only for a few listings, but it seemed not to produce any signal and, since it was a bit late in the competition, I have dropped it quickly.

## Conclusions

* My main problem was the lack of a well-performing single model. The best one scored LB0.515x. With stacking I was able to go down to LB0.500x, but I expect the result would have been much better with a good single model in the stack. I think that adding other statistics grouped by manager_id (without grouping by interest_level) and other features (e.g. rooms) would improve it a lot.
* Including other methods on top of my feature sets in the stack and including NN as final classifier would have helped a bit. In the end, I have only used XGB and StackNet. I have tried using LightGBM, which is indeed faster, but didn't know how to get evaluation history, which was necessary for my cv scheme to work, so I dropped it (also because, as another boosted tree method, it wouldn't lead to much model diversity anyways).
* I also think that having 10 folds in my cv scheme was unnecessary. All my models took about 2 full days to train on a laptop, which was a bit problematic in the final days of the competition with the leak and a few features that I wanted to add quickly.

I have started this competition with close to zero practical machine learning experience and, as a result, I have lost a lot of time in the beginning on trivial things like (over)tuning the 1 level models (which actually had an inverse effect on the score of the stacked models) or dealing with the fact that the classification is ordinal - I have generalized the approach in http://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf. Namely, I have used 1-vs-all classifiers and constructed the final probabilities from them: 

p_{low} =  alpha_{low} × p_{is low} + (1 - alpha_{low}) × [(1 - p_{is high}) – p_{is med}], 

where alpha_{low} is an additional parameter. I have found the optimal alpha_{low, medium, high} parameters through minimization of the cv result (this has actually improved my score!). I have spend about a week on this problem just to find out that it can be efficiently handled by stacking! :)

My improvement of skills during the competition would not have been so efficient without help from the community. Many thanks to all who contributed so much in the kernels and discussions! In particular (but not only) to rakhlin, gdy5, KazAnova, den3b, SRK, darnal, Ankita Sinha, and Branden Murray.
