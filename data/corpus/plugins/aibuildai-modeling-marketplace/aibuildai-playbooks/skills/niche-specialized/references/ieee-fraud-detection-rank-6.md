# 6th place solution: The Zoo

Competition: ieee-fraud-detection
Rank: #6
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111247

Thanks a lot to Kaggle and IEEE to hosting this competition, I hope that we will have another tabular competition soon enough. Big shout out goes to all my teammates @dott1718, @christofhenkel, @ebouteillon, and @maxjeblick it was really a pleasure to working with them.

From reading the solutions that have already been posted and from expecting what others did, I am pretty sure that most did (similar as in Santander) a very similar thing with some fine-grained tricks. Our solution has maybe 2-3 tricks which might be different to what others did. I am a bit tired, so sorry when not all things are super precise :)

**User IDs**

As everyone should know by now, identifying users is the key to achieving a good score. We did a lot of trial and error here, and @ebouteillon and @dott1718 were monumental in our identification of users. When we merged we combined our two approaches and in the end @dott1718 crafted a very elegant and well-working solution for user identification. You can find the script of how to do it here (let me call those ids uid6 from now on): https://www.kaggle.com/dott1718/ieee-userid-proxy

Our models used different variations of the user id, with some simpler forms that have much less unique values ranging up to the ids posted before, which are very restrictive but also the most precise ones. 

However, now comes the bit more trick part of choosing the CV setup. There are different ways of doing it, and there are different types of leaks you then introduce in your CV setup. For example, one way is to do monthly time-split leak which is also what we chose to use in the end as imho this CV kind-of represents test data the closest. We chose to look mostly at average CV across folds. The important fact here is though that you have ID overlaps between folds and you have more overlaps the more folds you have around your validation fold. IDs overlap way more in closer time proximity. So for example, Fold0 (Christmas) and the last fold perform worse than those folds in the middle, because they have more IDs overlapping to the training folds.

What is then also important is to understand how this overlap will behave in test:

 

Here you see different IDs (let’s focus on UID6) and on the x-axis the different months of test. The y-axis is the percentage of unique IDs in this month that we also observe in train. As the performance of the models is way better on overlapping ids compared to non-overlapping ones, we could expect private LB to have much lower score than public LB. This also means that it is more important on private LB to have a well working model for non-overlapping customers that you have never observed before in training.

As a result of this, we decided to split our efforts into both developing strong models for overlapping IDs and non-overlappping IDs. We created for both our CV setup as well for test two indices which both mark the overlaps, and the non-overlaps. For example, in a validation fold those transactions where we observe the ID in the other folds would be marked as overlap, and all others as non-overlap.

**Features**

To be honest, this is probably our weak point as we did not spend too much time on this. Maybe we could further improve our results with better features, but I don’t know. We use a lot of count encoded features (also the uids), as well as other aggregated features. We also introduce aggregated features on UID level. Overall this should not be too different compared to some public kernels.

**Models**

We utilize three types of models: LGB, Catboost, and Neural Networks all fitted on monthly CV setup. All models use pretty much the same / similar features with some differences of course. In Catboost it was necessary to include the IDs as explicit categorical variables to make it work. Neural networks of course needed some feature transformations and other tricks. @christofhenkel and @maxjeblick did most of the work there, so they are probably the best to elaborate on this further. We additionally did a bunch of pseudo-tagged LGB models where we used predictions from only overlapping UIDs (because we are confident with them) as either soft or hard labels. We use them partly in the blend, but hard for me to say if they help much.

**Blending**

Our blending is now probably the interesting part. First of all, we decided to do hillclimbing blending, mostly using mean or gmean on raw predictions. But instead of optimizing for overall CV, we optimize separately for overlapping and non-overlapping CV. So first, we only optimize the CV for those indices which we have marked as non-overlapping, and then we do the same for the overlapping indices. The cool thing is now that different models come into the mix for those different indices. 

LGBs usually work well for both parts, so they are part of both overlapping and non-overlapping blends. Neural networks help significantly for the non-overlapping part, but not so much for the overlapping part because they don’t overfit heavily on the IDs and it is even hard to force them to. Catboost models work well on both parts, but have some issues on the non-overlaps leaking too much which doesn’t work well on test, so they are only part of overlaps. To summarize: Overlap Blend consists of LGBs + Catboosts and nonoverlap blend consists of LGBs + NNs.

Now the next tricky thing is how to combine them? One way that surprisingly works well is to just fuse them by raw probability. Actually, most of our final models just do it like that. We had some other fancy methods for fusing, starting from simple mean adjustments, towards a tricky rank-based fusing approach that @christofhenkel came up with. The idea is roughly to rank all predictions using only the coverage model, take the nocoverage part and rank again, build a mapping of those ranks to overall ranks and apply it to the nocoverage ranks. 

**Post-processing**

What further helped us is the following trick. Take all predictions from a customer based on our UID6, and combine the predictions to a single value so that all transactions of a customer have the same value. This gave both a nice boost on CV as well as on LB. I think this is also something that helped us quite significantly on private LB because it makes the ranking much more precise by grouping customers together. One way is to just take the mean of the predictions, what we mostly did is to take 90% quantile of the predictions as this usually worked best on CV.

**Final submission**

Our blends had a certain random range on both CV as well as LB and as we could not fully decide which one to pick we made a blend of blends by taking rank-average of a bunch of our top blends we had. This was a good choice and it is nearly our top private LB solution we picked by doing so. 

In our second submission we tried to gamble a bit more by focusing even more on the non-overlaps. So we did a different CV setup splitting by UID instead of by month, so we had little to no UID leaking in the CV. Then we set the nonoverlap predictions based on a blend of that CV setup. It only scored a bit lower on private LB.
