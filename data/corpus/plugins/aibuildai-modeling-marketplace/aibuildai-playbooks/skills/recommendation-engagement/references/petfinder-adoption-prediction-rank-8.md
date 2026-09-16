# 7th Place Solution Summary

Competition: petfinder-adoption-prediction
Rank: #8
Source: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88705#latest-515055

###[The code Is public now](https://www.kaggle.com/adityaecdrid/best-sub-selected-re-run-478)

First of all, thanks to Petfinder.my and Kaggle for hosting this great and meaningful competition. And congratulations to the winners!
And it's the first time for many team members to grab the Gold!!
Thanks All :)

&gt;PS Pardon me if it isn't upto the mark, as it's my first time writing the solution desc..

[Kernel link](https://www.kaggle.com/adityaecdrid/best-sub-selected-re-run-478)

The main issue with the PetFinder JSON Parsing script shared on public was that it wasn't accounting for many other json keys that were there and when you will add this, the script will break and this is what Shaz meant... (though the file having such contents were quite less, but we still added them because in case the test set had them in depth). Also we used all the files, not just the $-1$ tagged for everything (images, sentiments and metadata iirc)

## FE's (Mainly, handcrafted)
Now coming to FE's, well we did a lot of work on that as well.
- We tried cleaning the names col
- Added Relative age feature [line](https://www.kaggle.com/adityaecdrid/best-sub-selected-re-run-478#L362)
- Also many interaction b/w Health, Sterelized etc cols as well :)
- We also tried to mimic the way PetFinder.my adds the rating to their images(i created an account on their website to try to see it myself as well), we called it [SEO Features](https://www.kaggle.com/adityaecdrid/best-sub-selected-re-run-478#L380)
- After that we also ranked them accordingly as if someone visits the website, the pets are ranked somehow, so we just tried to mimic the same [rankbyG](https://www.kaggle.com/adityaecdrid/best-sub-selected-re-run-478#L399)
- Also we added a lot of aggregates as well on basis of rescuer id/state as well ( i guess this worked for the ID because since the rescuer id was auto-generated, it had the TS effect to it hidden as well because if you will select first 4-5 chars and sort the df, you will see the same, like earlier joinied users might save more etc kinda analogy)
- Next we added [nlp feats](https://www.kaggle.com/adityaecdrid/best-sub-selected-re-run-478#L556) as well which were borrowed from Quora's Shaz's work.. We tried POS tagging as well but didn't add it in the final solution as it was time consuming (but it helped improved the model)
- You can see [this](https://www.kaggle.com/adityaecdrid/best-sub-selected-re-run-478#L804) for many basic features we added as well, basically adding many groupby feats, states stats, cat/dogs breeds stats, ranking, insta features, year, top breeds, pets alllowed according to malaysian laws etc..
- Next we also had [image_dims_feats](https://www.kaggle.com/adityaecdrid/best-sub-selected-re-run-478#L993)
- We also added [NIA](https://arxiv.org/abs/1709.05424) features as well (image quality, this ranked way higher for us)
- Added features to indicate the dullness/brightness (refer Avito Comp, Peter's solution and great kernels their)
- Also Lucas did a great job by verifying the CV with double CV strategy as well..

- Modelling includes lgb+xgb with different CV strategy and different STD as well for better ensembles.

-  And there are ton of things which didn't work as well....

(Will be updated with the CV used and other things as well by team mates as well)

Thanks And Happy Kaggling:)


&gt; Would request team-mates( @shaz13, @lucamassaron, @init27, @init927, @backaggle) (in no order) to drop whatever i missed
