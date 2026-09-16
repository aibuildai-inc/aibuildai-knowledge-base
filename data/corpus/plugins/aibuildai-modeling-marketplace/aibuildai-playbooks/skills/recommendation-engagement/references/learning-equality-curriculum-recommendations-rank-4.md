# 4th place solution

Competition: learning-equality-curriculum-recommendations
Rank: #4
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394984

Thanks a lot to Kaggle and the hosts, specifically @jamiealexandre, for hosting this competition. Unfortunately, we only could join after NFL competition finished, so we tried to do our best within an eleven days sprint, and we are very happy about the outcome. Improving every day can be quite thrilling - but at the same time we regret a bit not having more time for this competition as it was really interesting and fun participating here, and I believe our solution has not yet reached its ceiling. 

Due to time constraints, our solution is based only on an ensemble of embedding models with cosine similarity matching as well as additional post-processing.

### Data processing

We only use the following features as text input for out models: language, category, title and description
An "issue" with the data is, that some topics and content items do not have a proper description, or very inconclusive titles. So it is helpful to supplement them.

For topics, we replace the title with an inverse track of the category tree titles, so basically adding the breadcrumbs, we do the same for description. 
For content, we concatenate the description and text fields as a single column.

The benefit of this approach is, that it will add information to those records that have incomplete data. With tokenization and truncation, the models will prioritize the original information, if available, and otherwise use the supplemented information. So for instance, if a content item has description and text available, it will prioritize the description, but otherwise if description is missing, it will use the text.

### Validation setup and models

Due to lack of time, we focused on a single holdout. For this, we split out non-source categories. Contents matching to these categories can then be either already be part of training, or completely unseen. We got strong correlation with this setup to public and private leaderboard in relative terms.

For submissions, we always retrained models on full data. As we were running out of time, we could not always do both a validation and fullfit. So 3-4 days before end, we only did blind fullfits, and blended them on submission. And for local validation and testing post-processing we relied on some earlier models. While not ideal, it was a reasonable approach given the time constraints.

### Embedding models

We only use ArcFace models. For input and training the models we use two different schemas:

**Topic-based labeling.** Here, a single label is defined as a topic, and all content items that match to this topic. So for instance: Label 1: Topic A, Content A, Content C, ...

This means that content items occur as many times as they match topics, and each topic is only a single sample. This approach is strongest on its own with the F2-based metric.

**Content-based labeling.** Here, a single label is defined as a content item, and a topic that matches to this content. So for instance: Label 1: Content A, Topic A

This means that each topic-content pair is a single label. This method worked worse individually, but blended quite nicely with the topic-based approach.

Our final blend contains 7 topic-based, and 2 content-based models. Backbones are mostly xlm-roberta-base, xlm-roberta-large, paraphrase-multilingual-mpnet-base-v2, or deberta-v3-large.

### Post-processing
We played a lot with different post-processing techniques as this is always something that is useful in metric-learning matching. We optimize the treshold automatically in the kernel to a certain average number of matches per topic. Also, we found that penalizing the cosine similarities based on additional information helps. First, we slightly reduce similarity probabilities for content that only matches to a single topic in whole training. Second, we increase the probability of content items that are not available in training. We also always match new content to the top ranked topic, if above a certain threshold. Finally, we also additionally add new matches if we have less than five matches for a topic, but the additional probabilities are above a certain ratio to the higher ranked probabilities.

### What did not work (due to time)

We spent 2-3 days trying to tune bi-encoder text models for second-stage, but could not get anything that improved our first-stage embedding models to be worth the additional runtime. So we decided to drop it and focus on first-stage only.

Also, we spent some time on trying to tune LGB second-stage models. We were quite sure that they should be working well, and might replace also some manualy post-processing. But while CV looked reasonable, LB was dropping a bit, and we were not too confident in the validation setup for it, so we dropped it. Seeing other solutions, it definitely seems to be helpful, and I believe it could push our solution higher. 

### Efficiency sub

We also have an efficiency sub scoring 0.72 ensembling two smaller models on shorter token lengths running in 22 minutes. We use multiprocessing and ONNX. We probably lack a good 2nd stage LGB model to boost the score higher here. 

As always, cheers to my amazing team-mate @ilu000.

All training and inference code can be found [online](https://github.com/psinger/kaggle-curriculum-solution).
