# 5th place solution

Competition: learning-equality-curriculum-recommendations
Rank: #5
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394827

First of all, thanks to Kaggle, The Learning Agency Lab for interesting problem and kagglers for strong competiton!

You can find inference code example notebook there: https://www.kaggle.com/code/churkinnikita/lecr-example-0-705-public
It uses only 1 [SBERT + LightGBM] model and receives 0.705 public / 0.741 private and 11th place on the final LB.

**Validation setup**
From LB probing we know that there are approximately 9% of new channels (graphs) in the test set so I tried to mimic this logic.
I created 7Fold validation scheme when for every training fold we have 9-10% of unseen channels in the corresponding validation fold. But instead of 7 folds I used 1 or 2 folds to check improvements almost all the time. Correlation to the public LB was perfect. After some time I switched to classic 5Fold scheme (based only on topics ids) because learning that way led to better results on public (train set and test set are mixed). For the Stage2 model I used exactly the same folds for validation. For computing F2 score I filtered only "non source" data.

**Stage 1**

I basically used [SBERT](https://www.sbert.net/) package for learning stage1 model.
My solution includes usage of 2 models: `paraphrase-multilingual-mpnet-base-v2` with long training (250 epochs) for every language and `all-distilroberta-v1` for English language (learned only on English subset) with shorter training time. Full solution scheme is depicted below:


[MegaBatchMarginLoss](https://www.sbert.net/docs/package_reference/losses.html#megabatchmarginloss) was chosen as loss function, batch sizes in the range 270-310 provided the best performance.

Text input for topics was computed according to this formula:
`topic_text_input = language + channel + category + level + topic_title + topic_description + context(aka breadcrumbs) + parent_description + cousines_titles + children_titles`. 

Inputs for contents is much simplier:
`content_text_input = language + kind + content_title + content_description + content_text`. 

I created an esquisse to illustrate how topic text input looks like:


During training I used [different weights](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) for "source" and "non source" data (0.35 and 0.65 respectively) to sample "non source" instances more often: thery are much harder for the model to predict correctly.

New [Lion](https://arxiv.org/abs/2302.06675) optimizer led to very good optimization results out-of-the box but carefully tuned good old `AdamW` won in terms of the final F2 score (but required much longer training time).

Training for 250 epoch was extremely long: ~40 hours for 1 fold. I even bought RTX 4090 to be able to compute everything before the deadline but didn't manage to do that: for 250 epoch-model I computed only 3 folds (out of 5) and all-data model.

**Stage 2**

Second stage involved looking up for top 100 nearest contents for every topic in the corresponding language (for example, for topic in French we search only in French contents). After retrieving the list of possible candidates I generated ~30 features based on distance, language, text similarity [Jaro score](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) between titles, number of shared words), "geometry" of distance space, etc.

There is list of best features with explanation:
1. rank – number of neighbor (or ranked distance for topic).
2. distance.
3. dist_std –  Variance of distances to contents (for particular  topic).
4. dist_min – Minimum of distances to contents (for particular  topic).
5. dist_range – `dist_max – dist_min`.
6. dist_jump – argmax of distance differences for topic: where (on what neighbor number) the biggest “jump” in distances occurred.
7. dist_apart – `rank – dist_jump`.
8. dist_max_change – maximum change in distances to contents (for particular topic).
9. margin_forward – `distance(topic, nearest_content[i+1]) - distance(topic, nearest_content[i])`.
10. margin_backward – `distance(topic, nearest_content[i]) - distance(topic, nearest_content[i-1])`.
11. dist_mm – MinMaxScaled distances for topic.
12. margin_backward_mm – analogue of margin_backward but for dist_mm.
13. topic_cumsum_dist  –  cumsum of distances for given topic: `candidates.groupby('topic_id')['dist'].cumsum()`.
14. topic_language.
15. topic_level.
16. topic_len – length of topic’s title.
17. jaro –  [Jaro similarity score](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) between topic's title and current content's title.
18. topic_max_jaro – maximum Jaro feature for topic.
19. topic_median_jaro – median Jaro feature for topic.
20. nshared_words – actually length of longest common substring between topic's title and content's title.
21. nshared_words_lower – same as nshared_words but in lowercase scenario.
22. is_max_nshared – does that particular pair (topic, content) have maximal nshared_words feature for that particular topic.
23. jaro_forward  – `jaro(nearest_content[i], nearest_content[i+1])`.
24. content_desc_isnull – if content’s description is null.
25. content_text_isnull – if content’s text is null.
26. content_min_dist – minimum distance for content: `candidates.groupby('content_id')['dist'].min()`.
27. content_max_dist – maximum distance for content.
28. content_diff_dist – `dist – content_min_dist`.

LightGBM in binary classification mode was used as the Stage2 model (we assign 0/1 labels for candidates based on the correlation file). I used only non-source data to train and evaluate GBM.

**Prediction**

For final prediction I normalized predicted probabilities using MinMaxScaling for every topic, something that looks like:
 `prediction.groupby('topic_id')['proba'].apply(minmaxscale)`

This approach allowed to search for the optimal threshold that doesn't depend on particular topic id.

Final solution is *majority voting* of models learned on full train set + fold models (5 models in total): content is considered relevant if it appears in recommendation of (at least) 3 out of 5 models.
