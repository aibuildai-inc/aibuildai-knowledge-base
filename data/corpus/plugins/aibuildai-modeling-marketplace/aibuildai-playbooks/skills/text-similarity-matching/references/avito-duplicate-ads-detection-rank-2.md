# 2nd Place Solution: TheQuants

Competition: avito-duplicate-ads-detection
Rank: #2
Source: https://www.kaggle.com/c/avito-duplicate-ads-detection/discussion/22205

Hi everyone!

Here's a description of our team's efforts for this competition. I think this competition was very much driven on feature engineering, with meta-modelling more as a finishing touch.

**Our Strategy**

 - Understanding the Data, developing features until team merger deadline, metamodelling in the last week.
 - Merging early based on standing of the leaderboard at the time
 - Each individually building features to try and branch out as much as possible, with constant discussion about where to look next
 - Onboarding an experienced w̶i̶z̶a̶r̶d̶ Kaggler for the final stage. 

----------

**Data Cleaning:**

In order to clean the text, we applied stemming using the NLTK Snowball Stemmer, and removed stopwords/punctuation as well as transforming to lowercase.

**Validation Strategy:**

Initially, we were using a random validation set before switching to a set of non-overlapping items, where none of the items in the valset appeared in the train set. This performed somewhat better, however we had failed to notice that the training set was ordered based on time! We later noticed this and switched to using last 33% as a valset. This set correlated relatively well with the leaderboard until the last week, when we were doing meta-modelling and it fell apart - at a point where it would be too much work to switch to a better set. This hurt us a lot towards the end of the competition.

##Features:
In order to pre-emptively find over-fitting features, we built a script that looks at the changes in the properties (histograms and split purity) of a feature over time, which allowed us to quickly (200ms/feature) identify overfitting features without having to run overnight xgboost jobs. If there is interest, I would be willing to put it on github. 

After removing overfitting features, our final feature space had 587 features in it. Here’s a summary:

**General:**

 - CategoryID, parentCategoryID raw CategoryID, parentCategoryID one-hot
   (except overfitting ones) 
 - Price difference / mean
 - Generation3probability (output from model trained to detect   
   generationmethod=3)

**Location:**

 - LocationID & RegionID raw
 - Total latitude/longtitude
 - SameMetro, samelocation, same region etc.
 - Distance from city centres  (kalingrad, moscow, petersburg, krasnodar, makhachkala, murmansk, perm, omsk, khabarovsk, kluichi, norilsk)

Gaussian noise was added to the location features to prevent overfitting to specific locations, whilst allowing xgboost to create its own regions.

**All Text:**

 - Length / difference in length
 - nGrams Features (n = 1,2,3) for title and description (Both Words and Characters)
- Count of Ngrams (#, Sum, Diff, Max, Min)
- Count of Unique Ngrams
- Ratio of Intersect Ngrams
- Ratio of Unique Intersect Ngrams
 - Distance Features:
- Jaccard , Cosine, Levenshtein and Hamming Distance between the titles and descriptions
 - Special Character Counting & Ratio Features:
- Counting & Ratio features of Capital Letters in title and description
- Counting & Ratio features of Special Letters (digits, punctuations, etc.) in title and description
 - Similarity between sets of words/characters
 - Fuzzywuzzy/jellyfish distances
 - Number of overlapping sets of n words (n=1,2,3)
 - Matching moving windows of strings
 - Cross-matching columns (eg. title1 with description2)

**Bag of words:**

For each of the text columns, we created a bag of words for both the intersection of words and the difference in words and encoded these in a sparse format resulting in ~80,000 columns each. We then used this to build Naive Bayes, SGD and similar models to be used as features.

**Price Features:**

 - Price Ratio
 - Is both/one price NaN
 - Total Price

**JSON Features:**

 - Attribute Counting Features:
- Sum, diff, max, min
 - Count of Common Attributes Names
 - Count of Common Attributes Values
 - Weights of Evidence model on keys/values
XGBoost model on sparse encoded attributes

**Image Features:**

 - \# of Images in each Set
 - Difference Hashing of images
 - Hamming distance between each pair of images
 - Pairwise comparison of file size of each image
 - Image dimension Features:
 - Pairwise comparison of dimension of each image
 - BRISK keypoint/descriptor matching
 - Image histogram comparisons
 - Dominant colour analysis
 - Uniqueness of images (how many other items have the same images)
 - Difference in number of images

I found a possible image metadata leak (the creation dates of the images were embedded in the zip files) but didn’t try to use these for features - gotta play fair after all!

**Clusters:**

We found clusters of rows by grouping rows which contain the same items (eg. if row1 has items 123, 456 and row2 has items 456, 789 they are in the same cluster). We discovered that the size of these clusters was a very good feature (larger clusters were more likely to be non-duplicates), as well as the fact that clusters always the same generationMethod. Adding cluster-size features gave us a 0.003 to 0.004 improvement. It would be interesting to know if anyone else found these features :)

###Feature Graveyard:
Overfitting was probably the biggest problem throughout the competition, and lots of features which destroyed in validation didn’t do so well on the leaderboard. This is likely because the very powerful features learn to recognise specific products or sellers that do not appear in the test set. Hence, our feature graveyard:

**TF-IDF:** This was something we tried very early into the competition, adapting our code from the Home Depot competition. Unfortunately, it overfitted very strongly, netting us 0.98 val-auc and only 0.89 on LB. We tried adding noise, reducing complexity etc. but in the end we gave up.

**Word2vec:** We tried both training a model on our cleaned data and using the pretrained model posted in the forums. We tried using word-mover distance from our model as features, but they were rather weak (0.70AUC) so in the end we decided to drop these for simplicity. Using the pre-trained model did not help, as the authors used MyStem for stemming (which is not open-source) so we could not replicate their data cleaning. After doing some transformations on the pre-trained model to try and make it work with our stemming (we got it down to about 20% missing words), it scored the same as our custom word2vec model.

**Advanced cluster features:** We tried to expand the gain from our cluster features in several ways. I found that taking the mean prediction for the cluster as well as cluster_size * (1-cluster_mean) provided excellent features in validation (50% of gain in xgb importance), however these overfits for reasons I still have to investigate. We also tried taking features such as the stdev of locations of items in a cluster, but these also overfitted. I suspect that Avito used slightly different methods for generating the test set.

**Grammar features:** We tried building features to ‘fingerprint’ different types of sellers, such as usage of capital letters, special characters, newlines, punctuation etc. However while these helped a lot in CV, they overfitted on the leaderboard.

**Brand violations:** We built some features based around words that could never appear together in duplicate listings. (For example, if one item wrote ‘iPhone 4s’ but the other one wrote ‘iPhone 5s’, they could not be duplicates). While they worked well at finding non-duplicates, there were just too few cases where these violations occurred to make a difference to the score.

##Meta-model:
Using a meta-model improved our score by roughly 0.0015 versus our best XGBoost. In the end, a single model would have been enough to net us second place, but you can never be too careful!

Below is an overview of what our final meta looked like. Each generation is different set of features, with later generations having more features. Note that all scores are private LB scores:

![Meta-model schematic][1]

We also tried using Extra Trees, Random Forest, Adaboost & approximate kNN as base models, however these overfit to our validation-set and so they couldn’t be used in the final meta-model.

----------

If anyone has any questions about our solution, we would be happy to go into more depth!

**Finally, a great big thank you to my teammates Peter, Sonny and Marios for making TheQuants a brilliant success!** We couldn’t have done it without every one of you.

And thanks to everyone else for making this such an enjoyable competition

 \- Mikel,  
TheQuants


  [1]: https://files.slack.com/files-pri/T18TSG1C3-F1QTWCZ5G/avito-meta.png?pub_secret=14957b8a3e
