# congrats to the winners

Competition: home-depot-product-search-relevance
Rank: #3
Source: https://www.kaggle.com/c/home-depot-product-search-relevance/discussion/20427#117082

Here we would like to briefly describe our approach and main findings.

Our solution is actually a mix of two solutions: one coming from Chenglong and the other from me and Kostia. 

To a large extent, our approaches are similar to CrowdFlower winning solutions.

**Text processing:**

*Igor& Kostia:* using the Google replacements from the forum, some manual replacements plus automated spell-checker which uses product title as a vocabulary and difflib.SequenceMatcher() to get a measure of similarity between ‘misspelled’ word (i.e. not found in product title) and a potential correction. To generate some features we removed punctuation, removed stopwords (without ‘can’ which might mean ‘a container’) and stemmed the words. However, punctuation was important for tagging words in sentences and extracting the most meaningful parts from query and product title. 

We needed to make some words uniform. For example, both variants ‘mailbox’ and ‘mail box’ occurred in product title, query and other text information. In all text we made replacements for ~100 such words which can be spelled differently.

All words were tagged using NTLK.pos_tagger(), which was used for two purposes: 1) to separate nouns, verbs, adjectives, etc. and 2) to found the most important words in query and title. For example, in query ‘23.5 shower door nickel’ important words are ‘shower door’ and the most important word is ‘door’. We noticed a structure in product title which helped us separate the product and its attributes. For example, in product title ‘Husky 52 in. 10-Drawer Mobile Workbench with Solid Wood Top, Black’ the product is workbench, not wood top. I and Kostia started this work by observing irregularities in model predictions, but actually we come to something similar to [extracting the top trigram][1].

From attributes.csv we mostly worked with bullets (containing the text similar to product description), brand and material, but also used other information as well.

We also extracted brands and materials from all text field and then used separate parts of the string (e.g. whole string, string without brand or material, only brand or material) to generate count features and other features.

*Chenglong:* a separate, but similar processing. We believe that **different** text processing within one team was a key to getting a high score in this competition.

**Features:**

Largely the same as in CrowdFlower winning solutions. To avoiding repeating, here I just note some of key findings. The text with and without initial preprocessing was used to generate features. We also processed separately the words after ‘with’,’for’,’without’.

- Each time we wanted to generate a count feature, we actually generated a bundle of up to 5 features: number of distinct words in intersection, number of total words in intersection, number of letters in distinct words in intersection, ratio of words in intersection to the number of words query, ratio of letters in intersection words to the number of letters in query. I believe this was helpful to capture the information from the data. In fact, one of the most predictive our features was share of letters in intersection between query and product title. 

- Distance features included or incorporated Jaccard coefficients, Damerau–Levenshtein distances and sequence comparison (in terms of chars) with the help of difflib.SequenceMatcher(). The latter gave us two features in the top10.

- TFIDF features in different combinations. One of the most valuable are those where we used title or bullets for generate feature weights, then multiply them by the number of letters in the corresponding words, then calculate the sum of weights for words in query. In general, we observed that in this competition not only the word as a unit of information matters, but also the number of letters in the word is very important. To be precise, I must say we used stemmed words for TFIDF.

- Another important group of features are TFIDF (and some other features) calculated directly for products related to each unique search term.

- Word2vec: I and Kostia trained models on available text, Chenglong used pretrained models as well. These features proved to be very informative. Chenglong also used doc2vec.

- WordNet similarities between the different pairs of words from the most important trigrams for query and product title. We used [path similarity, Leacock-Chodorow similarity and Resnik similarity][2].  It gave us a few top20 features.

- Query expansion: queries grouped by the two last words in the top trigram, then we assume that the majority of relevances are closer to 3 than to 1. Then we find 20 the most frequent words in product description and calculate the percentage of how many of these words are in the product description for the matched product. This is one of our top10 features.

- Chenglong used LSA.

- Dummies for some brands and product names from the top trigrams.

- Me and Kostia ended up using dummy for id in part2 (see attached pdf).

**Models:**

Xgboost, gradient boost, random forest, SVR, extratrees. 

Although SVR model 0.01 higher RMSE than the best model, it was beneficial for the ensemble due to low correlation with other models.

**Cross-validation:**

Chenglong used an advanced split to be described separately. His CV was the same public and ~0.002 worse than private. 

I and Kostia used StratifiedShuffleSplit. Our CV was 0.002 lower (better) than public and 0.001 higher than private.

**Ensembling**
We prepared two separate ensembles. A better one from Chenglong who also employed our features. I and Kostia did not use features from Chenglong and get an ensemble with, as we thought, significantly different behaviour in different parts of the dataset (part1: id<=163700, part2: 163700<id<=221473, part3: id>221473). Since we observed regular patterns in the data as well (please see the attached pdf), we thought that one of the ensembles might be especially prone to overfitting in some parts. So, while blending our ensembles for final submissions, we actually made different bets assuming that in some parts one of the models would behave much worse in private than in public. In fact, we observed no major differences in our models performance in public and private.

**Text processing is the key**

During this competition I and Kostia tried different approaches to text processing and then generated the same features using the different input. The public RMSE was improving at a constant pace. Then, at some point we realized that we have chances to win and have to properly document our work and adhere to the requirements of the winning solutions. So we started recalculating features according to a unified text processing algorithm and observed only minor (if any) advances in the public score.

We now realize that one of the key reasons was that different preprocessing introduced variation in our features which resulted in the better model performance. When we streamline our feature generation, we lost some variance and, therefore, performance. I believed that we could produce a better model, but we failed to catch the lost variation

Actually, if our team had submitted our top leaderboard model (a blend of Chenglong’s model + old model from me and Kostia), we would have won the 2nd (or 1st) place with private RMSE 0.43188. We did not submit it because we neither posted our old replacement dict to the forum, nor had chances to reproduce the old results step-by-step (that ensemble consists of tens of models produced in different times of the competition). And to say the truth, I expected a major leaderboard shakeup because of the patterns in the data (see the attached pdf) and believed the new models would be much more robust to that shakeup.


**Final remarks**

At first, when I saw the 4th place, I was disappointed. Then I realized that not only the place matters. We have done a great work and I learned tremendous amount of knowledge from this competition, so I must be very happy with this experience. I and Kostia started the competition with the goal to enter in the top 10% and learn something new, but actually we competed on the same level with the top Kagglers. I am excited and confident that the future holds many machine learning successes for us. 

It is also an honor for us to have Chenglong as a team member. We are very proud that he joined our team and we enjoyed our collaboration much. I feel a little bit sorry that we failed to contribute to another 1st place for him.

We would like to thank all Kagglers and organizers for the wonderful experience in this competition.

PS. Chenglong will soon share the full solution on behalf of our team on his github.


  [1]: http://blog.kaggle.com/2015/07/22/crowdflower-winners-interview-3rd-place-team-quartet/
  [2]: http://www.nltk.org/howto/wordnet.html
