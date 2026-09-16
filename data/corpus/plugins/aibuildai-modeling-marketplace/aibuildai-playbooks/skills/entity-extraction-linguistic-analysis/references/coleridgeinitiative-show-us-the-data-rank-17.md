# 17th Place Solution - SpaCy 3 (EntityRuler) and NER CRF model

Competition: coleridgeinitiative-show-us-the-data
Rank: #17
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248657

# Approach

We initially approached the problem of missing labels by trying to label a subset of the data ourselves using prodigy and spacy with active learning. However we quickly found it intractable because, the default spacy NER was overfitting to specific dataset names and the actual context in which dataset names appeared seemed to be inconsistent with the way that we would identify a dataset in the wild as a human. 

This led us to think that we should focus on contexts where dataset names are identifable as datasets and not as potentially other types of entities. For instance, the sentence "ADNI was financed by XXX" does not give us any indication that ADNI is a dataset, while the sentence "We used the ADNI dataset to analyse ..." gives us the context required to correctly identify ADNI as a dataset.

As a result we made the decision early on to focus on sentences which included words like "data" or verbs like "analysed" in order to only consider entities that could be identified from context. We used a small SpaCy model to split the sentences and greatly reduced the space of text to comb through.

We also probed the training set for names of datasets that were mentioned that we could identify.

# Custom Spacy Abbreviation Finder component

In order to deal with abbreviations, we built a custom SpaCy abbreviation finder that also matched abbreviations with MiXeD casing and tried to pair them with spans either to the left or right of the abbreviation.

For instance given the sentence "This is a Fake Name Compendium FaNaCO". We identify "FaNaCo" as an abbreviation and look left and right for spans that match a certain number of the Capital letters present in the acronym. We used Jaccard Similarity with a threshold here to find similarity between all Capital letters in the span and the ones in the acronym.

We found this method to be decently robust and able to cover more dataset names than a more naive abbreviation finder. 

Once a span is found to be a likely meaning for the abbreviation we added it to a dictionary of abbreviation that was constructed over the entire document.

Unfortunately, we decided early on to not consider acronyms as dataset names, since ADNI was the only acronym present in the training set it was hard to know if including the acronyms of dataset would generalise to unseen datasets. This was a mistake, and would likely have given us several points on the LB had we tested the alternative.

# Custom Spacy EntityRuler component

Our initial approach (and most robust) was to use the SpaCy EntityRuler with custom rules to find candidate dataset names.

Here we build a set of rules to match sequences of tokens agains. For instance, if we have a rule "Title Title Study", and a sentence "We use the Great Age Study in our analysis". Then we would find the "Great Age Study" span as a candidate dataset.

However this is not generic enough and so we expand the component by adding a look backward and look forward functionality. 

So that if we have the sentence "We use the Title Title Study of Youth .." we will match the Title Title Study and then go backward and foward to see if we can expand the span. Here we would allow "of" (as an accepted foward stopword) and then see another Title (Youth) and also include it in the predicted span. Going backwards, "the" would not be accepted backward stopword and so we would stop. The final candidate dataset would then be "Title Title Study of Youth". 

With enough simple rules and a couple of manual use cases we are able to get a decent pipeline component that can identify candidate datasets.


# NER w/ CRF and masking

In the last leg of the competition and having exhausted the rule based approach, we decided to implemented a BERT NER model with a CRF (Conditional Random Field) layer.

Since we filter the text by sentences, the BERT model takes in a sentence and tries to identify a dataset(s) within it. For each word it predicts a tag (BIO scheme) whether or not the word is the Beginning of a dataset, Inside a dataset, or Outside a dataset. 

The CRF layer allows the model to be consistent (for instance prevent it from predictiong O then I), and helps us in this case because a lot of datasets have long names. In fact, we also don't train the model for dataset names that are shorter than 3 words (hyphenated words are counted as 1).

However, as other teams have also realised, an NER model can quickly overfit to known labels.

**We use a clever trick to help the neural network generalise better to unseen dataset labels.** 

By realising that for datasets names, the actual meaning of words doesn't matter as much as the structure/syntax of the dataset. For instance National Survey of Youth is just likely to be a dataset as National Survey of Soda. 

We therefore hypothesize that the actual word embeddings are making model overfit to certain types of datasets.

To reduce overfitting, we add special mask tokens to our embeddings `[TITLE]`, `[UPPER]`, `[MIXED]`. For each word (not including the first word) in the sentence we replace Title words with `[TITLE]`, UPPER words with `[UPPER]` and MiXeD words with `[MIXED]`. 

For words which are "neutral" in datasets, for instance stopwords and words like "dataset" and "study" then we do not mask them.

For example the sentence: **"We use the Heart Cancer Study of Aging HCA in our analysis."**

Would be transformed to: **"We use the [TITLE] [TITLE] Study of [TITLE] [UPPER]  in our analysis."**

Through k-fold cross validation, we found that this trick allowed our model to be able to succesfully detect datasets with more accuracy consistently than using the non-masked sentence. This was especially true in low data regime (with smaller train) and for labels that were held out during training.

# Classifier

Once we obtain our large set of candidate datasets, we built a simple LGBM classifier to replace a selection regex and decide whether or not to include the dataset or not. this is used for candidates that we are not sure on (cannot be automatically included or excluded based on regex) and which have appeared more than a couple times across all documents in the dataset. We

We use very simple features such as counts of tags, and counts of identified geographical names as input to generalise to unseen names as much as possible.

# Take Aways

Evaluating the accuracy of our approach was really tricky (as others have already stated), to try and estimate as much as possible without overfitting, we did some probing at the beginning and use some string matching in our predictions (for known labels) but found that metrics such as using no training labels (thanks @suicaokhoailang) got worse the more string matching we tried.

Our best submission is 0.451 private LB and does not include the NER model, but unfortunately due to not being able to accurately predict our model's performance we selected suboptimal kernels (same as most people in this competition 🙂 ).

I still think that the NER model could be just as performant if not more, given enough labels and training data.

I'll release the code/kernels when I have time to clean them up a little, and will update this post then.

Finally, thanks to my teammate @eeeedev!
