# 7th place solution

Competition: gendered-pronoun-resolution
Rank: #7
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90334#latest-530380

## Scores

Stage 1 LB: **0.2929** (trained on 2000 Test + 454 Validation, evaluated on 2000 Development)

Stage 1 5 Fold Out-of-fold: **0.2922** (train/evaluated on 2000 Test + 2000 Development + 400 Validation)

Stage 2:  **0.19473** (trained on the same 4400 examples as above, I left the other 54 validation data for sanity check)

## Model
I have a main model (referred to as “end-2-end” model below) and a secondary model (referred to as “pure bert” model below).

Both models use Bert features based on Matei’s [strong baseline](https://www.kaggle.com/mateiionita/taming-the-bert-a-baseline) (Thanks Matei!) Pure bert model has the same architecture as in Matei’s kernel; end-2-end model uses the top level architecture in https://arxiv.org/abs/1707.07045, as implemented by Chanhu’s [kernel](https://www.kaggle.com/chanhu/bert-score-layer-lb-0-475) I also added 10 linguistic features from pheel’s [kernel](https://www.kaggle.com/pheell/look-ma-no-embeddings) to the end-2-end model

For pure bert model, I concatenated layer -3 and -4; for end-2-end model, I only used layer -4 embeddings. I used Bert Large (both cased and uncased) with max token length 256 for both models.

I didn’t do any fine-tuning. 

## Augmentation
Bert embeddings of the names (A and B) contain information about both the context and the name itself. For coreference resolution task, it doesn’t matter whether a name is Alice or Betty or Carol or Debby. Therefore, I augmented the input data by replacing the A and B names by 4 sets of placeholder names:
```
[ { 'female':['Alice','Kate'], 'male': ['John','Michael']},
  { 'female':['Elizabeth','Mary'], 'male': ['James','Henry']},
  { 'female':['Kate','Elizabeth'], 'male': ['Michael','James']},
  { 'female':['Mary','Alice'], 'male': ['Henry','John']}]
```

I chose these names by looking at the most common names in training data while making sure not to choose similar sounding names. If you pay attention to popular name trends in the U.S., you will notice all these placeholder names are old school names that are not as popular anymore (see e.g. https://www.behindthename.com/name/mary/top/united-states and
https://www.behindthename.com/name/john/top/united-states). This is because in the training data, there are a large amount of articles about historical people.

I initially chose these 4 sets of names. Later I tried fewer, more, and different (more modern) names but didn’t find improvements.

For each input data row, I augmented it into 4 variations by replacing A by Alice, B by Kate if female, A by John B by Michael if male, etc. The bert embeddings are extracted 4 more times for these 4 augmented data. Then each training epoch will see each input data 5 times (original and 4 augmentations). This way, the embedding information about the particular names will be averaged out. Only the information about the sentence structure will be left. Or to put it in a different way, let’s say if 200 dimensions out of the 1024 Bert embeddings are about the name itself and the rest 824 dimensions are about the context, then the model will learn to rely on only those 824 dimension to resolve coreference and ignore the other 200 dimensions.

Another potential benefit of replacing all names by these one-word short names is that, if a name is long (first name + last name), it will be tokenized into many word pieces and make it harder for Bert to embed the necessary information into these token. This is only my guess, to be further tested. But I do have an evidence supporting this: in testing time, I did TTA (test time augmentation) the same way as in training. The order of the 4 augmentation’s scores varies depending on the model, but they all outperformed the un-augmented version, always. See the “original” row in the results table below.

When I did the name replacement, I replaced all the occurrences of the name in the document by placeholder names. In order to avoid confusion, I didn’t do the replacement in the following situations:
1.  If the placeholder name (i.e. Alice or Kate) already appear in original document
2. If A or B is full name (first + last name), but the first name or last name appear alone elsewhere in the document. For example, if A is “Michael Jordan” but “Jordan” is used to referred to the person later in the doc, if I replace all the “Michael Jordan” in the doc by “Henry” then Bert would think “Jordan” and “Henry” are different people
3. If the name has more than two words, such as “Elizabeth Frances Zane” or “Jose de Venecia Jr”, I don’t replace it because it would be difficult to implement the rule above

## Incorrect ground truth
As pointed in the Discussion, there are clearly wrong labels. I fixed 74 of them in development, and 85 in test and validation. (I did this before wayward’s [post](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/81331#503094). She reported similar number of corrections, so I assumed it would be similar without comparing hers to mine.)

One important thing is that, the model tends to make very confident predictions when trained with “clean” labeled, i.e. the output probabilities are very close to 0 or 1 for some rows. When scored with “dirty” labels, however, correct prediction of very small probability will lead to very large logloss. Therefore, I clipped all the output probabilities by a tuned threshold, usually 0.005 or 0.006

## Other changes
I did the following changes to the Bert features extraction par of Matei’s kernel. Without them, the extracted token embeddings are misaligned in some cases.
1. There are two documents (209 in dev and 921 in test) too long to be correctly embedded using max sequence length 256. I wrote a function to throw away first few sentences in such scenarios to make all embeddings valid. But in stage 2 data, there is no such case.
2. In Matei’s kernel, it always skip the first 2 tokens. I changed it to skipping two tokens only if the second token is `"`, because sometimes (I think when there are quotations in document) actual tokens start at the index 1 instead of 2
3. I added
 `if text=='#': return 1`
in functions `count_chars_no_special` and `count_length_no_special`


## Validation strategy and ensemble
Final model is `0.9 * end2end + 0.1 * pure bert`

Below is the breakdown of the model’s OOF scores, ensemble components and weights.I tuned all the weights using clean labels, but reported the scores with dirty (i.e. actual) labels. Each column is a separate model. The top 5 rows in any column are the evaluation results on the 5 TTA variations from the same model.

[results]


The model is trained on 4400 data in 5 folds. In each fold there are 3520 train data and 880 OOF validation data. Each 3520 training data is further divided into 5 folds, with each fold having 2816 training set and 704 early-stop validation set. As mentioned, the 2816 data is augmented into 14080 data per epoch. 


##Code
Code is available at: https://github.com/boliu61/gendered-pronoun-resolution

##Update on 4/30/2019
After reading other solutions, I realized layer -5 and -6 are the best. For some reason I thought -1 to -4 were the only available layers to extract features from. I just trained my models again, with everything else unchanged except changing Pure Bert model from layer -3 and -4 to -5 and -6; End2end model from layer -4 to -5.

Stage 1 5fold CV score improved from **0.2922** to **0.2846**
Stage 2 Late submission score improved from **0.19473** to **0.18075** (only one submission, no LB probing)

I think this shows the augmentation approach can be as good as fine-tuning.

##Update on 5/6/2019
paper is available at https://arxiv.org/abs/1905.01780
