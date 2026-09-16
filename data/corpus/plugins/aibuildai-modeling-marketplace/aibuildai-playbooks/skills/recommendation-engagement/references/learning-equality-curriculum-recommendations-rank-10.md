# 10th place Solution

Competition: learning-equality-curriculum-recommendations
Rank: #10
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/395190

I want to thank the organizer for a fun competition for an important mission. I learned a lot through participating in it. Thanks also for everyone who is sharing the solutions, I've been learning a lot from the many approaches being presented.

My solution had 3 stages which followed the most popular paradigm in this competition- retrieval, reranker (cross-encoder) and a stage to calibrate the results on the validation set for threshold selection.

A high level diagram of the pipeline:


My overall strategy was to keep stage1 inference simple and fast and do the heavy lifting in stage2 with the heavier transformer models. 

## Stage 1: Bi-Encoder
The baseline model was sentence-transformers/paraphrase-multilingual-mpnet-base-v2 and iteratively fine tune it with better training sets. 

**~~K~~NN**
The output of stage1 is a set of content candidates assigned to each topic. Candidate selection was based on similarity score threshold (not top-K neighbors). This was helpful in several ways:
1. It helped avoid unnecessary processing for many irrelevant candidates
2. Initial experiments showed it gave better stage 1 CV
3. The score from stage 1 helped define “sampling regions” that I used in stage2 and 3. More on this later.

**Stage 1 features:**
For each topic, I built a reverse topic tree from the training set. I’ll call it [topic path] in the rest of the note:
(example)
/12. 20: Bird Reproduction/12: Vertebrates/Book: Introductory Biology/Introductory and General Biology/Bookshelves/Libretext Open Educational Resource Library


I used the following columns concatenated to strings as the input for the models:
Content: [Kind]< K >[title]< D >[description]< T >[text]
Topic: [topic path]< O >[description]

**Training**
The key for success in stage 1 was finding an optimal training set. The challenge has been how to select the negative samples for training as naive sampling leads to weak models. 

In order to create an optimal training set I created the following process:
1. Predict topic-content assignments on the training set based on the latest trained model
2. Use the topic-content assignments to generate a new training set. This would create a new training set with hard false positives on the top scores and easy false positives. The threshold in which I would cut off candidate would be when reaching an average of ~50 candidates per topic. 
3. Add all the missing topic-content assignments from the training set
4. Train a new model with the new training set.
5. Go to step 1

**Multiple Negative Ranking Loss**
One of the key decision that helped my CV/LB early on was to use Multiple Negative Ranking Loss, which helped accelerate training significantly. From the training process described above, the false positive candidates work great with MNRL because in one batch we can have a good blend of hard to easy samples across multiple topics. There is one problem though- one of the requirements of MNRL is that for each sample pair (a_i, p_i): all p_j (j!=i) and all n_j are considered negative, which is likely to break especially in case of hard negative samples. In order to make it work I created a carefully crafted training set preparation process that “pre arranged” the batches before training started.

For each batch:
1. pick a positive topic-content assignment
2. For each false positive content assignment for the above topic, find a correct topic to pair to that content and add it to the training set. Remember previously encountered topics and skip topic-content pairs where the topic has already been seen.
3. Repeat the above process, in case there are no more valid samples to choose, pick another random topic-content pair.
4. In case of a dead end- pick a sample from another language. And start over. Conveniently topic-content assignment are always valid within the same language so there are no collisions across languages.

I passed the training data as-is to the data loader without shuffling where the batch size is the same one I used in the pre-processing step.

**Contrastive Loss**
After training enough iterations of the above process (eventually F2/Precision/Recall stops improving), I took the best model and further fine tuned it with contrastive loss on the latest training set using a standard training process with the training data as is.

For stage1 the best F2 score in my CV is 0.65. I didn’t submit a solution based on stage1 when getting to these results so I don’t have the corresponding LB scores.
The average number of candidates per topic vs. recall at different threshold was
(threshold : avg. number of candidates : recall)
0.91 : 9.7 : 0.77
0.9 : 16.45 : 0.81 
0.89 : 29.8 : 0.85 
0.88 : 55.7 : 0.89
0.87 : 103.4 : 0.91

## Stage2: Cross-Encoder
I’ve been thinking about this step much like a “zero shot” learning approach because the topics in the test set were not seen in the training set. Furthermore, there is additional content that was not in the training set.

**Feature engineering**
From each topic-conent pair from the previous stage (both for positive and negative labels) I generated two samples:

**Training Sample 1:**
Same as the one for stage one but topic and content are concatenated and separated by the relevant [SEP] token


**Training Sample 2:**
An addition that I had in this stage is to add the correlated topic paths of the content to the content features:

content: [Kind]< K >[title]**< C >[correlated topic path 1]< C >[correlated topic path 2] .. < C >[correlate topic path n]**< D >[description]< T >[text]

example:


The topic text is exactly the same as stage 1:
Topic: [topic path]< O >[description]
example:



The final input text is concatenation of both the topic and content separated by the [SEP] token relevant for the model.

The reason for the two variants is that I wanted to make sure that the model have two learned capabilities:
Predict topic-content assignment based on text features alone - this would help with new unseen content in the test set.
Predict topic-content assignments based on relationship of the content to other topics. This helped resolve assignment ambiguities where the attributes of the content alone were not informative enough to decide whether a certain content is relevant to the topic or not (e.g. similar math concept but for different grades)

**Sampling**
The cross-encoder was hard to train. The key for successful training was the sampling strategy. Similar to stage 1 it’s also important to find a good balance between hard and easy negatives so the model generalizes well with high predictive power.

The best strategy that worked for me was - I took samples from stage one where the threshold was above 0.89. Between 0.88 and 0.89 I sampled 0.1 of the negative samples and below 0.88 I sampled 0.02 of the negative samples. I always added all the positive samples

**Models:**
I trained 4 models:
XLM-Roberta-Base, seq max length 256
DeBERTa-v3-xsmall, seq max length 512
DeBERTa-v3-small, seq max length 512
DeBERTa-v3-base, seq max length 400

It’s interesting to point out that although the Deberta models are English only models their tokenizers had the non-english character sequences and they were able to train well on the competition dataset. I didn’t use mdeberta because I was unsuccessful in making it work with fp16 training and training was too slow.

One interesting observation is that more epochs consistently resulted in higher LB scores both in public and private LB. I suspect that I had room to further improve scores by spending more hours in training more epochs, but eventually I ran out of time. And it was also becoming expensive given that I was using Colab Pro+ credit for training.

## Stage3: Logistic Regression
The last model is taking the scores from stage1, the models from stage2 and trained on the validation set to produce the final score for topic-content assignments. Due to training on the validation set and using the results to find the threshold for maximizing F2 I decided to use a linear model in order to have low model complexity and less likelihood of overfitting. 

The approach I took here for creating separate features for each stage 1 threshold and further split the features based on whether the content was in the training set or not.
It looks something like that.

Here is a snapshot of the coefficients from the best model to illustrate the above

Threshold above 0.95:
Content In Training Set:
stg_1_score, xlmr_score, deb_xs_score, deb_s_score, deb_base_score
0.16605078,  0.67237598,  3.0603439 ,  0.92530175,  2.55978453, 

Content Not In Training Set:
stg_1_score, xlmr_score, deb_xs_score, deb_s_score, deb_base_score
3.01727545,  1.06343818,  0.9088478 ,  0.33693999,  2.01195161,

Threshold between 0.94 and 0.95:
Content In Training Set:
stg_1_score, xlmr_score, deb_xs_score, deb_s_score, deb_base_score
0.25057614,  1.64266565,  1.98558391,  0.50577352,  1.99118334,

Content Not In Training Set:
stg_1_score, xlmr_score, deb_xs_score, deb_s_score, deb_base_score
2.60065444,  0.93282139,  0.68057603,  0.82555645,  1.23466772,



For the submission I selected a different threshold that maximizes F2 on the validation set for content in the training set and out of the training set.

My best submission had CV 0.747,  Private LB 0.741, Public LB: 0.708

**Validation Set:**
For my evaluation set I sampled 10% of the topics in correlations.csv randomly, and removed half of their correlated content from the training set as well. In hindsight removing some channels in addition would have been better to mimic the distribution of the data in the competitions test set more closely. Because the validation set was used to calibrate the final predictions and the F2 maximizing thresholds I believe that a better validation set would directly translate to better LB scores.

In this competition I used 1 fold CV shared across all stages. I was initially concerned about the high training cost for more than 1 fold, but over time I got more comfortable with the 1 fold CV due to a consistently good correlation between the CV and the LB results. I did have quite a bit of shake-up anxiety toward the end though so I’m happy it wasn’t a bad shakeup case.


For training I used Sentence-Transformers for stage1, Hugging Face Transformers for stage2 and scikit-learn for stage3.


Thanks and looking forward to more competitions!
