# [29th Place Notes] how to train a single deberta model of 0.972 on public LB

Competition: pii-detection-removal-from-educational-data
Rank: #29
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497187

I'm really a beginner in ML and feel very humbled to reach silver zone this time. Many thanks to all kind people that share their thoughts and code, which inlude: @nbroad & @valentinweber & @thedrcat & @emiz6413 for tokenization & metric & training piplines, @mpware for high quality external data generation, @nischaydnk for alternative longformer and h2o solutions, and countless discussion posts that gave me inspirations and new ideas.

I'm so grateful for being part of this jounery and the knowledge I learnt. I hope I will be able to contribute more soon.



#  ▶ TRAIN PARAMS
- learn rate: 2e-4~2.5e-5
- accumulated batch size:  8~16
- epochs: 3~5
- downsampling rate for essays with no PII: 0.4
- o class weight: 0.2
- training max length: 3072
- training truncation stride: 512
- optimizer : AdamW
- scheduler : Cosine Annealing



#  ▶ TRAIN TRICKS
1. **Careful manual CV splitting for equal label disctribution in all folds.** 
Instead of using document%4 approach, I manually assigned the data into 4 folds to achieve the almost identical label discribution in each fold. Special attention was paid to the essays where mutliple types of rare labels coexist.
(my data spliting notebook here: https://www.kaggle.com/code/yannan90/pii-df-fold)

2. **stripe away B-/I- prefixes and train on 8 labels only. (this is the most important trick!!)**
My models trained on 13 labels never exceeded 0.96 on public LB no matter how I tuned the training params. While examing the wrong preds, I noticed that the model cannot understand the logic of assigning "B-" or "I-" prefixes, especially for names. 
So I striped away the B-/I- info from the labeling and changed my pipeline to 8-label training. (with B-/I- assignment performed in post-processing) Then the performance of my models improved significantly, with public LB scores ranging from 0.966 to 0.972, and better alignment between CV and LB. This adjustment also helped to identify overfitting, as models with high CV (0.98~0.99) tend to get lower LB (0.95~0.96).



#  ▶ POST-PROCESSING (NOT MUCH)
1. Stick to a threshold of 0.9.
2. Add a whitelist of safe URLs.
3. Address the special case where "/n" might be labeled as ADDRESS if it sits between ADDRESS-labeled tokens.
4. Add prefixes "B-" and "I-" as mentioned above.
 
Useful post-processing tricks learned from others' solutions that I didn't use:
- Removed all NAME_STUDENT predictions that are not title-cased, or of length 1, or contain a digit, or through blacklist (Mr., Ms., Dr., ...).
- If a name is mentioned multiple times in one document and one of them is pii, mark them all as pii. (very impactful)
- If an article precedes the prediction of NAME_STUDENT, it is removed unless "'s" follows, in which case it is not removed.
- Turn PHONE_NUM prediction that is a number with 9 or more digits into ID_NUM.
- Use Regex expression for phone numbers.
- Drop "B-ID_NUM" predictions for ":" and "-".


#  ▶ FINAL SUBMISSION INFER STRATEGY
1. assemble 3 deberta-v3-large models trained on different folds, giving the highest weight to the one with highest public LB score (0.972)
2. due to different tokenizers used for training, one spacy token could be inevitably split into several infer tokens therefore more than one preds will be generated, in that case, I tired 3 strategies to pick the best one: a. the first pred; b. the pred with min O_pred, c. the pred with max positive_pred. In the end I pick strategy c.
 
(my submission notebook saved here: https://www.kaggle.com/code/yannan90/pii-infer-ensemble-ignorebi)

#  ▶ LESSON LEARNED & FURTHER IMPROVEMENT

- **Use llama-3 70b / mixtral-8x-7b to generate external data.** Prompt the model to add tailered content to training data
(https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221)
- **Paraphrase false positive samples and added them to the training dataset** using paraphraser (https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl).
- **Clean & relabel incorrect labels in train data.**
- **Add a bilstm layer before the classifier**: helped in diversity and smoother in training curve
- **Knowledge Distillation** (usually used for transfering knowledge from big models to smaller models): use models trained on disparate datasets as teachers for a student with the same validation folds. 
- **Augmentation**: swap/change names in certain essays
- **Adding \n\n, \n as new tokens to the tokenizer.** `tokenizer.add_tokens(["\n", "\n\n"], special_tokens=True)`
- **Apply lower weight to external data in loss function**: 1.0 for training data samples and 0.5 for additional dataset samples.
- **Argmax after scaling down class O probabilities** for hard prediction, instead of using O prob thresholding.
- **More post-processing** could be employed through observing patterns in the wrong oof preds
- **Diversify models for ensemble:**
a. different model types, especially using different tokenizers
b. models trained/inferring  with prefix + without prefix
c. models trained/inferring on different max_length
PS: I only submitted the ensembles of deberta-v3-large, since they performed best on both CV and LB. Now I know my another ensemble of debert-v3-large and longformer hit a private LB of 0.9662 (gold zone), which I didn't submit due to not trusting longformer for its poor performence on CV and LB. what a pity LOL! 

- **Tricks to accelerate ensemble inference**: 
a. Use FP16 for inference
b. Use onnx (https://www.kaggle.com/code/lavrikovav/0-968-to-onnx-30-200-speedup-pii-inference)
c. Dynamic Padding (https://huggingface.co/learn/nlp-course/chapter3/2#dynamic-padding)
d. Applying different batch sizes for each token length
 
-----------------------------------------------------

> ***Solution References***
- https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374
- https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352
- https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497367
- https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497306
- https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497177
- https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497968
