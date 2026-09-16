# 4th place solution - Llama3 🦙 70B is all you need

Competition: pii-detection-removal-from-educational-data
Rank: #4
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497367

Thank you Learning Agency Lab and Kaggle, for hosting another competition. 

Congratulations to my teammates @kpriyanshu256 for becoming Kaggle Competition Master and @lucamassaron on achieving the rank of Kaggle Competition Grand Master. 🎊🎊🎊

**Context:**  https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/overview

**Data:**  https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data

**Overview of the approach**

## Models

Deberta V3 Large - 2x - 5 fold each - total 10 Deberta V3 large - Inference in less than 7 hrs. (Maxlen = 4000 and Stride = 1024)  
Training Datasets - meta-llama/Meta-Llama-3-70B-Instruct and thank you @nbroad for the [dataset](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221)
All models were trained with Focal Loss and a combination of BiLSTM and GRU head. 
- Epochs=3
- Batch Size = 1 or 2
- LR = 1e-5 or 2e-5
- Max len = 1280
- AdamW / Cosine Scheduler with Warmup

Other attempts: 

- Deberta Small, Deberta V3 Base - These worked well and were part of the other two selected ensembles.  

- Longformer Base, Gemma 2b - didn’t perform as well, so they weren’t part of the ensemble/selected submissions. 

- Masking augmentation from 2% to 5%, but that wasn't part of ensemble/selected submissions.

- Pretraining with this [model](https://huggingface.co/Gladiator/microsoft-deberta-v3-large_ner_conll2003), however it didn’t work for us. 

- All our code was pure PyTorch - no HF Trainer or other libraries. 


## Cross-validation 

- Stratified 5 fold - Stratified based on PII - Yes / No.  We also calculated OOF scores only on the competition dataset. 

Other attempts: 
- We also tried a more granular multi-label stratified kfold, but that didn’t work. 
- Document %4, as suggested in the forums, had quite a bit of variance in folds. 
- Also, trying a larger number of folds resulted in a slight improvement in the performances, at the cost of longer training times and a reduced possibility to ensemble later.

Our choice of a 5-fold-stratified cross-validation, based on the presence of PII in a document, helped us avoid overfitting when training based on folds and have a reliable measure when choosing the final submissions for the most performing and generalizable solutions without relying too much on the public leaderboard only. 

## Stride and Inference Max Length 

Models trained on 1280 sequence length performed well if inferenced with a max length of 4000. Models trained with stride were restricted to 0.967 on public lb. We found the optimal setting was to use a max length of 4000 with a stride of 1024. Training for max length > 1280 also didn’t work.

## Text preprocessing 

- Minor cleanup of the training dataset, there was some really messy text [document 21657], relabeling incorrect labels as pointed out [here](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/492909).
- High Impact preprocessing - There were multiple kinds of strings in the dataset that returned True for .isspace(). Hence, we replaced all such strings with a [SPACE] token, which we added to the model’s tokenizer. In addition, we observed many strings to contain Unicode characters, which caused a mismatch in inference results (results should be the same if we map predictions from Deberta-tokens to Spacy tokens or get predictions to the Spacy-tokens from the Deberta-tokens). Therefore, we converted all such characters into normalized characters using the unidecode library.

```
def replace_space(lst):
    return ["[SPACE]" if x.isspace() else x for x in lst]

train_df["tokens"] = train_df["tokens"].apply(replace_space)

```

## Data Generation 

We primarily followed the strategy described [here](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221) to generate more data. 
- We created personas first using a LLM - mistralai/Mistral-7B-Instruct-v0.2 or Gemini Pro 1.0 API or GPT3.5 turbo. 
- We added a sample for each type of tool i.e. Learning Launch or Visualization tool or Mind Mapping tool or Story Telling tool to the prompt. 
- For the actual PII labeling we used [faker](https://faker.readthedocs.io/en/master/#) library. 
- We generated data (~4600 samples) from meta-llama/Meta-Llama-3-70B-Instruct, mistralai/Mixtral-8x7B-Instruct-v0.1. 
- meta-llama/Meta-Llama-3-70B-Instruct had the best results - single model trained with dataset generated from LLama3 70B Instruct model is better than our best ensemble. 
- For model inferencing we used [vLLM](https://github.com/vllm-project/vllm). 

Some of the non PII samples had names of people/instructors in the dataset, model identified that as PII - so we paraphrased these false positive samples and added them to the training dataset. We used [https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl model) model for paraphrasing. 

[This](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221) was the only dataset that worked for us, we tried many other datasets shared in the forums. 

## Post processing 

- BERT Classifier  – 
For the STUDENT_NAME labels, we observed that many of the False Positives generated by models were associated with teacher names and with fictional characters introduced in the essay for explaining storytelling.
We extracted sentences from the training data for classes FICTIONAL_CHARACTER, TEACHER_NAME, and OTHER  and constructed a BERT Classifier using them.
We attempted to filter out the predicted STUDENT_NAME tokens by the main model using this classifier. Unfortunately, due to the dataset's significant noise and the heavy penalty imposed by the metric to False Negatives, this approach did not perform well.

- Other attempts at post-processing were made to reduce the number of false positives, such as using regex to filter names and ids (for instance, removing tokens that contained the names of non-obfuscated instructors: "Angela Meyer", "Jeanne Liedtka", "Ed Hess", "Tim Olgivie"), or using an XGBoost model as a final validator of a positive token, by using features related to output logits and token characteristics of the positive token as well as of the nearby preceding and following tokens or statistics for the tokens detected in the entire document. Most of these approaches were left apart in the end because they brought no results in OOF or the public leaderboard. 

- We observed a slight boost in public lb with the following rules, which are a part of our best submission
  - Removing STUDENT_NAME tag if the token string contains a number
  - Remove instructor names

## Others:  
Error Analysis with [Spacy Visualizers](https://spacy.io/usage/visualizers#ent), [Neptune.ai](http://Neptune.ai) for logging (Logging with Neptune.ai helped us look at individual PII Types), [Optuna](https://optuna.readthedocs.io/en/stable/index.html) to get weights and thresholds for different models. 

## Inference Code/Models/Datasets :
[Inference](https://www.kaggle.com/code/rashmibanthia/pii-data-detection-deberta-v3-large-4th-rank?scriptVersionId=173827944)
[Kaggle Models](https://www.kaggle.com/models/rashmibanthia/pii-data-detection-deberta-v3-large) 
[Llama3 70b External Dataset](https://www.kaggle.com/datasets/rashmibanthia/pii-llama3-70b-external-dataset/data)

Thank you @aman1391,  @kpriyanshu256, @lucamassaron and @steubk for collaborating , it was pleasure working with you all 🙏
