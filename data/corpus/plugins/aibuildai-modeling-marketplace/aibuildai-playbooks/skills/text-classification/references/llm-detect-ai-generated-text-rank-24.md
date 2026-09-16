# 24th solution: DeBERTa & TF-IDF Vectorizer Ensemble

Competition: llm-detect-ai-generated-text
Rank: #24
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470077

Thanks to the host for opening this competition. We learned valuable lessons that will undoubtedly contribute to our growth.  

Here is the brief solution of our's.
We used an ensemble of DeBERTa models & TF-IDF Vectorizers.
Details are below.

# 1. DeBERTa

### CFG
Model: DeBERTa-v3-large
epoch: 5~6
train max_length: 256~300
inference max_length: 512

### Data
- Human Data: 300k++
- AI Data: 300k++ (included self-made data via ChatGPT3.5, 4)
- Used diverse Data, not only 'RDizzl3_seven'.
- AI generated text with more than 10 typos all excluded. typo checked by Pyspellchecker.
- CV split: Stratified KFOLD - 0.9:0.1 


# 2. TF-IDF Vectorizer

Special Thanks to @datafan07 for providing us the amazing public notebook! It was awesome.

### Model
MultinomialNB, SGDClassifier, LGBMClassifier

### Data
- DAIGT-V3 by @thedrcat and we added our self-made data via ChatGPT3.5, 4
- [llama, ada, babbage, claude] -> we excluded these data since a lot of them had too much typo.
- Used diverse Data, not only 'RDizzl3_seven'.
- AI generated text with more than 10 typos all excluded. typo checked by Pyspellchecker.


# Ensemble

- TF-IDF Vectorizer: MultinomialNB x 1, SGDClassifier x 3, LGBMClassifier x 3
- Deberta x 4
- TF-IDF Vectorizer : DeBERTa = 0.65:0.35

Thanks again to the competition hosts and everyone who participated!! Good luck everyone for your next competitions!

You can check our submission notebook here: 
https://www.kaggle.com/code/kimseunghee/24th-place-notebook-public-0-966-private-0-927

### Team member
@danielchae 
@kimseunghee
