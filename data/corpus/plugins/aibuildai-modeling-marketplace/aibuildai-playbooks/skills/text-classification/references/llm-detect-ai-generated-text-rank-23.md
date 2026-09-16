# A kaggle newbie's 🥈23rd solution:  Just follow-up excellent public works & ideas

Competition: llm-detect-ai-generated-text
Rank: #23
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470153

Thanks to the host and all kagglers with their sharing of excellent works and ideas! As a kaggle novice, I'm very fortunate to get a silver place in my first competition. Although it's a bit pity that I didn't achieve many groundbreaking original work during my trials, I'm willing to share my final submission and choices.

Here's my [solution](https://www.kaggle.com/code/spraut23333/daigt-public-0-965-private-0-927)

### Details of Solution

**1. Data**

@thedrcat 's great [daigt-v2 dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset) and @carlmcbrideellis 's [mistral-7b dataset](https://www.kaggle.com/datasets/carlmcbrideellis/llm-mistral-7b-instruct-texts) with prompts  [mentioned by](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/467820) @bianshengtao excluded. 

Data augmentation with  @aerdem4 's [unsupervised method](https://www.kaggle.com/code/aerdem4/daigt-superfast-unsupervised-baseline) to generate pseudo test label for a small subset of test data. 

**2. Tokenization & TF-IDF vectorization**

I followed the [general pipeline](https://www.kaggle.com/code/datafan07/train-your-own-tokenizer) of training BPE tokenizer on the test data by @datafan07 and chose [SentencePiece tokenizer](https://www.kaggle.com/code/verracodeguacas/sentencepiece-constructions) for final submission as introduced by @verracodeguacas .


**3. Text correction**

I have seen discussions about the license of different text-correction libraries，but I'm still unclear about the permission for usage of these tools since haven't seen a clear claim from the host. My final solutions include text correction on the test data with autocorrect, which is LGPL-3.0 license. But I also chose a version without any text correction for submission and it still stands on silver zone with private LB 0.923.  

**4. ML-Models for TF-IDF features**
 
MultiNomialNB + SGDClassifier(linearSVC) + LightGBM(GBDT) + LightGBM(DART with GOSS) + CatBoost

Focal loss with different gammas for the two lightgbm tree models.

Some model parameters inspired by @batprem 's [generous sharing](https://www.kaggle.com/code/batprem/llm-daigt-analyse-edge-cases).

**5. Transfromer-based Model**

I added @mustafakeser4 's [DistilRoBerta](https://www.kaggle.com/code/mustafakeser4/inference-detectai-distilroberta-0-927) to my ensemble in the last few days before DDL and get an instant boost from 0.962 to 0.965 on public LB, which gave me much confidence and motivation.

**6. Other tricks**

Fixing the max features of TF-IDF to 5M.

[Post processing](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/468363) method by @hyunsoolee1010 .
