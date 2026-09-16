# 13th place solution - Transformers only

Competition: llm-detect-ai-generated-text
Rank: #13
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470593

Thank you *The Learning Agency Lab and Kaggle* for hosting another competition. 

**My solution** 
Score - private LB 0.938  and public 0.957 
My best private submission was also the selected submission (this would have been about rank 2000 on public LB)

Selected submissions (sorted by private score) - 

[[Screen-Shot-2024-01-24-at-11-54-35-AM.png]](https://postimg.cc/0rQ424Nr)

**Dataset** 
I used some of the publicly shared datasets. Amongst all the datasets I tried @thedrcat's v4 was the magic dataset for me. Thank you Darek. I also generated some data on my own. 

Here I share some of the Prompts that I used -
```
System Message 
f"""You are a average non native english speaker grade {str(np.random.randint(6, 13))} student who writes argumentative essays \
for given context. Essay must be between 450 and 650 words. Assume that essays are written without any help of word processing software. \
Please add a minimal amount of typos and mistakes as your grade equivalent student would do in a time constraint environment. """

Prompt 
'Write an argumentative essay about - Exploring Venus -  In "The Challenge of Exploring Venus," the author suggests studying Venus is a worthy pursuit despite the dangers it presents. Using details from the article, write an essay evaluating how well the author supports this idea. Be sure to include: a claim that evaluates how well the author supports the idea that studying Venus is a worthy pursuit despite the dangers; an explanation of the evidence from the article that supports your claim; an introduction, a body, and a conclusion to your essay. '
```

Persuade Corpus [Paper](https://zenodo.org/records/8221504) has race/ethnicity / gender information - I tried to incorporate some of these in the prompts. Also ensuring that there's a good balance of all 7 prompts.

I also removed duplicates that were in persuade corpus and provided training set. I used `BAAI/bge-base-en-v1.5` embeddings and cosine similarity to find matches. 

Overall I had a dataset of about 45K samples. 

**Cross validation**
Practically none, unless the dataset is fixed we can’t evaluate and compare. I was experimenting with different datasets frequently. Towards the end I did fix the dataset and started observing CV at 4th decimal place - which didn’t really work. 


**Modeling**
Ensemble of 3 transformer models - 

All these were trained with the same hyper parameters - max len=512 (anything higher didn’t work on LB) , 5 folds. 
- DebertaXLarge - Private 0.92  -  Public 0.939  -  Ensemble Weight  - 50%
- Deberta v3 Large - Private 0.847 - Public 0.914 -  Ensemble Weight  - 25 %
- Roberta Large - Private 0.849 - Public 0.923 -   Ensemble Weight  - 25 %

Neptune.ai for logging - first time user, pleasant experience, will try next time

**Other Submissions**

My other selected submissions (best public LB - 0.97, private - 0.927 - public rank 42) had TF-IDF based approaches more or less similar to public kernels. I ensembled these with my own deberta models and score goes up to 0.97 on public LB.   I didn’t try to tune hyperparameters for the TF IDF models,  they were from public kernels and it seemed like they were tuned collaboratively anyways on public LB. I did change the datasets a little for these though. Some of the datasets that seemed to work with transformer models didn’t really work well with boosting/MNB/SGD models. So I ended up using different datasets for different models.

**Things didn’t work**
- Generating data with mistralai/Mistral-7B-Instruct-v0.2 after fine tuning with Persuade Corpus (similar to Darek’s dataset) but I wasn’t able to make this work.  Should have used base [Mistral model](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395#2617818)

- Some other transformer models - Electra, other deberta variants. 
- MLM pretraining with deberta v3 large.
- Log probabilities with LGBM - these seemed to have done well in private, I didn’t get to spend much time on it though. 
- Fine tuning 7B model - poor private LB 0.69 and public LB ~0.85 (100% AUC)
- Tensorflow models from public kernels
- Pretty much no hyperparameter tuning/different heads for transformer models also because no validation set. 
- I couldn’t try a lot of the other things that I normally would in any other competition because there’s no validation set. I am hoping this “no dataset” and “no validation set” doesn’t become a norm on Kaggle. 

**Summary** - Just transformer models performed better compared to TFIDF approaches or even ensemble of both approaches for me. Also final submission selection is very important !

Thank you for reading ! 

---

**Links to Models/Code**

[Ensemble Inference](https://www.kaggle.com/code/rashmibanthia/llm-detect-13th-gold-solution/notebook)

[Deberta xLarge Model](https://www.kaggle.com/models/rashmibanthia/llm-detect-deberta-xlarge) and [Deberta xlarge Inference code](https://www.kaggle.com/code/rashmibanthia/llm-detect-debertaxlarge)

[Deberta v3 Large Model](https://www.kaggle.com/models/rashmibanthia/llm-detect-deberta-v3-large) and [Deberta v3 Large Inference code](https://www.kaggle.com/code/rashmibanthia/llm-detect-deberta-v3-large-inference)

[Roberta Large Model](https://www.kaggle.com/models/rashmibanthia/llm-detect-roberta-large) and [Roberta Large Inference code](https://www.kaggle.com/code/rashmibanthia/llm-detect-robertalarge)
