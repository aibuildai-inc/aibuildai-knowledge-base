# 13th(14th) place solution

Competition: kaggle-llm-science-exam
Rank: #13
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446231

First of all, thanks to competition organizers for hosting this competition.  
I also like to appreciate @radek1, @leonidkulyk, @mozattt, @nlztrk, @mgoksu for sharing useful dataset, @jjinho for sharing retrieval approach, @cdeotte for summary these dataset and approach in [this discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436383). Without your work, we could not have achieved these results. I learned a lot from you!

I start from @cderotte's [this notebook](https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1). I made four changes from this notebook: <b>dataset, retrieval, models, ensemble</b>


# 1. dataset
Because some of the data (mainly numbers?) were missing as discussed [here](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/434913), I changed wiki dump from https://www.kaggle.com/datasets/jjinho/wikipedia-20230701 to cirrussearch wiki dump.

# 2. retrieval
I tried to improve search accuracy.

## 2-1. dataset for evaluation
I create 250 question for evaluate retrieval by ChatGPT. To accumulate a lot of experiments, only data starting with "a" were used in the evaluation, not the entire wikipedia data. Finally, I achieve ``0.94@recall1``, ``1.0@recall30`` on this dataset. (question from wiki whole dataset: about ``0.85@recall1``)

## 2-2. text processing
For more accurate retrieval, I have split the wikipedia text. Specifically, the text is read from the beginning, and when it reaches 90 words or more, the data is saved, leaving the last three sentences to continue reading the text. 

The pseudo code is as follows:

```python

def count_words(text_processed):
    return len(text_processed.split(" "))

def leave_last_sentence(text, n=3):
    ret = text.split(".")[-n:]
    return ".".join(ret)

texts = load_wiki()
texts_processed = []
text_processed = ""
length = 90
window = 3
for text in texts:
    text_split_period = text.split(".")
    for sentence in text_split_period:
        text_processed += sentence
        if count_words(text_processed) > length:
            texts_processed.append(text_processed)
            text_processed = leave_last_sentence(text_processed, window)
```

I tried ``(length, window) = (60, 2), (75, 2), (90, 2), (90, 3), (90, 4), (120, 4), (150, 6)`` and choose the best LB score.

## 2-3. faiss
I use this parameters: ``{"nlists": 1, "M": 64, "nbits": 8}``

## 2-4. retrieval model
I use ``gte-base`` and ``e5-base``. I tried {``gte``, ``bge``, ``e5``}_{``small``, ``base``, ``large``} and choose the best. (In the beginning, I select the model by looking at the search accuracy in only-a. However, I realized that this retrieval accuracy was not correlated with LB, so I checked it by looking at LB at the end😭.)

# 3. model
Almost the same as baseline notebook.  
I set ``max_length`` 256 for training, 786 for inference. I wanted to train with ``max_length`` 786, but I did not have enough memory resources. 256 was the best when I trained models with 256, 384, and 512 ``max_length``, and then predicted them with 768 ``max_length``.

training parameter is below:
```python
training_args = TrainingArguments(
    warmup_ratio=0.01, 
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=config.epochs,
    report_to='wandb',
    optim="adamw_hf",
    overwrite_output_dir=True,
    fp16=True,
    gradient_accumulation_steps=8,
    load_best_model_at_end=True,
    metric_for_best_model="eval_map@3",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    save_total_limit=1,
)
```

I ensemble 4 models, combination of the following elements
- models
    - ``OpenAssistant/reward-model-deberta-v3-large-v2``
    - ``deepset/deberta-v3-large-squad2``
    - ``microsoft/deberta-v3-large``
- retrieval models:
    - ``gte-base``
    - ``e5-base``
- dataset:
    - all
    - without 3, 7, 8, 9 : worse score compared from other datasets.
    - without 10 : better score compared from other datasets.
    
    number is [this dataset's](https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2) source.

# 4. ensemble
## 4-1. TTA
I use the low ranking retrieval results for inference.
I have 4 tta below:
- ``[ 0,  1,  2,  3,  4,  5]``
- ``[ 0,  6,  7,  8,  9,  10]``
- ``[ 0,  11,  12,  13,  14,  15]``
- ``[ 0,  16,  17,  18,  19,  20]``

the number of array is retrieval ranking of question.

## 4-2. ensemble
After TTA, Instead of the simply average the score, I took the sum of the maximum and the average like below:

```python
df = pd.read_csv("test.csv")  # len(df) == n_test_data
df["id"] = np.arange(len(df))
df = ensemble(df)  # len(df) = n_test_data * n_tta * n_models
df = df.groupby("id").mean() + df.groupby("id").max()
```

# 5. Not worked / Can't get result for me.
- Create question by ChatGPT.
  - I try many prompt, but score is decreased.
- Create the CV correlated with LB.
  - I try 300 row per @cderotte's dataset, but finally use @wuwenmin's 300 + test 200. But CV is not correlated with LB.
- More complex TTA.
- Improve model accuracy.
  - It was almost same accuracy as the @cderotte's notebook.
  - I thought that improving the model was not the key point of this competition, so I focused on the retrieval.
