# 41st Place Solution for the Kaggle - LLM Science Exam

Competition: kaggle-llm-science-exam
Rank: #40
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446369

### 41st Place Solution for the Kaggle - LLM Science Exam
First of all thank you to the competition office for organizing such a great competition! Also, thanks to [@cdeotte](https://www.kaggle.com/cdeotte), [@MB](https://www.kaggle.com/mbanaei) and all the competition participants who shared useful ideas and codes. It was a very exciting and interesting competition! My result was 58th for Public LB, 41st for private LB and I got a silver medal.

### Context
- Business context: https://www.kaggle.com/competitions/kaggle-llm-science-exam/overview
- Data context: https://www.kaggle.com/competitions/kaggle-llm-science-exam/data


### Overview of the approach
Our final model was a combination of seven single microsoft/deberta-v3-large models (each with a different seed, data set, and pause token).The final model was a combination of an average ensemble of seven microsoft/deberta-v3-large models and TF-IDF-based retrieval.The final result was Public LB=0.916 / Private LB=0.909.


### Details of the submission
The datasets used for training were the [@cdeotte](https://www.kaggle.com/cdeotte)'s 60k dataset and the 99k dataset. Also, a training model that takes into account pause tokens was created by referring to "Think before you speak: [Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226) to build a learning model that takes pause tokens into account.Then, TF-IDF-based context retrieving method was performed, referring to [@MB](https://www.kaggle.com/mbanaei)'s method during inference.

Below is a summary of CV, Public LB, and Private LB for each model(Use of TF-IDF based contextual search method during inference).The validation data for CV used 200 data from the [@cdeotte](https://www.kaggle.com/cdeotte)'s 60k dataset in all cases.

| model | CV | Public LB | Private LB |
| :---:| :---: |:---:|:---:|
| microsoft/deberta-v3-large (seed=42, dataset=60k, without pause token)| 0.915 | 0.907 | 0.905|
| microsoft/deberta-v3-large (seed=52, dataset=60k, without pause tokens) | 0.893 | - | - |
| microsoft/deberta-v3-large (seed=62, dataset=60k, without pause tokens) | 0.910 | - | - |
| microsoft/deberta-v3-large (seed=72, dataset=60k, without pause tokens) | 0.895 | - | - |
| microsoft/deberta-v3-large (seed=82, dataset=60k, without pause tokens) | 0.905 | - | - |
| microsoft/deberta-v3-large (seed=42, dataset=60k, with pause tokens) | 0.906 | 0.902 | 0.902 |
| microsoft/deberta-v3-large (seed=42, dataset=60k+99k, without pause tokens) | 0.904 | - | - |

### Freeze layers
Some layers were frozen because microsoft/deberta-v3-large did not learn well as it was. (if the layers were not frozen, the MAP@3 was not good with CV~0.37 at the first epoch.)

### Max length
The max_length for training was 256, 512, and 1024, and 512 was the best CV, so 512 was used.

### TF-IDF-based context retrieving method
I tried updating [@MB](https://www.kaggle.com/mbanaei)'s method, combining two different contexts and setting a threshold for frequency of occurrence, etc., but the most effective was applying logarithmic scaling of TF-IDF frequencies and updating the stopwords. I combined the original stopwords with ENGLISH_STOP_WORDS from sklearn.feature_extraction.text.

### Pause tokens
The following was done to take pause tokens into account when learning.(According to the [paper](https://arxiv.org/abs/2310.02226), the accuracy of Language Models improves when pause tokens are added during training, just as humans do better when they think carefully, so hopefully the CV will improve a little more... I tried the following, but if the method is more creative, the CV may increase.)

- Add pause tokens to tokenizer
```
tokenizer.add_tokens(["<pause>"])
model.resize_token_embeddings(len(tokenizer))
```
- Add pause tokens in random position on input
```
PAUSE_TOKEN_COUNT = 2


def insert_pause_tokens(sentence, count):
    tokens = tokenizer.tokenize(sentence)
    for _ in range(count):
        position = random.randint(1, len(tokens) - 1)
        tokens.insert(position, "<pause>")
    return tokenizer.convert_tokens_to_string(tokens)


def preprocess(example):
    context_with_pause = insert_pause_tokens(example["context"], PAUSE_TOKEN_COUNT)
    first_sentence = ["[CLS] " + context_with_pause] * 5
    second_sentences = [
        " #### " + example["prompt"] + " [SEP] " + example[option] + " [SEP]"
        for option in "ABCDE"
    ]
    tokenized_example = tokenizer(
        first_sentence,
        second_sentences,
        truncation="only_first",
        max_length=MAX_INPUT,
        add_special_tokens=False,
    )
    tokenized_example["label"] = option_to_index[example["answer"]]
    return tokenized_example
```
- Custom Loss Settings
```
def custom_loss(outputs, labels, attention_mask):
    loss = F.cross_entropy(outputs, labels, reduction="none")
    # pause_id = tokenizer.convert_tokens_to_string(['<pause>'])
    # pause_mask = (labels == pause_id).float()
    pause_id = tokenizer.convert_tokens_to_ids(["<pause>"])[0]
    pause_mask = (labels == pause_id).type(torch.float)
    masked_loss = loss * (1 - pause_mask)
    return masked_loss.mean()
```
- Setup and run custom trainers
```
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = custom_loss(logits, labels, inputs["attention_mask"])
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_valid,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

trainer.train()
```

### Sources
- https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2
- https://www.kaggle.com/datasets/cdeotte/99k-data-with-context-v2
- https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1
- https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles
