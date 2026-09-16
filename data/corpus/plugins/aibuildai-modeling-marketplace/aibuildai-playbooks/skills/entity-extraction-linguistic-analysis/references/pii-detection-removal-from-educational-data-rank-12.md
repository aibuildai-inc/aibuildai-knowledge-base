# 12th place solution

Competition: pii-detection-removal-from-educational-data
Rank: #12
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/498130

Thanks Kaggle and the host of the competition for your great work and efforts! 
Congratulations to everyone participated in it! 
Thanks my dear teamates @hustzx @xiamaozi11 @dylanzhao2012 for great ideas and work!

Inference code: https://www.kaggle.com/code/decalogue/infer-ner-pii?scriptVersionId=168424636

Our solution has a high degree of consistency in CV and LB. It's a bag of deberta-v3-large models  **without any** pre/post processing and thresholds.

# Datasets
First thanks nb: @nbroad https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated (very useful!)


We have counted the pii vocab of the competition to pii_cp.json and filtered pii_200k_parsed.csv. Then we got a big mixed vocab pii_mix.json and generated a lot of data with prompt below.

```
prompt_template = """Assuming you are a student enrolled in a massively open online course, please help me continue writing essay based on the given topic or designing your own topic that details your experience of applying a specific tool or approach to address a complex challenge.
This essay should not only narrate the process but also critically analyze the effectiveness of the chosen tool or approach, reflecting on its strengths and potential limitations.

If you mention your personal information, you can use information such as the following 7 types:
NAME_STUDENT - The full or partial name of a student that is not necessarily the author of the essay. This excludes instructors, authors, and other person names.
EMAIL - A student's email address.
USERNAME - A student's username on any platform.
ID_NUM - A number or sequence of characters that could be used to identify a student, such as a student ID or a social security number.
PHONE_NUM - A phone number associated with a student.
URL_PERSONAL - A URL that might be used to identify a student.
STREET_ADDRESS - A full or partial street address that is associated with the student, such as their home address.

The information:
{info}

The topic:
{topic}

Please continue writing the essay:
"""
```
topics:
```python
with open('data/pii_cp.json', 'r', encoding='utf-8') as f:
    pii_cp = json.load(f)
with open('data/pii_mix_v1.json', 'r', encoding='utf-8') as f:
    pii_mix = json.load(f)
NAME_STUDENT = pii_cp['NAME_STUDENT']

topics = []
with open('data/train.json', 'r', encoding='utf-8') as f:
    jdata = json.load(f)
for d in jdata:
    text = d['full_text']
    ts = text.split('\n\n')[:random.choice([1, 2])]
    t = '\n\n'.join(ts)
    if len(t) > 256:
        continue
    for name in NAME_STUDENT:
        t = t.replace(name, 'NAME_STUDENT')
    topics.append(t)

with open('data/test.json', 'r', encoding='utf-8') as f:
    jdata = json.load(f)
for d in jdata:
    text = d['full_text']
    ts = text.split('\n\n')[:random.choice([1, 2])]
    t = '\n\n'.join(ts)
    if len(t) > 256:
        continue
    for name in NAME_STUDENT:
        t = t.replace(name, 'NAME_STUDENT')
    topics.append(t)

topics += ['Please design your own topic'] * 4000
```
prompt and text:
```python
for i,topic in tqdm(enumerate(topics)):
    kv = {k:random.sample(pii_mix[k], random.randint(0,1)) for k in pii_mix}
    if 'NAME_STUDENT' in topic:
        if kv['NAME_STUDENT'] == []:
            kv['NAME_STUDENT'] = random.sample(pii_mix['NAME_STUDENT'], 1)
        topic = topic.replace('NAME_STUDENT', kv['NAME_STUDENT'][0])
    prompt = prompt_template.replace('{info}', json.dumps(kv)).replace('{topic}', topic)
    res = chat(prompt)
    if res == '':
        continue
    text = f'{topic}\n\n{res}'
    text = clean(text)
```

Our generated data:
mm:   pii_cp_data + Qwen_7B_1k + Mistral_7B_1k (pii_cp.json)
mm6: pii_cp_data + Mistral_7B_8k (pii_mix.json)
mm8: pii_cp_data + Qwen_14B_8k (pii_mix.json)

# Models
We use MultiLayer DebertaV2ForTokenClassification with weighted CrossEntropy which improved model stability and making them insensitive to thresholds.
```python
weight = [10.] * (num_labels - 1) + [1.]
loss_fct = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float())
```
Training args
```python
args = TrainingArguments(
    output_dir=f'{task}/{fold}', 
    fp16=True,
    max_grad_norm=10,
    weight_decay=0.01,
    learning_rate=2e-5,
    adam_epsilon=1e-6,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2, 
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    logging_steps=500,
    metric_for_best_model="fbeta",
    greater_is_better=True
)
```
LB single model
|pii_vocab|data|model|fold|LB|
|:--:|:--:|:--:|:--:|:--:|
|pii_cp.json|Nb: pii_cp_data + nb|deberta-v3-large|1|0.962|
|pii_mix.json|mm: pii_cp_data + Qwen_7B_1k + Mistral_7B_1k|deberta-v3-large|2|0.96|
|pii_mix.json|mm6: pii_cp_data + Mistral_7B_8k|deberta-v3-large|1|0.966|
|pii_mix.json|mm8: pii_cp_data + Qwen_14B_8k|deberta-v3-large|1|0.964|


Finally we get 0.96670 on LB while the highest score notebook we did not submit before can get 0.96816 close to 6th place. It only ensembled 5 single models above. So the inference code we share is the highest scoring version.

Not relying on any pre/post processing and thresholds promoting us to continuously design more robust solutions.
