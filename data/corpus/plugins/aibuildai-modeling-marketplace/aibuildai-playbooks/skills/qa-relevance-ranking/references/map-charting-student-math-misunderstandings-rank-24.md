# Public 8th / Private 24th Solution (Miss a gold solution)

Competition: map-charting-student-math-misunderstandings
Rank: #24
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/public-8th-private-24th-missing-a-gold-solution

Thanks to hosts and everyone in this competition. And thanks to my every great teammate @l1ghtsource @lechengyan @chronoscop @danilamalinka ! It's a pity that we miss a gold submission :( . But this's my first competition about LLM and I really learn a lot in this competition.

Our final submission consists of one of my casual lm model (lb 0.948) and four sequence classification (one lb 0.949 and three lb 0.948) from @l1ghtsource with some postprocessing. I will explain my parts first, and my teammate will add more details about his parts.

# I2nfinit3y's part

## Data
All training data with 65 classes.

## Model and prompt
My best casual lm model is qwen3-reranker-8b. I give context (question, answer, student's explanation, is correct, common misconception and error rate) and 65 options (every option corresponds to a classes) in the prompts with markdown format, and then I let model output the most likely option. 

```
system_content = (
            "You are an expert AI assistant specializing in educational assessment. "
            "Your task is to analyze a student's reasoning and select the single most accurate "
            "classification from a list of options. The options may include specific mathematical "
            "misconceptions or broader categories indicating a correct or irrelevant explanation."
        )

def choice_collate_fn(batch):
        prompts = []
        labels = []
        for example in batch:

            user_content = "Based on the student's reasoning provided in the context below, select the single best descriptive option.\n\n"
            
            user_content += "### Context and Student Data\n"
            user_content += f"- **Question**: {example['QuestionText']}\n"
            user_content += f"- **Student's Answer**: {example['MC_Answer']}\n"
            user_content += f"- **Was Student's Answer Correct?**: {'Yes' if example.get('is_correct') == 1 else 'No'}\n"
            

            if 'question_difficulty' in example:
                user_content += f"- **Question Difficulty**: {example['question_difficulty']}\n"
            if 'common_misconception' in example:
                user_content += f"- **Common Misconceptions for this Question**: {example['common_misconception']}\n"
            
            user_content += f"\n### Student's Explanation to Analyze\n"
            user_content += f"```{example['StudentExplanation']}```\n\n"
            
            user_content += "### Options\n"
            for i, miscon in enumerate(target_classes):
                user_content += f"{choice_tokens[i]}. {miscon}\n"
            
            user_content += "\n### Your selection:\nThe most accurate option is"


            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            prompts.append(prompt)
            labels.append(example['label'])

        tokenized = tokenizer(prompts, padding="longest", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        tokenized['labels'] = torch.tensor(labels, dtype=torch.long)
        return tokenized
```

In the training, I only get token ids of these 65 options from last token, and then calculate cross entropy loss.

```
choice_tokens = list(string.ascii_lowercase + string.ascii_uppercase + string.punctuation)[:n_classes]
    choice_token_ids = []
    
    for token in choice_tokens:
        encoded = tokenizer.encode(f"{token}", add_special_tokens=False)
        choice_token_ids.append(encoded[-1])

......

outputs = model(**model_inputs)
logits = outputs.logits
last_token_logits = logits[:, -1, :]
choice_logits = last_token_logits[:, choice_token_ids]
ce_loss = loss_fct(choice_logits, batch['labels'])
```

## Inference
I use vllm and logits processor in the inference, so that I can only get logprob of 65 options. The model's best LB is 0.948.

## Training params
lora_rank = 16, lora_alpha=32, dropout=0.1, lr=5e-4, batch_size=128, lr_scheduler='linear'

## What didn't work.
**Full finetune model** : It was very easy to overfit.
**Other 7b~9b model** : It seems that qwen3-reranker-8b is the best model in my experiment.
**Larger model** : I have tried other 14b model but all of them will hurt the result. So weird.
**Distillation** : I tried to use self-distillation and use better model as a teacher model. But there's no boost.
**Rank loss** : Pairwise loss and Listwise loss.
**synthesis data** : Adding any synthesis data will hurt my model's performance.
**Chain of thought**

# lightsource part

My code: https://github.com/l1ghtsource/map-misunderstandings-2025

Interestingly, I had a 0.948 private LB submission made over a month ago. It's just a few weak 0.945 models and models from public notebooks. Apparently, further strengthening my models only made them worse on private LB, which is strange.

## Preprocessing

First, I performed deduplication:

```python
if USE_DEDUPLICATION:
    train = train.drop_duplicates(subset=['QuestionId', 'MC_Answer', 'StudentExplanation', 'Category', 'Misconception'])
    print(f'after deduplication {train.shape=}')
```

Then I experimented with different targets:

```python
if USE_FRACTION_COMBINE:
    print('using fraction combine')
    train.loc[train['Misconception'] == 'Wrong_fraction', 'Misconception'] = 'Wrong_Fraction'

if USE_CATEGORY_REDUCING:
    print('using category reducing')
    train['Category'] = train['Category'].apply(lambda cat: cat.split('_')[-1])

if TARGET_TYPE == 'default':
    train['target'] = train['Category'] + ':' + train['Misconception']
elif TARGET_TYPE == 'category':
    train['target'] = train['Category']
elif TARGET_TYPE == 'misconception':
    train['target'] = train['Misconception']
else:
    print('target will be not defined! look at TARGET_TYPE')
print(f'using {TARGET_TYPE=}')
```

I tried removing the True/False prefix and combining identical classes into one, but this did not yield good results on the leaderboard. I also tried training a model with two heads: predicting category and misconception separately. This seemed logical, as the model would theoretically be able to predict combinations that were not in the train dataset. However, I found it best to just use the original target (65 classes).

```python
idx = train.apply(lambda row: row['category_for_fe'].split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train['is_correct'] = train['is_correct'].fillna(0)

idx_explanation = train.apply(lambda row: row['category_for_fe'].split('_')[0], axis=1) == 'True'
correct_info_df = train.loc[idx_explanation].copy()
correct_info_df['c'] = correct_info_df.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
correct_info_df = correct_info_df.sort_values('c', ascending=False)
canonical_correct_info = correct_info_df.drop_duplicates(subset=['QuestionId'])
canonical_correct_info = canonical_correct_info[['QuestionId', 'MC_Answer', 'StudentExplanation']].rename(
    columns={'MC_Answer': 'Correct_Answer', 'StudentExplanation': 'Correct_Explanation'}
)
train = train.merge(canonical_correct_info, on='QuestionId', how='left')
train['Correct_Answer'] = train['Correct_Answer'].fillna('N/A')
train['Correct_Explanation'] = train['Correct_Explanation'].fillna('N/A')

possible_answers = train.groupby('QuestionId')['MC_Answer'].agg(set).to_dict()

qa2labels = train.groupby(['QuestionId', 'MC_Answer'])['label'].unique().to_dict()
```

In the end, my prompt looked like this:

```python
converter = LatexNodes2Text()

def delatex(text):
    if DO_DELATEX:
        return converter.latex_to_text(text)
    return text

def format_input(row):
    x = 'Yes' if row['is_correct'] else 'No'
    
    variants_text = ''
    if ADD_VARIANTS_TO_PROMPT:
        answers = possible_answers.get(row['QuestionId'], set())
        sorted_answers = sorted(answers, key=lambda v: (str(v)))
        labels = ['A', 'B', 'C', 'D']
        variants_lines = [f'{label}) {ans}' for label, ans in zip(labels, sorted_answers)]
        variants_text = 'Options:\n' + '\n'.join(variants_lines) + '\n'

    possible_targets = ''
    if ADD_POSSIBLE_TARGETS:
        allowed = list(set(qa2labels.get((row['QuestionId'], row['MC_Answer']), [])))
        allowed = [str(x) for x in allowed]
        possible_targets = 'Possible Targets: ' + ', '.join(allowed)
    
    if ADD_CORRECT_ANS_TO_PROMPT:
        return delatex(
            f'Question: {row["QuestionText"]}\n'
            f'{variants_text}'
            f'Answer: {row["MC_Answer"]}\n'
            f'Correct? {x}\n'
            f'Student Explanation: {row["StudentExplanation"]}\n'
            f'Correct Answer: {row["Correct_Answer"]}\n'
            f'{possible_targets}'
        )
    else:
        return delatex(
            f'Question: {row["QuestionText"]}\n'
            f'{variants_text}'
            f'Answer: {row["MC_Answer"]}\n'
            f'Correct? {x}\n'
            f'Student Explanation: {row["StudentExplanation"]}\n'
            f'{possible_targets}'
        )
```

I used the following things:
- Answer options: for each question, there is always a fixed set of four answer options
- Possible targets: I mapped (question_id, answer) -> [cat:misc list] for each such pair
- Delatex (only helped for the phi4 model)
- Correct answer to the question

## Modeling

Throughout the competition, I used a 90/10 random split. In the last week, I switched to training my best models on the full dataset. As a result, I had one 0.949 model and three 0.948 models.

Here are my final selected models:

[imgs]

All (almost) experiments can be viewed at the link: https://docs.google.com/spreadsheets/d/1yHdaMrEjK2xZncWzvF7Z0I0LkWl35BHMnv0dS3bd7So

I trained all models with 8-bit bnb QLoRA using Unsloth, which sped up experiments x2 and allowed me to quickly test ideas on a single A100.

## Unsuccessful experiments

1) Training an adapter for each QuestionID: since the list of questions is fixed, I trained 15 LoRA adapters and used a separate adapter for each group of questions during inference. This resulted in a total of 0.938 lb.

2) Reranker as 2nd stage: I mined hard examples for training (took the top 7 predictions for each sample and excluded the correct one, considering the correct one a positive example and the remaining 6 negative ones). Then, using `Open-Retrievals`, I trained the Qwen3-8B reranker. More details [here](https://github.com/l1ghtsource/map-misunderstandings-2025/blob/main/base_pipes/map-train-reranker.ipynb).

3) Use as an initial model the EEDI winners models 

4) Target='Q_ID:CAT:MISC'

5) Synthetics generation

6) Augmentations

## Synthetics generation

For synthetic data generation, I used GPT-4.1-mini. The main idea was to create additional examples of student explanations for each combination of (QuestionId, MC_Answer, Category, Misconception).

I created 3 prompt versions that randomly alternate to increase variability. Each prompt includes:
- Task Context: full question text + all answer options
- Target Descriptions: for each cat:misc combination, wrote detailed descriptions via Claude Sonnet 4 - exactly what should be in the student's explanation
- Few-shot Examples: 5-10 real examples from train with the same (question, answer, target) combination
- Style Instructions: two sets of instructions for text diversity

Generation code: [link](https://github.com/l1ghtsource/map-misunderstandings-2025/blob/main/base_pipes/generate-map-data.ipynb)

Unfortunately, I was unable to benefit from synthetics; they always worsened my LB score and CV. 

## Augmentations

For text augmentations, I implemented a class with 8 different augmentation techniques applied in random order:

**Numerical transformations:**
- Fractions to words: "1/2" → "one half", "3/4" → "three quarters" (p=0.7)
- Numbers to words: "24" → "twenty-four" (p=0.7)
- Spacing around operators: "2+3" → "2 + 3" or keep as is (p=0.6)
- Decimal format: "0.5" → ".5" or "0.50" (p=0.6)

**Text variations:**
- Synonyms replacement: "because" → "since/as/due to", "divide" → "divided by", "times" → "multiplied by/x" (p=1.0)
- Contractions: "it is" → "it's", "cannot" → "can't" (p=0.8)
- Phrase markers injection: add "I think", "maybe", "probably" at the beginning (p=0.2)

**Realistic errors:**
- Keyboard typos: replace random character with keyboard neighbor (a→q/w/s/z) (p=0.4)
- Letter duplication: "good" → "goodd" (p=0.2)
- Order shuffling: randomly reorder steps separated by "then", "next", "after that" (p=0.5)

The idea was to make the model more robust to student writing variations and typos. Unfortunately, augmentations didn't improve my scores. They either had no effect or slightly worsened CV.

## Blending and post-processing

The final ensemble looks like this (1 I2nfinit3y model and 4 of mine):

```python
all_probs = [probs_infinity, probs1, probs2, probs3, probs4]
model_weights = [0.28, 0.29, 0.11, 0.25, 0.07]

n_samples, n_classes = probs_infinity.shape
probs = np.zeros_like(probs_infinity)

base_w = 0.6
agr_w = 0.3
conf_w = 0.1

for i in range(n_samples):
    row_probs = [all_probs[m][i] * model_weights[m] for m in range(len(all_probs))]
    base_score = np.sum(row_probs, axis=0)
    top_classes = [np.argmax(all_probs[m][i]) for m in range(len(all_probs))]
    agreement_bonus = np.zeros(n_classes)
    for cls in top_classes:
        agreement_bonus[cls] += 1
    agreement_bonus /= len(all_probs)
    confidence_bonus = np.max(np.stack(row_probs, axis=0), axis=0)
    probs[i] = base_score * base_w + agreement_bonus * agr_w + confidence_bonus * conf_w
```

Before averaging the probabilities, I replaced softmax with entmax(alpha=1.05), which made the prediction more crispy. This worked well on both validation and leaderboard, giving a small boost.

Then I selected the rare classes and multiplied their probabilities by a coefficient:

```python
DO_RARE_MULTIPLY = True
COEF = 3
TOPN = 10

RARE_CLASSES = {
    'True_Misconception:Wrong_term': 8,
    'True_Misconception:WNB': 8,
    'True_Misconception:Mult': 8,
    'True_Misconception:Incomplete': 8,
    'True_Misconception:SwapDividend': 8,
    'False_Misconception:Incorrect_equivalent_fraction_addition': 7,
    'True_Misconception:Duplication': 6,
    'True_Misconception:Wrong_fraction': 6,
    'False_Misconception:Shorter_is_bigger': 6,
    'False_Misconception:Wrong_Operation': 6,
    'True_Misconception:Division': 5,
    'True_Misconception:Inversion': 5,
    'True_Misconception:FlipChange': 4,
    'True_Misconception:Denominator-only_change': 4,
    'True_Misconception:Definition': 3,
    'True_Misconception:Multiplying_by_4': 3,
    'True_Misconception:Subtraction': 2,
    'True_Misconception:Positive': 2,
    'True_Misconception:Incorrect_equivalent_fraction_addition': 2,
    'True_Misconception:Adding_across': 1,
    'True_Misconception:Base_rate': 1,
    'True_Misconception:Longer_is_bigger': 1,
    'True_Misconception:Not_variable': 1,
    'True_Misconception:Whole_numbers_larger': 1
}

rare_idx = {le.transform([cls])[0] for cls in RARE_CLASSES.keys() if cls in le.classes_}

if DO_RARE_MULTIPLY:
    adjusted_probs = probs.copy()
    for i in range(adjusted_probs.shape[0]):
        topN = np.argsort(-adjusted_probs[i])[:TOPN]
        for idx in topN:
            if idx in rare_idx:
                adjusted_probs[i, idx] *= COEF
        adjusted_probs[i] /= adjusted_probs[i].sum()
    probs = adjusted_probs
```

Unfortunately, I didn't spend much time on this, but dynamic coefficients and the RARE_CLASSES extension probably gave some additional gains.

Then comes simpler post-processing:

```python
if DO_QUESTION_ID_POSTPROCESSING:
    topk = np.argsort(-probs, axis=1)
    final_top3 = []
    for i, (qid, ans) in enumerate(zip(test['QuestionId'].values, test['MC_Answer'].values)):
        allowed = set(qa2labels.get((qid, ans), []))
        chosen = []
        for lbl in topk[i]:
            if lbl in allowed:
                chosen.append(lbl)
            if len(chosen) == 3:
                break
        while len(chosen) < 3 and chosen:
            chosen.append(chosen[-1])
        if not chosen:
            chosen = list(topk[i][:3])
        final_top3.append(chosen)
    top3 = np.array(final_top3)
else:
    top3 = np.argsort(-probs, axis=1)[:, :3]

flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

if DO_CATEGORY_TRUE_FALSE_POSTPROC:
    adjusted = []
    for labels, corr in zip(top3_labels, test.is_correct.values):
        new_labels = []
        for lab in labels:
            parts = lab.split('_', 1)
            _, rest = parts
            prefix = 'True' if corr == 1 else 'False'
            new_labels.append(f"{prefix}_{rest}")
        adjusted.append(new_labels)
    top3_labels = np.array(adjusted)

preds = [' '.join(row) for row in top3_labels]

sub = pd.DataFrame({
    'row_id': test.row_id.values,
    'Category:Misconception': preds
})
```

The first part uses mapping (question_id, answer) -> [possible targets] to exclude impossible combinations. The second part simply changes the True/False prefix based on the correctness of the answer.
