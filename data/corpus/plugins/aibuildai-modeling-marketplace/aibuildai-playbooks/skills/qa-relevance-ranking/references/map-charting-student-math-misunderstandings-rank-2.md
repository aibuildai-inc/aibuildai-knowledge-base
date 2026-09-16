# MAP2025_Private&Public 2nd

Competition: map-charting-student-math-misunderstandings
Rank: #2
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd

Thanks so much for the host and I am very delighted to join the competition. My teammate Baiph was very hard working and we did the same contribution to the final results, both of us have the different training pipeline in that way we can be 2nd in public and private leaderboard.

P.S.  I am looking for a remote job in NLP or LLM area, perefered UTC +8 time zone, please email me if it's possible.

# Overview of our solution
The task of this competation is to predict the misunderstandings according to the student's explaination. This is a very easy classification problem but the metric of this comptation is MAP@3 which is used in Ranking Problem normally, so I think the label is ambiguous sometimes, the student's explainations can be labeled in more than one target.

Based on above analysis, what we need to do is denoise or give the student's explaination some soft labels according to oof in case of label leakage.

As for data, there are over 30 labels had limited data, so I used many commerical LLM to generate external data.

As for model selection, we used LLM model as the backbone we tried qwne3, qwen2.5, mistral, phi-4.

As for training strategy, we all convert the classification problem as the multi-choices generation problem, only in this way, we can use vllm to finish the inference timely.


## HZM Part
I am here to introduce my best private sigle model qwen3 -14b which can be much more easy to understand.
The training pipeline can be divided into 4 steps.

### Step1: Generate the ext data (both I and Baiph used them)
Generate one student output with a rare candidate for each question.
Incorporate contrastive learning with the labels.

```python
For example:
if I want to generate student's explation which is True_Misconception:Additive(use this as hard label in step4 training), 
I will use 
False_Misconception:Additive_same_math_problem and False_Misconception:Additive_different_math_problem
+
True_Misconception:Additive_same_math_problem and True_Misconception:Additive_different_math_problem
put the contrastive examples let the ChatGPT or Claude generate better data
```

I used gpt-4, gpt-5, claude-sonnet, gemini, seed, doubao, and generated 80K data.

### Step2: Train LLM to label ext data
Train 4 LLM models on all train.csv and then label ext data (use this as soft label in step4 training), which were phi_4_reasoning_14b,qwen3_32b,mistral_12b,qwen2_5_72b
```python
The following is training prompt:
        """Analyze the student's answer and explanation.            
        Determine if the student's answer is correct (True) or incorrect (False)
        Evaluate if the explanation shows correct reasoning, contains a misconception, or is neither
        If a misconception is present, identify the specific type
        Select exactly ONE label from the 65 options below \n"""
    "{0}"
    "Your Answer: [Select one letter option from A through BM]\n"
    "Input is :\n"
    "Question: {1}\n"
    "Option: {2}\n"
    "Correct Answer: {3}\n"
    "Student Answer: {4}\n"
    "Student Explanation: {5}\n"
```

### Step3: Train LLM in 5 fold to generate soft label
Train 4 LLM models in 5 fold, avg the oof.csv (use this as soft label in step4 training)

### Step4: Multi-loss desgin for qwen3-14b
I changed the SFTTrainer and combined the generation loss (hard label) and classification loss (soft label)
target_token_id meant that the each option will convert to a token id in vocab.

```python
class SFTChoiceTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, return_choice_logit=False):

        target_token_id = self.model.config.target_token_id
        target_token_id = torch.tensor(target_token_id, device=model.device)

        labels = inputs['labels']
        mask = torch.isin(labels, target_token_id)
        labels[~mask] = -100
        inputs['labels'] = labels
        _, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        logits = outputs.logits
        loss = outputs.loss
        #print(outputs)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_target = []
        for i in range(len(shift_labels)):
            lbl = shift_labels[i].cpu().numpy()
            target_idx = np.where(lbl != -100)[0][-1]
            logits_target.append(shift_logits[i][target_idx][target_token_id])
        #
        # # (batch_size, 26)
        logits_target = torch.stack(logits_target, dim=0)
        # # (batch_size)
        labels_target = inputs['soft_label'].to(outputs.logits.device)
        soft_loss = F.cross_entropy(logits_target, labels_target)

        #weight = self._soft_weight()
        loss = loss +  soft_loss
        return (loss, outputs) if return_outputs else loss
```

After the above steps the CV on fold-0 can achieve** 0.955** and public is **0.950** private is **0.946**


## Baiph Part

### 1、**Misunderstandings expansion**

Convert 65 categories into 37 categories and use DeepSeek to expand and explain brief misunderstandings

```python
mis2reason = {
        "SwapDividend": "Incorrectly swapping the positions of dividend and divisor in division operations.",
        "Tacking": "Arbitrarily adding zeros or decimal points to the end of numbers, believing the value remains unchanged or changes incorrectly.",
        "Additive": "Mistakenly using addition to solve problems that require other operations (multiplication, subtraction, etc.)",
        "Wrong_term": "Incorrectly identifying or handling terms in algebraic expressions.",
        "Wrong_Fraction": "Completely misunderstanding fraction concepts or representation methods.",
        "Incomplete": "Providing incomplete solutions missing crucial steps or explanations.",
        "Unknowable": "Mistakenly believing a problem is unsolvable or lacks information when it is actually solvable.",
        "Not_variable": "Treating variables as specific numerical values, or vice versa.",
        "Firstterm": "Overemphasizing the first term in a sequence while ignoring the importance of other terms.",
        "Irrelevant": "Using information or criteria unrelated to the problem for reasoning.",
        "Inverse_operation": "Incorrectly applying inverse operations or confusing relationships between operations.",
        "Multiplying_by_4": "Specific error: Always multiplying by 4 without considering the specific context.",
        "Base_rate": "Ignoring base probabilities or benchmark values, focusing only on specific cases.",
        "Definition": "Misunderstanding mathematical concept definitions or terminology meanings.",
        "WNB": """Mistakenly believing "the whole is not the sum of its parts" or similar part-whole relationships""",
        "Whole_numbers_larger": "Believing decimals with larger whole number parts are always larger, ignoring decimal parts",
        "Incorrect_equivalent_fraction_addition": "Incorrectly performing fraction addition operations",
        "Inversion": "Mistakenly reversing the order of numbers, fractions, or operations.",
        "Mult": "Mistakenly using multiplication to solve problems that require other operations.",
        "Adding_terms": "Incorrectly adding terms directly in algebraic expressions.",
        "FlipChange": "Incorrectly handling numerator-denominator conversions in fraction operations.",
        "Division": "Mistakenly using division to solve problems that require other operations.",
        "Duplication": "Incorrectly repeating numbers or operations.",
        "Interior": "Incorrectly handling interior angles or internal elements in geometric figures.",
        "Certainty": "Providing definite answers for uncertain problems, or vice versa.",
        "Shorter_is_bigger": "Believing numbers with fewer digits are larger.",
        "Wrong_fraction": "Misunderstanding fraction concepts, including numerator-denominator relationships.",
        "Adding_across": "Incorrectly adding across place values (e.g., adding tens to ones directly).",
        "Wrong_Operation": "Choosing completely wrong mathematical operations.",
        "Denominator-only_change": "Changing only the denominator while ignoring corresponding changes in the numerator.",
        "Scale": "Misunderstanding scale factors or proportional relationships.",
        "Longer_is_bigger": "Believing numbers with more digits are larger.",
        "Positive": "Mistakenly believing all mathematical results should be positive numbers.",
        "Ignores_zeroes": "Ignoring the place value or importance of zeros in numbers.",
        "Subtraction": "Mistakenly using subtraction to solve problems that require other operations.",
        "Correct": "Student Explanation is Correct",
        "Neither": "This explanation is confusing and it doesn't fall into any of the above categories"
    }
```

## 2、Prompt

```python
<|im_start|>user
You are now tasked with analyzing math problems and classifying student responses. Given a math problem, the student's chosen answer, whether it's correct, and the student's explanation, you need to determine the appropriate Misconception classification.
(1) Assesses whether the explanation contains a misconception. (Correct, Misconception, or Neither in Category; e.g., True_Correct)
(2) Identifies the specific misconception present, if any.

Below are the available Misconception classifications you can choose from.
Always provide your response using only the specified format.

A: Incorrectly swapping the positions of dividend and divisor in division operations.
B: Arbitrarily adding zeros or decimal points to the end of numbers, believing the value remains unchanged or changes incorrectly.
C: Mistakenly using addition to solve problems that require other operations (multiplication, subtraction, etc.)
D: Incorrectly identifying or handling terms in algebraic expressions.
E: Completely misunderstanding fraction concepts or representation methods.
F: Providing incomplete solutions missing crucial steps or explanations.
G: Mistakenly believing a problem is unsolvable or lacks information when it is actually solvable.
H: Treating variables as specific numerical values, or vice versa.
I: Overemphasizing the first term in a sequence while ignoring the importance of other terms.
J: Using information or criteria unrelated to the problem for reasoning.
K: Incorrectly applying inverse operations or confusing relationships between operations.
L: Specific error: Always multiplying by 4 without considering the specific context.
M: Ignoring base probabilities or benchmark values, focusing only on specific cases.
N: Misunderstanding mathematical concept definitions or terminology meanings.
O: Mistakenly believing "the whole is not the sum of its parts" or similar part-whole relationships
P: Believing decimals with larger whole number parts are always larger, ignoring decimal parts
Q: Incorrectly performing fraction addition operations
R: Mistakenly reversing the order of numbers, fractions, or operations.
S: Mistakenly using multiplication to solve problems that require other operations.
T: Incorrectly adding terms directly in algebraic expressions.
U: Incorrectly handling numerator-denominator conversions in fraction operations.
V: Mistakenly using division to solve problems that require other operations.
W: Incorrectly repeating numbers or operations.
X: Incorrectly handling interior angles or internal elements in geometric figures.
Y: Providing definite answers for uncertain problems, or vice versa.
Z: Believing numbers with fewer digits are larger.
a: Misunderstanding fraction concepts, including numerator-denominator relationships.
b: Incorrectly adding across place values (e.g., adding tens to ones directly).
c: Choosing completely wrong mathematical operations.
d: Changing only the denominator while ignoring corresponding changes in the numerator.
e: Misunderstanding scale factors or proportional relationships.
f: Believing numbers with more digits are larger.
g: Mistakenly believing all mathematical results should be positive numbers.
h: Ignoring the place value or importance of zeros in numbers.
i: Mistakenly using subtraction to solve problems that require other operations.
j: Student Explanation is Correct
k: This explanation is confusing and it doesn't fall into any of the above categories

Please analyze the given input and provide your classification.

### Question:
What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]

### Choices:
(A) \( \frac{1}{3} \) (B) \( \frac{3}{9} \) (C) \( \frac{3}{6} \) (D) \( \frac{3}{8} \)

### Selected Answer:
A. \( \frac{1}{3} \)

### The selected answer is correct.

### Student Explanation:
I think that 1/3 is the answer, as it's the simplest form of 3/9.<|im_end|>
<|im_start|>assistant
```

## 3、Training strategy

LR=2e-4，BS=4*2 Epoch=2

Additional data combined with distillation can enhance LB and PB, but there is no benefit after fusion

| model| data| LB | PB | use final | infere time|
| --- | --- | --- | --- | --- | --- |
| Qwen25-14B-AWQ | total data| 0.948 | 0.946 | submission 1（best） | 20min |
| Qwen3-14B-AWQ | fold1 | 0.948 | 0.943 | submission 1（best） | 20min |
| Qwen25-32B-AWQ | fold1 | 0.948 | 0.944 | submission 1（best） | 40min |
| Qwen25-32B-AWQ | total data| 0.949 | 0.945 | submission 1（best） | 40min |
| QWQ-32B-AWQ | ext data 80k total data training distal| 0.949 | 0.948 | submission 2 | 40min |
| Qwen25-32B-AWQ | ext data 80k fold1 distal| 0.949 | 0.947 | submission 2 | 40min |
| Qwen3-14B-AWQ | ext  data 80k fold1 distal| 0.950 | 0.945 | submission 2 | 20min |

## 4、**Ensemble**

- Four 37-classification models
- Four 65-classification models
- Equal-weight average fusion
