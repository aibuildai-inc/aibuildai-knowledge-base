# 18th Place - Pyramid Ensemble

Competition: map-charting-student-math-misunderstandings
Rank: #18
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/18th-place-pyramid-ensemble

Thank you Kaggle and community for an enjoyable competition. My final solution is a "pyramid ensemble" to maximize the efficient use of limited submission inference time and compute. 

First we infer 100% test samples with small models, then we determine which predictions the ensemble is more uncertain about. Next we predict top 50% uncertain test samples with medium models. We incorporate these predictions with 2x weight and recompute prediction uncertainty. Next we predict top 10% uncertain with large models. We incorporate these new predictions with 4x weight and recompute prediction uncertainty. Finally we predict top 6% with an extra large model and incorporate these predictions with 8x weight. See diagram for more info.



My final "pyramid ensemble" submission has 5 KFold `CV = 0.951`, `Public LB = 0.952`, and `Private LB = 0.947`.

# Diversity
I joined this competition in the beginning, so I have 3 months to investigate many diverse approaches. In 3 months, I trained 1000 LLM+ models. Every day I had about 4xA100 GPUs running experiments. I investigated the following things:
* Different  Backbones (i.e. Qwen, Gemma, Llama, etc)
* Different Paradigms (i.e. Sequence classification, Text generation, Multiple choice, multiple heads, etc)
* Different Prompts (i.e. list MC_Answer choices, add 'is_correct = Yes/No`, list target class choices, add system roles, etc)
* Generative AI assistance. (i.e. Create synthetic data, vibe coding, etc)
* Different training configurations (Full finetune, LORA, QLORA, layer decay, warm ups, etc)
* Different inference configurations (vllm, huggingface, awq, bitsandbytes, etc)

# Multiple Paradigms
## Paradigm - Sequence Classification and/or Text Generation
There were many ways to solve this problem. The most straightforward was using `AutoModelForSequenceClassification()` with 65 classes or remove True False for 37 classes. Another powerful approach is `AutoModelForCausalLM()` where the model generates a single token to predict a target class and we extract that token's logits/probability.

## Paradigm - Multiple Choice
Another approach that worked well was converting the problem to LLM Multiple Choice like:

    Question: A box contains 120 counters. The counters are red or blue. 3/5 of the counters are red. How many red counters are there?
    Correct Answer: 72
    Student Answer: 24
    Student Explanation: B because 3 fifths of 5 is 75
    Choices: A) Correct B) Irrelevant C) Incomplete D) Wrong_Fraction E) Neither

Once we remove the prefix of "True_" and "False_", there are only 37 target classes. Then for each question, there are always 6 or less possible target classes. So every train sample and test sample can be reformulated with choices A, B, C, D, E, F (where sometimes C, D, E, and/or F are empty). And we can randomly shuffle the choices. The model must learn to read the choices, understand the choices, and pick 1 of the 6. Afterward, we map the prediction into a Numpy array of probabilities with 65 columns (or the 65 original target classes).

## Paradigm - Multi-head Architecture
The most fun paradigm was multihead. It was difficult to get work but I finally got mulit-head to work and perform well using 3 heads. The first head predicts the column `Category` = ["True_Correct", "True_Neither", "True_Misconception", "False_Correct", "False_Neither", "False_Misconception"]. And the second head predicts `True Misconceptions` and the third head predicts `False Misconceptions`. For a given train sample, the first head always provides loss and then heads 2 and 3 may or may not provide loss.

For example if the full label is `True_Correct:NA`. The total loss is just the first head (and heads 2 and 3 provide no loss). If full label is `True_Misconception:Shorter_is_bigger`. Then the total loss is first head loss plus second head loss (and head 3 provides no loss). And if full label is `False_Misconception:WNB` then the total loss is first head plus third head (and head 2 provides no loss).

# Prompt Diversity
There were so many options with prompt creation. For example, do we add `is_correct = Yes/No` into the prompt, or do we let the model determine this itself? Decisions like this made a big difference. Also we can list the MC_Answer choices. And we can list the possible target class choices for a specific QuestionId, MC_Answer combination. Also we can use chat formatting with system and user role. Or we cannot. We can add extra info to help classify targets. My best single model [notebook here][1] (CV 0.949, LB 0.949) used the following prompt:

    You are an expert mathematics assessor.
    Question: What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]
    Choices: A) \( \frac{1}{3} \), B) \( \frac{3}{6} \), C) \( \frac{3}{8} \), D) \( \frac{3}{9} \)
    Response: \( \frac{3}{9} \)
    Explanation: 6/9 is shaded therefore 3 must be left therefore the answer is 3/9.
    Now classify the explanation as Correct explanation (regardless of whether response is correct), Misconception:Incomplete, Misconception:WNB, or Not an explanation.

Here are some details about this prompt:
* Adds a system like instruction
* Enumerates MC_Answer choices
* Lists target classes for this specific QuestionID and MC_Answer
* Clarifies some target classes like explain that "Correct" is "regardless of whether response is correct".

# Generative AI
I tried many experiments with synthetic data using GPT. First, using my local OOF for all train data, I determined which classes are being confused. Then I prompted ChatGPT4 to specifically make synthetic data which will help an LLM learn the difference between these confused classes. Of course I did this in an "out of fold fashion" to prevent leakage. The results were mixed, sometimes improvement but mostly not. In the end I didn't use any synthetic data because I did not trust it.

# Train Setup
For models 14B and less, I mostly did full finetuning. For models 14B and larger I mostly did LORA or QLORA. (For 14B models I did both LORA and non-LORA). I trained models using A100 GPU 80GB VRAM. Usually I would train using a single GPU but with the largest 72B LLM, I trained using 2xA100 or 4xA100. My typical LORA settings were r=32, a=64, dropout=0.05 or 0.1, target modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]. I would use gradient checkpointing and large batches like 64. Learning rate was usually around 1e-4 or 2e-4. Optimizer was often AdamW4bit.

# Infer Setup
I did lots of experiments to optimize inference. My main innovation was to make sure both 2xT4 GPU were always running at 100%. So if a model only needed 1 GPU, I would run two threads and infer the first half of test.csv while the second thread was inferring the second half of test.csv. For larger models, using CasualLM and vllm was the most efficient and fastest. But I also used pure PyTorch and/or HF Trainer for inference too.

# Backbones
Wow! There are so many LLM out there to try. I probably tried every LLM backbone between 1B and 72B. Haha, yes I trained a lot of models. For each backbone, I would try different paradigms, prompts, train setups, and inference setups. Using ChatGPT helped me get everything up and running fast and solve any errors that arose.

[1]: https://www.kaggle.com/code/cdeotte/single-model-cv-0-949-lb-0-949-wow
