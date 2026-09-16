# 1st Place Detailed Solution

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #1
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551688

I would like to thank Kaggle and EEDI for hosting this interesting competition! It was challenging and provided many opportunities to learn/apply new techniques. As always, I'm grateful to the Kaggle community for innovative ideas and engaging discussions. I'm excited to share my detailed solution and hope others will find it helpful!

## 1 Task
Given a diagnostic math MCQ, together with the correct answer and an incorrect answer, the task was to recommend top 25 misconceptions (sorted by their affinity to the incorrect answer) from a pool of 2.5k+ misconceptions. 
Even though the standard retrieve-rerank framework was a natural fit, it came with a few challenges:
- Current LLMs often struggle to mimic either (1) a `Novice Learner` i.e. generating the incorrect answer stemming from a specific misconception or (2) an `Expert Tutor` i.e. identifying the misconception that explains a given incorrect answer. LLMs excel at solving math problems, but they are not so great at counterfactual reasoning.
- The misconception pool ties together closely related conceptual and computational mistakes, demanding high discriminative power to spot subtle differences with precision. This is highlighted in the competition overview:
>Initial efforts to use pre-trained language models have not been successful, likely due to the complexity of the mathematical content in the questions.

- The task setup requires the model to not only do well on known misconceptions but also generalize to new misconceptions.

## 2 Meta Thoughts
At the start of the competition, I had the following hypotheses on what could be important for a good solution:
- Synthetic data should help with out-of-distribution generalization
- High quality synthetic data generation should be feasible since Math domain can be verified and curated objectively
- Distillation from a good teacher(s) should be very powerful, as Large LLMs are significantly better at reasoning
- Distilling Chain of Thoughts (CoT) from top LLMs (e.g. Claude 3.5 Sonnet) should help to tackle difficult examples
- Quantization method and the calibration datasets could be important. We may need to add recovery adapters to the quantized models for maintaining top 1 accuracy.
- Full training might be better than LoRA

These hypotheses guided me to design my experiments throughout the competition and helped me to stick to an idea even though the initial attempts were not successful.

## 3 Pipeline
I implemented a retrieve-and-rerank system that involved a sequence of four steps:
- **Retrieval**: Identify top misconception candidates for each of the MCQ-Incorrect answer combinations. Specifically, I retained the top 32 retrieved candidates and included up to 32 additional candidates based on a dynamic threshold. The dynamic threshold is computed as the `top candidate similarity score - constant`. Candidates with similarity scores higher than the dynamic threshold are kept. 
- **14B Ranker**: Use a fine-tuned `Qwen/Qwen2.5-14B` ranker to process all retrieved candidates and shortlist top 8.
- **32B Ranker**: Use a fine-tuned `Qwen/Qwen2.5-32B` ranker to process top 8 candidates and narrow them down to top 5.
- **72B Ranker**: Use a fine-tuned `Qwen/Qwen2.5-72B` ranker to finalize the ranking of top 5 candidates.

A schematic diagram of the pipeline and the leaderboard scores from each stages are shown below:
[Inference Pipeline]

The final 25 misconception predictions comprised of:
- 1st to 5th candidate from the 72B ranker
- 6th to 8th candidate from the 32B ranker
- 9th to 25th candidate from the 14B ranker.

## 4 Data
The training data mix for both the retriever and ranker models included competition data (1.8k examples) and synthetic data (10k examples). The synthetic data played a crucial role in improving both raw performance and generalization capability with respect to unseen misconceptions.

### 4.1 Synthetic Data
In an ideal scenario, synthetic MCQs should have the following properties:
- The designated correct answers are indeed the correct answers (high accuracy).
- The incorrect answers directly stem from the target misconception (high diagnostic power).
- Among all misconceptions in the misconception pool (2.5k+), the indicated misconception should provide the most precise explanation for a given incorrect answer (high resolution).
- The generated MCQs should cover a broad range of subjects and constructs (high diversity).

I started generating diagnostic MCQs with a simple prompt that specified the expected format and a misconception of interest. However, it resulted in a dataset that violated almost all the properties to varying degrees. It was expected since
- The simple prompt only included information about a single misconception. Hence, the data-generating LLM was completely unaware of closely related misconceptions within the misconception pool.
- Lack of few shot examples meant lack of reference on the expected quality, difficulty level, tone, and language.
- Even the advanced LLMs struggle with handling misconceptions and counterfactual reasoning effectively.
- No curation was performed.

I experimented with several strategies to address the issues and managed to create a high quality synthetic dataset with a `grouped synthetic data generation` approach and incorporating LLM based filtering. Specifically, I first created clusters of closely related misconceptions leveraging co-occurrence statistics of misconceptions in retriever/ranker predictions on validation data. [This notebook](https://gist.github.com/rbiswasfc/8d2bfec5c2a358e8beeb2df390111f9d) demonstrates the clustering process. Here is an example of a misconception cluster:
```
- Thinks x^2 - a^2x is a valid form of difference of two squares
- When factorising a quadratic without a non variable term, tries to double bracket factorise
- Incorrectly factorises a quadratic
- Believes the constant in an expanded quadratic comes from adding the two numbers in the brackets
- Believes the coefficent of x in an expanded quadratic comes from multiplying the two numbers in the brackets
- Does not recognise difference of two squares
- Does not realise a quadratic must be in the form ax^2+bx+c=0 to be factorised
- Does not know how to identify common factors from algebraic terms
- Believes the difference of 2 squares method of factorising also works for the addition of 2 squares
- When factorising, believes they can choose different factors for each term
- Believes they can factorise a difference of two squares by placing the constant in both brackets without square rooting
```

I next generated 5 new MCQs for each cluster using `Claude 3.5 Sonnet` with a refined prompt that included 4-5 reference examples on misconceptions from the cluster. I used the [metaprompt](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/metaprompt.ipynb) tool from Anthropic cookbook for prompt engineering. Here is the final prompt used for data generation:

```
You will be generating Multiple Choice Questions (MCQs) that diagnose specific mathematical misconceptions. Here are the misconceptions you should focus on:

<misconceptions>
{cluster_misconceptions}
</misconceptions>

Here are reference MCQs that demonstrate how to effectively diagnose these misconceptions:

<reference_mcqs>
{reference_mcqs}
</reference_mcqs>

Your task is to generate {num_mcqs} new MCQs that diagnose misconceptions not already covered by the reference MCQs.

First, analyze the reference MCQs carefully:
1. For each reference MCQ, identify in your <analysis> tags:
   - Which misconception it targets
   - How the incorrect answers map to specific misconceptions
   - What makes the question effective at diagnosing the misconception
2. Note the style, difficulty level, and precision of language used

Then, in your <planning> tags:
- List which misconceptions still need coverage
- For each needed misconception, brainstorm mathematical contexts where it commonly appears
- Design questions where the misconception leads naturally to specific wrong answers
- Take notes on how you can craft new MCQs that adheres to the reference MCQs' style, difficulty level, and precision of language

Finally, generate new MCQs following these important guidelines:
- Make sure each incorrect answer maps clearly to exactly one misconception
- Use precise mathematical language matching the style of reference MCQs
- Make questions challenging enough that students must demonstrate real understanding
- Ensure wrong answers are plausible and stem from genuine misconceptions, not careless errors
- Use the exact wording of misconceptions as given in the misconceptions list
- Pay careful attention to subtle differences between the misconceptions and observe which one is the most appropriate for a given incorrect answer
- Keep the construct name and subject name as short as possible hiding the details of the misconception
- Questions should be of higher difficulty level than reference MCQs
```

### 4.1.1 Curation
I used LLM-as-a-judge to filter out low quality synthetic MCQs. I prompted `GPT-4o` to rate the quality of the synthetic MCQs as below:

```
You will analyze how well an incorrect answer reflects a suspected misconception in a mathematics problem. Your goal is to determine whether there is a clear, logical connection between the misconception and the wrong answer.

Here is the problem with both correct and incorrect answers. The suspected misconception is also provided:
<problem>
{PROBLEM_DATA}
</problem>

First, analyze the problem in your scratchpad:
<scratchpad>
1. Solve the problem independently to verify the correct answer
2. Examine how someone holding the suspected misconception would approach the problem
3. Trace the logical path from misconception to incorrect answer
4. Identify any gaps or inconsistencies in this connection
</scratchpad>

Then provide your evaluation using this format:
<evaluation>
1. Brief explanation of how the misconception could lead to the wrong answer
2. Score from 0-10 based on these criteria:
   - 10: Perfect alignment - wrong answer is direct result of misconception
   - 8-9: Strong alignment - clear logical path from misconception to answer
   - 5-7: Moderate alignment - connection exists but has some gaps
   - 1-4: Weak alignment - connection is unclear or requires assumptions
   - 0: No alignment - misconception does not explain wrong answer
</evaluation>

Important guidelines:
- Focus solely on the logical connection between misconception and wrong answer
- Do not speculate about other possible misconceptions
- Be specific about how the misconception leads to the error
- Flag and deduct scores if any assumptions are required to connect misconception to answer
- Consider whether a student with this misconception would consistently arrive at this wrong answer
```

I selected `GPT-4o` as judge based on vibe test as it felt more sensitive to the scoring scheme and as consistent as Claude 3.5 Sonnet.

### 4.1.2 Incorporating Additional Misconceptions
While creating synthetic data, LLM produced many misconceptions (4k+) that were not part of the host provided misconception pool. I decided to keep the external misconceptions to improve generalization capability of the retriever and ranker models. 

Naively merging the external misconceptions with existing ones would have introduced conflicts and noise, since many of them were just rephrasing of the existing misconceptions. I adopted the following steps to handle this carefully:
- Attempt to match with existing misconceptions using string normalization (lowercasing, removing trailing punctuation, etc.)
- Attempt to match with existing misconceptions using similarity scores from an embedding model (`thenlper/gte-base`) with very high threshold (0.995)
- Remove new misconceptions that have fairly high similarity (between 0.95 and 0.995) with any of the existing misconceptions
- Keep rest of the remaining external misconceptions

### 4.1.3 Dataset
The final dataset is uploaded here: [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/datasets/conjuring92/eedi-mcq-dataset/). It contains
- 1.8k competition MCQs + 10.6k generated MCQs
- 4791 misconceptions
and follows the same format as the original competition dataset.

Reference: I'd highly recommend reading the synthetic data generation post from answer.ai: [How To T̶r̶a̶i̶n̶ Synthesize Your D̶r̶a̶g̶o̶n̶ Data](https://www.answer.ai/posts/2024-10-15-how-to-synthesize-data.html).


### 4.2 Chain of Thought (CoT) Generation
To distill the reasoning capability of an advanced LLM in my solution pipeline, I first created a dataset containing thought processes generated by `Claude 3.5 Sonnet`. I provided Claude with: (i) the problem statement, (ii) the correct answer, (iii) an incorrect answer, (iv) the ground truth misconception, and (v) several closely related misconceptions. Using this information, I prompted Claude to generate the likely thought process that led a student to select the incorrect answer. The prompt is shown below:

```
You will analyze a student's incorrect answer to identify the specific reasoning flaw that led to their error.
Your goal is to explain precisely how their misconception caused them to arrive at the wrong answer.

Here is the problem information:
<problem_data>
# Question: Simplify the expression: \[x \cdot y \cdot x\]
# Correct Answer: \(x^2y\)
# Incorrect Answer: \(x^2\)
# Primary Misconception: Ignores variables without explicit coefficients when multiplying
</problem_data>

Here are related misconceptions that are similar but do not explain this specific error as precisely:
<related_misconceptions>
- Thinks only like terms can be multiplied
- Fails to combine all instances of the same variable
- Incorrectly identifies an incomplete variable factor
- Does not understand how to multiply algebraic terms
</related_misconceptions>

First, examine all components of the problem carefully:
1. The problem statement and question asked
2. The correct answer and solution method
3. The student's incorrect answer
4. The primary misconception given
5. The related misconceptions that should be distinguished from the primary one

Then, reconstruct the student's likely thought process:
- Identify the exact point where their reasoning diverged from the correct solution path
- Note which specific mathematical operations or concepts they misapplied
- Connect their error directly to the stated primary misconception
- Verify that this explanation better fits the error than the related misconceptions

Write your analysis in <evaluation> tags, following this structure:
- Show the correct calculation first
- Show the incorrect calculations that demonstrate the error
- Explain the specific flaw in the student's reasoning
- Demonstrate how the misconception led to this particular error
- Distinguish from the related misconceptions
- Keep your explanation to 5-6 clear, non-repetitive sentences
- Focus solely on the reasoning that produced this specific error

Guidelines for writing your explanation:
- Do not restate the problem or name the misconception
- Be precise about the mathematical concepts involved
- Show exactly how the misconception led to the error
- Distinguish from related misconceptions
- Avoid repetition
- Stay focused on this specific error
```

Using this dataset, I fine-tuned three Qwen 2.5 series models (`Qwen/Qwen2.5-Math-7B`, `Qwen/Qwen2.5-14B`, and `Qwen/Qwen2.5-32B`) to act as reasoners. Notably, I omitted the true and related misconceptions from the input for fine-tuning. During inference, these models would see: (i) the problem statement, (ii) the correct answer, and (iii) an incorrect answer. They would then generate the likely thought process of a student behind selecting the incorrect answer. I used `vllm` for CoT generation during inference with `temperature=0.7, top_p=0.8, repetition_penalty=1.0, max_tokens=256`.

The CoT dataset is uploaded here: [Eedi - CoT Dataset](https://www.kaggle.com/datasets/conjuring92/eedi-sonnet-cot-comp-sharp-dec-09/).

## 5 Train-Validation Split
I split the competition dataset into 5 folds, initially based on the `QuestionId` field. However, this produced overly optimistic estimates with a large gap between validation and leaderboard scores. Next, I tried GroupKFold on `SubjectId`, but this resulted in overly pessimistic estimates. I finally settled on GroupKFold with `ConstructId`, which yielded a narrow Validation/LB gap and good correlation up to ~0.62 public LB score range.

During development, I always trained my models on folds 1-4 and validated on fold 0. The synthetic data was marked as fold 99 and was included in the training set. The final models were trained on the entire dataset (full fit) using the same hyper parameters from the development stage.


## 6 Retrievers
I used the standard [MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) to fine-tune the retrievers, while referencing the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/finetune/embedder/decoder_only) codebase for developing training scripts. At this stage, I tracked both `map@25` and `recall@32` metrics and prioritized optimizing for `recall@32`, since recall was more important for overall pipeline performance.

I fine-tuned several retrievers with the following scores:
[Retriever Scores]

Incorporating hard negatives into training batches and distilling re-ranker scores consistently improved `map@25` performance, but did not positively impact `recall@32`. For my final submission selections, I chose the high-recall `Qwen/Qwen2.5-14B` encoder (Model 3) instead of the best `map@25` encoder (Model 4). Following the same logic, I also excluded the BGE encoder (Model 2).  Notably, my best encoder-only submission (Model 1 + Model 2 + Model 4) scored 0.524 on public LB and 0.475 on private LB.

Retrievers were fine-tuned using LoRA with: `r=64, alpha=128, learning_rate_lora_a=1e-5, learning_rate_lora_b=5e-5, LoRA on all linear layers, batch_size=128, and epochs=12`. A few key factors that improved recall performance were:
- Setting temperature to 0.01, instead of the typically used value of 0.02 in LLM-based encoders
- Ensuring only one demonstration per misconception appeared in each training batch, as multiple demonstrations would introduce label noise through in-batch negatives
- Pretraining the `Qwen/Qwen2.5-14B` encoder with all available synthetic data (before the curation step in section 4.1.1).

What didn't work:
- Iterative hard negative mining
- Increasing batch size [through cross-device negatives](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/abc/finetune/embedder/AbsModeling.py#L194)
- Custom batching strategies e.g. having (query, misconception) positive pairs from the same SubjectId in the same batch
- My attempts at converting LLM retrievers to bi-directional encoders similar the the strategy used in [nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2/blob/main/modeling_nvembed.py#L33).

## 6 Re-rankers
The Qwen 14B ranker did most of the heavy lifting by processing all retrieved misconceptions (32-64 candidates) and identifying top 8 candidates. This model was trained with a pointwise approach, where it sees one misconception at a time in the context window. The model input was structured as follows:
[Pointwise Ranker]

The logits were computed by taking the difference between the 'Yes' and 'No' token scores, as below:

```python
outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
logits_yes = outputs.logits[:, -1, self.yes_loc]  # Yes token logit at the last position [bs]
logits_no = outputs.logits[:, -1, self.no_loc]  # No token logit at the last position [bs]
logits = logits_yes - logits_no  # [bs]

logits = logits.reshape(-1, self.group_size)
labels = labels.to(logits.device).reshape(-1) 
ce_loss = self.loss_fn(logits, labels)
```

Each batch contained 1 positive and `batch_size-1` negative misconceptions for a given Question-Incorrect Answer combination. The label was set to the index of the positive misconception and cross entropy loss was used for training. The models was trained with LoRA with `r=64, alpha=128, LoRA on all linear layers`. The language modelling head (`lm_head`) of the LLM (`Qwen/Qwen2.5-14B`) was frozen during fine-tuning.


### 6.1 Ablations
I found the following 4 strategies to be particularly significant for boosting the reranker performance and its generalization capability:

### 6.1.1 Few Shot Examples
To retain and leverage the LLM's in-context learning capabilities, I optionally included a few reference misconception examples (0 to 2) in the model input context (as shown in the diagram above). Not all training examples contained these few-shot demonstrations. The objective was to encourage the model to use reference examples when available, and otherwise rely on its internal reasoning. During inference, I used 1 or 2 examples from the entire training set as demonstrations. This approach was particularly effective for the 14B ranker's performance boosting private LB score from 0.495 to 0.531 **(+0.036)**. At this point, I was only using the competition data for training.

### 6.1.2 Distillation / Pseudo Labelling
Next, I incorporated the synthetic examples (post curation) into the training pipeline through pseudo labeling. I first fine-tuned two pointwise 72B models (`Qwen/Qwen2.5-Math-72B` and `Qwen/Qwen2.5-72B`) using only the competition dataset. These models were then used to generate pseudo labels for the synthetic examples. Finally, I used competition + pseudo-labeled data to fine-tune the next iteration of the 14B rerankers. This strategy boosted the 14B reranker's private LB score from 0.531 to 0.575 **(+0.044)**.

### 6.1.3 Negative Ratio
Next, I experimented with increasing the number of negative samples per positive. Showing the rankers a large number of negatives during training helped to improve performance. For each positive, I next increased the number of negatives to 24 from earlier 16. At this point, I also added scaled number of synthetic examples 2x. These two modifications boosted the 14B reranker's private LB score from 0.575 to 0.596 **(+0.021)**.

### 6.1.4 Chain of Thought (CoT)
I finally added the fine-tuned CoT reasoners (Section 4.2) into training and inference pipeline. During training, CoTs produced from fine-tuned `Qwen/Qwen2.5-14B` reasoner were optionally added to the model input. 50% of training data used CoTs and 50% did not. This encouraged the model to use external reasoning when available and otherwise rely on its internal reasoning. CoT boosted the 14B reranker's private LB score from 0.596 to 0.615 **(+0.019)**.

The impact of these strategies is summarized in the following table:
[Ablation]

The live eval during fine-tuning showed consistent improvement in the 14B reranker's performance.
[Live Eval]

### 6.2 Pointwise Re-Ranker (`Qwen/Qwen2.5-32B`)
I fine-tuned the `Qwen/Qwen2.5-32B` model in exactly the same way as the 14B reranker (with a different seed). My best 32B model had validation score of `0.663` vs that of `0.646` for the 14B reranker. During inference, the 32B reranker was used to process top 8 candidates from the 14B reranker and narrow them down to top 5 candidates. Incorporating the 32B reranker boosted private LB score from `0.615` to `0.625` **(+0.010)**.

### 6.3 Listwise Re-Ranker (`Qwen/Qwen2.5-72B`)
The top 5 candidates were finally ranked in a listwise manner using a fine-tuned `Qwen/Qwen2.5-72B` model. Here, the model sees all of the top 5 candidates at once. An example model input is shown below:
[Listwise]


This model had access to the even more information than pointwise re-rankers i.e.
- It sees 3 Chains of Thought (CoT) during inference - one each from the 7B, 14B, and 32B reasoners
- It sees up to 5 reference examples from the training set - 1 for each of the 5 candidate misconceptions
- It sees all of the candidates at once in the context window.

The model had the best `MAP@5` among all of the rerankers (validation MAP@5 of `0.661`). LLM based Listwise rankers typically suffer from candidate position bias. One way to mitigate this is to run multiple inference runs by shuffling the position of the candidate misconceptions (Test time augmentation (TTA) like impact). I was unable to run TTA during inference due to time constraints. Incorporating the 72B reranker boosted private LB score from `0.625` to `0.638` **(+0.013)**.

### 6.4 Quantization
I quantized the fine-tuned rankers using [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). I used task specific calibration datasets for each model to mitigate performance degradation. I tested the impact of calibration dataset with one of my 32B rerankers. The results are shown below:



### 6.5 What didn't work
- Merging strategy as described in [Continuous Fine-tuning Without Loss Using Lora and Mergekit](https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?tab=t.0)
- [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview): I naively fine-tuned it using the same steps as with `Qwen/Qwen2.5-32B` but got worse results. The QwQ model likely requires additional research to find out its proper usage.

## 7 Links
- [Training code](https://github.com/rbiswasfc/eedi-mining-misconceptions)
- [Inference Notebook](https://www.kaggle.com/code/conjuring92/eedi-a2-pipeline?scriptVersionId=211785645)
- [Dataset with Synthetic + Competition MCQ](https://www.kaggle.com/datasets/conjuring92/eedi-mcq-dataset)
- [CoT Dataset](https://www.kaggle.com/datasets/conjuring92/eedi-cot-sonnet-6k)

## 8 References
- [Novice Learner and Expert Tutor: Evaluating Math Reasoning Abilities of Large Language Models with Misconceptions](https://arxiv.org/pdf/2310.02439)
- [vllm](https://github.com/vllm-project/vllm)
- [How To T̶r̶a̶i̶n̶ Synthesize Your D̶r̶a̶g̶o̶n̶ Data](https://www.answer.ai/posts/2024-10-15-how-to-synthesize-data.html)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
