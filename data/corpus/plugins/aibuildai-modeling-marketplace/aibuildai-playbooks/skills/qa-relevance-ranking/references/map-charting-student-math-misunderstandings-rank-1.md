# 1st Place Solution

Competition: map-charting-student-math-misunderstandings
Rank: #1
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/1st-place-solution

# Solution Writeup

I would like to thank the competition hosts and Kaggle for organizing this interesting challenge。
Winning 1st place in an NLP competition for the first time feels great! Below is my solution writeup.

# 1. Data

I dropped duplications, resulting in 35,960 samples. Created 5-fold cross-validation splits stratified by "Category".

# 2. Modeling

I modeled the problem as a suffix classification task.

Given the same context(prefix), models are trained to predict the correct suffix from a set of candidates.

Context is formatted as follows:
```
<|im_start|>user
**Question:** {QuestionText}
**Choices:** {MC_Choices}
**Correct Answer:** {Answer}
**Common Misconceptions:** {MisconceptionCandidates}
**Student Answer:** {MC_Answer}
**Student Explanation:** {StudentExplanation}
<|im_end|>
<|im_start|>assistant
```

Suffixes are formatted in submission format.
```
False_Correct:NA<|im_end|>
```

Each `QuestionId` has 8, 10 or 12 possible suffix candidates based on possible misconceptions of the question.

Last-token features of `[prefix ++ suffix0, prefix ++ suffix1, ...]` are extracted and fed into a nn.Linear(hidden_size, 1) to get logits, then cross-entropy loss is computed.
In practice, I organized inputs in prefix-shared format: `prefix ++ suffix0 ++ suffix1 ++ ...`, using custom attention masks with FlexAttention.

```python
def custom_mask(b, h, q_idx, kv_idx):
    causal = q_idx >= kv_idx
    is_prefix = suffix_ids[kv_idx] == -1
    same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx])
    same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
    return causal & (same_suffix | is_prefix) & same_doc
```

# 3. Training

While working on the `WSDM Cup` competition earlier this year, I implemented  [offload_adam](https://github.com/tascj/offload_adam) to enable efficient full-parameter training of models up-to 32B on single A100 80G GPU. So in this competition, I could run all experiments on single A100 80G or RTX Pro 6000 Blackwell. Most of the training runs were performed with same hyperparameters: `epoch=1, batch_size=32, learning_rate=1e-5`.

Probably due to label noise (mainly `Neither`), validation score varies a lot using different seeds. I had to use ensemble of multiple runs with different seeds to get a stable validation score.

I did a 5-fold x 5-seeds run using `Qwen3-8B`. 3 seeds ensemble seemed to be stable, so I used the most difficult fold and ensemble of 3 runs for further experiments. This approach is not new - the [1st place solution](https://www.kaggle.com/competitions/feedback-prize-effectiveness/writeups/team-hydrogen-team-hydrogen-1st-place-solution) in "Feedback Prize - Predicting Effective Arguments" also used 3-seed ensemble for validation.

**Main Validation Results (seed ensemble [seed1, seed2, seed3])**

| model                                    | loss                            | MAP@3                           |
|:-----------------------------------------|:--------------------------------|:--------------------------------|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  | 0.2716 [0.2809, 0.2871, 0.2835] | 0.9444 [0.9426, 0.9428, 0.9421] |
| Qwen/Qwen3-8B                            | 0.2677 [0.2813, 0.2777, 0.2756] | 0.9455 [0.9433, 0.9450, 0.9437] |
| zai-org/GLM-Z1-9B-0414                   | 0.2627 [0.2783, 0.2762, 0.2761] | 0.9469 [0.9455, 0.9433, 0.9452] |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 0.2621 [0.2782, 0.2698, 0.2738] | 0.9464 [0.9421, 0.9443, 0.9450] |
| Qwen/Qwen3-14B                           | 0.2614 [0.2707, 0.2744, 0.2695] | 0.9477 [0.9442, 0.9451, 0.9461] |
| Qwen/Qwen3-32B                           | 0.2589 [0.2687, 0.2718, 0.2700] | 0.9484 [0.9465, 0.9451, 0.9454] |
| zai-org/GLM-Z1-32B-0414                  | 0.2560 [0.2713, 0.2681, 0.2682] | 0.9480 [0.9450, 0.9463, 0.9462] |
| Qwen/Qwen3-32B+zai-org/GLM-Z1-32B-0414   | 0.2530                          | 0.9496                          |


The experimental results clearly demonstrated that:

1. I should not trust single-seed validation score
2. The larger the model, the better the performance
3. It might be better to trust loss rather than MAP@3

I also compared multi-fold ensemble with multi-seed ensemble. Multi-seed ensemble was much better. Therefore, for the final submission, I trained `Qwen/Qwen3-32B` and `zai-org/GLM-Z1-32B-0414` on full dataset with 3 different seeds and slightly different data format each.

### Auxiliary SFT Loss Experiment

In the final days of the competition, I experimented with using `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8` to generate short justifications for the labels, and trained models with auxiliary SFT loss on the generated content. 

An example of generated justification:
```
True_Correct:NA
The categorization **True_Correct** with **NA** misconception is justified because:  
- **Answer correctness**: The student's answer \( \frac{1}{3} \) is mathematically correct. With 9 total triangles and 6 shaded, 3 are unshaded. Simplifying \( \frac{3}{9} \) yields \( \frac{1}{3} \), matching the required simplest form.  
- **Explanation quality**: The explanation ("There is 9 triangles and 3 aren't shaded") correctly identifies the total parts (9) and unshaded parts (3), demonstrating valid reasoning for the fraction \( \frac{3}{9} = \frac{1}{3} \). It is concise but clear and mathematically sound.  
- **Misconception**: No error exists in the reasoning (e.g., miscounting shaded/unshaded parts or incorrect simplification), so "NA" applies. The explanation omits explicit simplification but implies it by providing the correct simplified answer.
```

This approach showed some improvement for `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, but had minimal effect on `Qwen/Qwen3-8B`. I included it in the ensemble, though I estimate its contribution is marginal. Anyway, it was an interesting attempt.

# 4. Inference

There are two main challenges for inference: compute and memory.

## Compute

For short-sequence prefilling workload, the main bottleneck is matrix multiplications in `nn.Linear` layers. T4, while claiming 65TFLOPS of FP16 on whitepaper, could only achieve unstable 20TFLOPS in practice. 

The W8A8 INT8 kernel in LMDeploy achieves stable 40+TFLOPS. To enable W8A8 INT8 inference, I quantized models using SmoothQuant(alpha=0.75). Ensemble performance in validation is almost identical to unquantized models.

## Memory

I used layer-wise inference to enable 32B model inference on a single T4 GPU:
1. Initialize and keep only 2 transformer layers on GPU
2. Overlapping `execution of current layer` with `loading states of next layer from disk`

I used 640 samples (40 micro batches of 16 samples) in each forward run to keep GPUs busy.

It takes around 65 minutes to finish inference of 16,000 samples with T4×2. Implementing prefix-caching/sharing of question context could further reduce inference time (more than half of the tokens could be cached), but I didn't implement it.

### Kaggle Notebook Environment Notes

Kaggle notebook environment's storage caused me some trouble.

1. `/kaggle/input` - Not local storage, very slow access
2. `/tmp/` - Local Copy-on-Write storage with capacity limits, **files cannot be truly deleted**
3. `/kaggle/working` - Regular local storage, files can be deleted to free up space

I initially stored checkpoints of layers in `/tmp`, which led to a crash due to capacity overflow. After debugging, I only had time to submit an ensemble of 4 models (out of 6 uploaded). Anyway, the results were still quite good.

# 5. Summary

For me, the most critical aspect of this competition was finding the right way to obtain reliable validation scores. Single-seed validation scores were highly unstable and misleading. Multi-seed ensemble provided stable and trustworthy validation metrics, which then enabled effective model tuning and selection.

[Submission notebook](https://www.kaggle.com/code/tascj0/map-submit/notebook)
[Training](https://github.com/tascj/kaggle-map-charting-student-math-misunderstandings)
