# 3rd place solution

Competition: llm-prompt-recovery
Rank: #3
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494621

First of all, I’d like to thank my teammates and friends @tomirol and @pedromb – these guys are simply awesome.

I’d also like to thank Kaggle and Google for hosting this competition. Despite the whole “mean prompt” thing, working on this comp was quite fun and also an amazing learning experience as it was my first time finetuning LLMs.

As many other teams, our solution is a hybrid between mean prompt and models’ predictions. It involves 5 main components:
1. A mean prompt template (we format the string with model predictions)
2. A `MistralForCausalLM` finetuned to predict the full prompt.
3. A `MistralForSequenceClassification` trained to filter out blatantly wrong prompt predictions (a gate of sorts).
4. A `MistralForCausalLM` that predicts tags for the sample, e.g., “shanty”, “summarize”, “formal tone”, etc..
5. Two clustering models that cluster the test sample and selects the best mean prompt template for it.

We got to 0.71 using 1, 2, 3 and 4 alone. The cluster strategy improves the 0.71 but not enough to get us to 0.72 (locally provided +0.005CV) :(

The final solution is the mean prompt + tags + full prompt (if passes the gate). For the tags and the full prompt we only add unique words (meaning if they are already present in the mean prompt they are not added again). We also add the tags + full prompt after the third word in the mean prompt - we tried to find an optimal place for it and this one had the best results in local validation. The mean prompt used is selected based on the cluster.


Details for each component below:

# The Mean Prompt

To get to the mean prompt we did 3 steps:

1. Generate the data. First we generated a dataset of potential rewrite prompts (see the Model to predict full prompt section below to see how the dataset was prepared). 
2. Once we had this dataset we ran a procedure to select a subsample that follows the same distribution of the dataset in the public LB. We did this by matching the results of the LB when using a single prompt prediction with the selected subsample. The script to do it is here: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/prompt_selection.ipynb
3. We then ran a simple beam search to find the combination of words that would optimize the score over the selected dataset using a couple thousand words: https://github.com/matheuspf/llm-prompt-recovery/blob/main/src/optimization/mean_prompt_tokens.py

We noticed a very strong correlation between the score achieved on the subsampled dataset and LB using the optimized mean prompt, which indicated that we were indeed getting something useful. Here is our final (global) mean prompt:

>"improve phrasing text {here we format full prompt prediction + tags} lucrarea tone lucrarea rewrite this creatively formalize discours involving lucrarea anyone emulate lucrarea description send casual perspective information alter it lucrarea ss plotline speaker recommend doing if elegy tone lucrarea more com n paraphrase ss forward this st text redesign poem above etc possible llm clear lucrarea"

# Model to predict full prompt

Like many others we struggled to finetune a model that predicts the full prompt to get good results in this comp. Once we found the magical mean prompt we spent less effort on this. Early in the competition we got a zero-shot model that scored 0.61, but most of our fine tuned models wouldn’t go above 0.6. Eventually we got a model that scored 0.62 alone and iterated with it to find the version that would work the best with the mean prompt.

The final version used is a LoRA fine tuned `mistralai/Mistral-7B-Instruct-v0.2`. This mistral model was far and above the best on all our tests, much better than v0.1 and miles ahead of Gemma. Thanks to `unsloth`, we were also able to finetune it very efficiently, probably the best finding that came out of this competition. The training itself was very standard, no secret sauce here (script is here: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/train/lora_mistral_7b_unsloth.py), the gains came from the data generation process itself. Here is the final strategy that helped improve the results:

1. Generate prompt candidates using some LLM (mostly we used Gemini and gpt3.5 turbo). Important to prompt in a way that you also get the expected characteristics of the input text. This helps when selecting/generating the original text to prompt Gemma to transform. This is the script we used: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/prompt_generation.py
2. After that, generate some variations of the original prompts by directing the LLMs to change it somehow. This helps increase diversity and most importantly it creates variations of the tuple `(original_text, rewrite_prompt)` where the original text is the same and the rewrite intention is the same but expressed in a different way. The script used is this one: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/prompt_variations_generation.py
3. Generate the input text. We preferred to generate the original texts using an LLM to guarantee the input text characteristics matched the prompt. Again, mostly used gemini and gpt3.5 turbo for this task. Here is the script used: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/text_generation.py
4. Generate the Gemma versions of the rewritten text. In the beginning we were using the Gemma 7b-it-quant version, but this was super slow so eventually we settled on using the Gemma 2b-it-quant version instead together with `unsloth`. We still wonder if this affected the final results, it might, but with the mean prompt it became less relevant. Script used is here: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/gemma_2b_rewritten_text_generation_unsloth.py
5. Cluster the prompts and balance the train dataset by cluster. To do that we get the mean size across all clusters and select a sample of (max) that size for each cluster (I think I missed the script for this step, will try to find it to post later - but the clusters were generated using the T5 embeddings as features and `HDBSCAN` through `sklearn` for clustering, if a prompt was not clustered I would add it to the cluster that was more similar to it - using the mean of the embeddings of each cluster to calculate the similarity - if the max similarity was < 0.9 the prompt would be a single prompt cluster).
6. Postprocess the text to remove all the “Sure here is…” and other things to improve the overall dataset quality. Script is here: https://github.com/pedromb/llm-prompt-recovery/blob/main/src/data_generation/process_data.ipynb 

Important to note we also added all the public datasets shared in the comp to the final dataset and ran step 2 to generate the variation of the prompts. Step 2 and 5 combined was the breakthrough to get to 0.62. I believe forcing the model to see examples where the original text is the same and the rewrite instructions are similar in semantics but lexically different would force the model to learn how the variations on the rewrite prompt translates to the rewritten text. Maybe using the actual 7b version to generate the rewritten versions would have led to better results, but after getting here and with the discovery of the mean prompt we just tried different training settings and different clustering strategies, but nothing led to further improvements when combined with the mean prompt.

# Gate Model
At some point we realized that most of the time when the model that predicts the full prompt makes a mistake is usually quite noticeable, i.e., nothing to do with the ground-truth `rewrite_prompt`. Thus, we thought it would be possible to have another model to “gate” or filter our wrong predictions.
We trained the gate model by simply instantiating Mistral as `MistralForSequenceClassification` with a single class and we would use the following as prompt:
```
prompt = (
            "<s>[INST]Given the original text and a candidate prompt for rewriting it, you are expected to evaluate if the rewritten text makes sense given the candidate prompt. You will have positive examples where the rewritten text really was created by applied the proposed prompt and negative samples where the rewritten text was created by a different prompt."
            f'\n\nOriginal text: \n"""{original_text}"""\n\nCandidate Prompt: \n"""{candidate_prompt}"""\n\nRewritten text: \n"""{rewritten_text}"""[/INST]'
        )

```
At training time, 40% of the time we would use the correct triplet (positive sample), other 20% we would simply randomly select another `rewrite_prompt` from the dataset (easy negative) and 40% we would select another candidate prompt that is closer to the correct one in the T5 embed space (hard negative).

# Tags Model
We realized that it’s quite hard, if not impossible, to predict certain aspects of the `rewrite_prompt`. For example the main verb, e.g., “rewrite”, “rephrase”, “change”, “transform”, or the subject (even if `original_text` is a poem, the `rewrite_prompt` could be “Make the text happier” and predicting “poem” would not be very helpful). Thus, we trained a second MistralForCausalLM but to only predict tags about the sample and we noticed that this model would work well along the full prompt one provided that we simply remove tags that were already mentioned in the full prompt prediction.

# Clustering
Well, if 1 mean prompt is good,  2 or 3 or 12 might be better, right 😂? 
Jokes aside, when we discovered about the mean prompt we ran a LBFGS to optimize directly on the T5 embed space of our local validation set and we found that the best possible solution would score 72. Of course, the embedded space is of a continuous nature whereas when using words through T5 we can only approach it discreetly, which limits our score to ~0.7CV (0.69LB).

However, there is a way we can move even further from 0.7. If we think that the test distribution is, in fact, composed of many clusters we could try to find the mean prompt for each cluster instead. In the limit where the number of clusters equals the number of samples we have the full prompt prediction. We were basically trying to approach the task from the two extreme solutions (mean prompt vs. individual sample prediction) but we could actually try to operate at any point in the max score vs. task difficulty trade-off curve.

With that in mind we fitted a 12 cluster KMeans on the T5 embeddings of our local validation and ran the same LBFGS optimization. For 12 clusters the theoretical max scores were:
```
[(0, 0.7262467741966248),
 (1, 0.7863955497741699),
 (2, 0.8009814620018005),
 (3, 0.7827126383781433),
 (4, 0.8333203792572021),
 (5, 0.7971279621124268),
 (6, 0.8103494048118591),
 (7, 0.7536653876304626),
 (8, 0.7482798099517822),
 (9, 0.7608581781387329),
 (10, 0.7638632655143738),
 (11, 0.7885770201683044)]
```
And the weighted (by number of samples in each cluster) average would be ~76CV. Of course the issue with this approach is that we need to assign the correct cluster to the test sample since mis-assigning it leads to a high penalty. 

So we trained a `MistralForSequenceClassification` to classify each (`original_text`, `rewritten_text`) pair into one of the 12 clusters (we trained it using the `KMeans` prediction on the ground-truth `rewrite_prompt` as label). At inference time we would run both our classifier and `KMeans` on the predicted `rewrite_prompt` by the full prompt model and select the cluster only if both agreed. Otherwise, we would conservatively use the global mean prompt template.

Link to the training code for cluster, gate and tags models: https://github.com/arc144/kaggle_llm_prompt_recovery_public
Link to the inference kernel: https://www.kaggle.com/code/arc144/cluster-prompts/notebook

Extra challenge: the text above was rewritten by a LLM from the original write-up text, try guessing the rewrite prompt. Hint it was not `improve phrasing text lucrarea tone lucrarea rewrite this creatively formalize discours involving lucrarea anyone emulate lucrarea description send casual perspective information alter it lucrarea ss plotline speaker recommend doing if elegy tone lucrarea more com n paraphrase ss forward this st text redesign poem above etc possible llm clear lucrarea` 🤣
