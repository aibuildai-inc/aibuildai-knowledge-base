# 21st place private LB solution

Competition: kaggle-llm-science-exam
Rank: #21
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/447589

Hi fellow kagglers,

We are the “Contextual Kagglers” team ([Alex](https://www.kaggle.com/alexryzhkov), [Giba](https://www.kaggle.com/titericz), [Olya](https://www.kaggle.com/olyatsimboy) and [Dima](https://www.kaggle.com/ey4read1mk4realtv)) with the 0.916 private LB score (21st place) and 0.928 public LB score (11th place). Yep, we were in gold medals on public and gone out of it on private - sad news 🙁

Here we want to disclose our solution and share some interesting insights on the data and competition itself. One more thing - I want to mention here that the **majority of ideas and work were done by [Olya](https://www.kaggle.com/olyatsimboy) and [Dima](https://www.kaggle.com/ey4read1mk4realtv)** so they have had a huge impact on the final solution score. I also would like to congratulate [Olya](https://www.kaggle.com/olyatsimboy) with **her new Kaggle Competitions Expert tier**.

## Final structure

The **structure of our solution** is like the scheme below:



## Retrieval strategies

It contains **2 different context retrieval strategies**:
- the first is based on the **full pack of wikipedia articles sentences (around 117 mln elements)** and `all-MiniLM-L6-V2` and `GTE-base` based FAISS quantized indices (we use IVF, PQ64 index). Here we used **both variants (only prompt and prompt + all answers)** to find the best context - so the final count of the contexts from this part is 4:
    - `all-MiniLM-L6-V2` with prompt,
    - `all-MiniLM-L6-V2` with prompt and all answers, 
    - `GTE-base` with prompt, 
    - `GTE-base` with prompt and all answers
- the second one is the TF-IDF retrieval part from the great kernel created by [@mbanaei](https://www.kaggle.com/mbanaei) with the only change: we have added `sublinear_tf=True` for 200k parsed paragraphs dataset, according to our validation this was the best option. So here we receive 2 more context variants - for 200k and 270k datasets. 

## Single models

**As for single models**, we have used 2 common architectures as a base for our fine tuned checkpoints:
- DeBERTaV3 Large
- LongFormer Large
We finetune our models using 60k [@cdeotte](https://www.kaggle.com/cdeotte) dataset and ours generated STEM 39k dataset. To enrich the variety of contexts for 39k custom dataset we generated refined contexts using GTE-base index both using prompt only and prompt with all answers. For finetuning we train `ckpt-3850`, `ckpt-7200` and `ckpt-8750` for DeBERTa architecture, for each step we relabeled least confident samples, where a finetuned model put the correct answer to the last place. For LongFormer we took only the last checkpoint due to the inference time limitations. During inference we use 2300 character limited context for DeBERTa-s and no limit for LongFormer (we grid search over 2000, 2100, 2200, 2300, 2400 and 2500 locally on 500-samples dataset and 2300 limit appears to be optimal in terms of inference time and final metric MAP@3).

## What we have tried and it didn’t work:

- CutOut, CutMix augmentations
- separate models for each contexts
- reward training with and without hard negatives
- multiple negatives loss
- RoBERTa, ELECTRA, BERT finetuning
- averaging over answers order during retrieval: `mean(prompt + A + B + C + D + E, prompt + E + A + B + C + D , prompt D + E + A + B + C … )` performs exactly the same score on 500 samples, `mean(prompt + A, prompt + B, prompt + C, prompt + D, prompt + E)` got less in MAP@3
- t5, e5, GTE and BGE, but for [@mbanaei](https://www.kaggle.com/mbanaei) datasets, TF-IDF was the best option
- different index versions, the only meaningful parameter was nprobs
- TF-IDF ranker + BM25 as reranker.
- Training a single question+single answer pair classifier.

**We do regret about not probing the other embedders’ options for full wikipedia  index creation and using only sentences not chunks.**

## Compositions

**To create compositions** we have checked a lot of crazy variants including blending, stacking and backup strategy - all of them were tested and compared on the 500 rows publicly available (200 + 300 rows). The best variant we have found here is to blend 4 predictions from the same DeBERTaV3 checkpoint based on contexts from the first retrieval strategy to receive so called `blend_7200` and `blend_8500` along with average backup for predictions `DeBERTa_3850` using the LongFormer predictions. The final formula looks like: 0.58 * blend_7200 + 0.42 * blend_8500 + TF-IDF_backups
Important notes we found about blending:
- Blending predictions for more than 4 models at a time on the 500 rows validation dataset decrease the MAP@3 score on both public and private LB
- Blending predictions compositions from different retrieval strategies also made the things worse in comparison with simple averaging
- Backup strategy for TF-IDF retrieval works better if we replace `DeBERTa_3850` predictions with `max_probability < 0.514` using the blend `0.5 * DeBERTa_3850 + 0.5 * LongFormer` instead of LongFormer predictions itself.

## Submissions timing

As our notebooks can run only for 9 hours max, we **need to monitor how long each submission was**. This can help us to find out how many more models we can add to our solution and also helps to find out in what part the OOM/exception was. To check the submission time we have created a specific script which was run in tmux session with `while True` loop inside to check the submission status every 1 minute. So we know that **our best submission is 08:52:43 out of 9 hours timeout** - pretty awesome usage of the time, right? 🙂 

As this is not the last kernel competition, **I would like to share my script [here](https://www.kaggle.com/code/alexryzhkov/submissions-timer-script/notebook)**, which you can use in the further competitions on your side. You can also change the `GMT_OFFSET` variable to make it more suitable with your timezone.

## Another tricks

We also have tried to use **pseudolabeling techniques** to make our predictions look more similar for the test data distribution. Basically adding the 4k test samples on the top of the 500 OOF dataset and run ML models didn’t score better than simple weighted average approach.

## Conclusion

Thanks Kaggle and other participants for such a great competition. For me even after 12 years of kaggling this is still full of learning and fun.
