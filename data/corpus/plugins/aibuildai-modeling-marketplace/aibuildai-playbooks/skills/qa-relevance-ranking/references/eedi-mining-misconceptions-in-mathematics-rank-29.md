# 30th place solution - just public notebook(Public 35th)

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #29
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551409

Thank you to the excellent public notebook, from which I've learned a lot.

I made simple modifications to the public notebook and achieved 30th place.

## Recall stage:
Reference: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519

I fine-tuned qwen2.5-14b use [[FlagEmbedding]](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder/decoder_only), using the unfine-tuned SFR-Embedding-2_R as the seed model, obtaining 100 negative sample instances, with epoch=20.
Inference stage, I used load_in_8bit.

## Rerank stage:
I actually did not run this part of the code.

## logits_processor_zoo + logit_score_rerank
Reference: 
- https://www.kaggle.com/code/aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo
- https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550223

I fine-tuned qwen-72B with accelerate+qlora [[use LLaMA-Factory]](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/extras/fsdp_qlora/train.sh) , and then quantified it with GPTQ, to better adapt it to the 9-out-of-1 task. The fine-tuning brought about an improvement in performance.

Unlike the public notebook which uses logits_processor_zoo to get the candidate with the highest probability, I added the logprobs parameter in vllm.SamplingParams to obtain the logits for each option from 1 to 9, and then performed intra-group sorting, which also enhanced the results.

The complete inference code is as follows:
https://www.kaggle.com/code/zhudong1949/recall-reranker-logit-72b

TIPS: Synthetic data improved my recall score, but the overall score did not increase, so I ultimately did not submit the version with synthetic data. This should be due to the insufficient quality of my synthetic data.
