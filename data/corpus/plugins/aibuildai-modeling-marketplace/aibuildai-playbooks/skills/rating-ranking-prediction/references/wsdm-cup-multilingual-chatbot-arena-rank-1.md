# 1st Place Solution

Competition: wsdm-cup-multilingual-chatbot-arena
Rank: #1
Source: https://www.kaggle.com/c/wsdm-cup-multilingual-chatbot-arena/discussion/569902

First of all, thanks to the organizing teams, to my teammate Michael, as well as other contestants for making this challenge fun, see you all next time ˙ᵕ˙

### Overview

Our solution is a merge of distillation approach from @sayoulala and inference method from @tascj0 (the [1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629) and the [2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685) solutions from the last competition) using [Qwen2.5-14B-Instruct](https://qwenlm.github.io/blog/qwen2.5/) as a base model.

Our [inference code](https://www.kaggle.com/code/orangeaugust/lmsys2?scriptVersionId=220505454) and [training code](https://github.com/maxreciprocate/kaggle-lmarena-1st-place) are public.

### Setup

We use a default `ForSequenceClassification` model with two output classes for binary classification. For the input we concatenate the prompt and both responses, while proportionally truncating from the middle each in case the formatted sample exceeds the maximum length. For training we include both response orders in a single gradient step. 

### Stage 1. Pretraining

We pretrain both Qwen2.5-14B-Instruct (student) and Qwen2.5-72B-Instruct (teacher) on the following datasets:

- [mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) (40k)
- [opencsg/UltraFeedback-chinese](https://huggingface.co/datasets/opencsg/UltraFeedback-chinese) (50k)

For both models we do a full parameter finetuning using DeepSpeed ZeRO3. Learning rate schedule for all experiments has 50 warm-up steps followed by linear decay into 10% of the peak lr (4e-6). We increase gradient clipping threshold to 64.0 for both (as was noted in the [2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685) and the [3rd](https://github.com/rbiswasfc/lmsys-arena/blob/4007b8dfdb4c19899252962ff7f690e5f7038c86/conf/r_llama/conf_r_llama_m205.yaml#L56) solutions from the last time).

### Stage 2. Teacher training

We further train 5 teachers starting from the pretrained Qwen2.5-72b-Instruct on 5 folds of the following datasets:

- [lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) (57k)
- [dataset for this challenge](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/data) (48k)

After filtering out ties, multi-turn responses and deduplicating prompts and responses the combined dataset had 80k samples. In addition for training of teachers we change the input format by adding model names before each response.

### Stage 3. Distillation

Using five teacher models we soft label out of the fold the above datasets and manually review samples which had the highest disagreement with human annotations in order to correct mislabelled pairs. In addition we soft label the following data:

- Above lmarena datasets including ties and the first turns from multi-turn conversations (106k)
- [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) (33k)
- [lmarena-ai/PPE-Human-Preference-V1](https://huggingface.co/datasets/lmarena-ai/PPE-Human-Preference-V1) (16k) (despite it being an evaluation dataset and we initially suspected that this is the LB due to a close correlation, we still included it into training mix fearing others would)
- [lmarena-ai/Llama-3-70b-battles](https://huggingface.co/datasets/lmarena-ai/Llama-3-70b-battles) (1k)
- [lmarena-ai/gpt-4o-mini_battles](https://huggingface.co/spaces/lmarena-ai/gpt-4o-mini_battles/) (1k)
- Datasets from @nbroad 🤗 ([v1, v2 and v3](https://www.kaggle.com/datasets/nbroad/wsdm-open-models-nbroad)) (25k)
- [Our synthetic data](https://huggingface.co/datasets/reciprocate/kaggle-lmarena-synth-50k) seeded by prompts from [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (50k)

The last dataset was made by filtering easy prompts by labeling them with a complexity score (as done in [Arena-hard](https://lmsys.org/blog/2024-04-19-arena-hard/)) and taking prompts which have ≥ 3 complexity score, while preserving language distribution (50% non-English) as well as prompt domain distribution (using 8 categories from [MT-Bench](https://arxiv.org/abs/2306.05685)) mirroring the challenge's dataset. For the responses we uniformly sampled completions from the following models using vLLM:
- meta-llama/Llama-3.1-405B-Instruct-FP8
- meta-llama/Llama-3.3-70B-Instruct
- Qwen/Qwen2.5-72B-Instruct
- Nexusflow/Athene-V2-Chat

Distillation was done using three weighted losses (CE against the original label if present, otherwise a hard label, KL and Cosine loss on soft labels). The final model is a linear merge of two models: one trained on the full data and one excluding the last two datasets.

### Inference

For inference we use sequence concatenation (unpadding) by adapting code from the previous [2nd](https://www.kaggle.com/code/tascj0/lmsys-0805/notebook?scriptVersionId=191245382) solution for Qwen2.5. We do two inference passes: first using the merged model on all samples, then second time with response order swap using the original model (trained on complete data) only on 33% of samples with the most uncertain predictions.

### Unsuccessful experiments

- We trained a separate critique model to add CoT for training and inferencing teachers, by creating a dataset of 20k critiques from GPT-4o filtered by consistency with human votes. Yet training on soft labels made using critiques was repeatedly worse than without them.
- We tried using Qwen2.5-72B for inference with vLLM and AWQ for the second inference pass. However due to a higher inference cost, only 5% of samples could be processed with TTA, which ended up looking significantly worse on LB.
- We did experiments with layer pruning based on activation similarity, which would enable inferencing a much larger model. However, layer pruning gemma-2-27b or Qwen2.5-14B came out the same or worse than just using gemma-2-9b in our experiments.
- Also tried enforcing similar output embeddings before the linear head for the two response orders to unbias the model to the input order, which could remove the need for a second pass or let us use less resources for it.
