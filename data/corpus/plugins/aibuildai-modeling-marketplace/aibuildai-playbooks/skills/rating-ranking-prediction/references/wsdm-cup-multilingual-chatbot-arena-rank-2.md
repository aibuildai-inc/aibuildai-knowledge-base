# The 2nd Place Solution

Competition: wsdm-cup-multilingual-chatbot-arena
Rank: #2
Source: https://www.kaggle.com/c/wsdm-cup-multilingual-chatbot-arena/discussion/567948

Thanks to Chatbot Arena and Kaggle for hosting such an amazing competition. Congratulations to all the winners.🎉🎉🎉

# Code Framework

Based on the work shared by @tascj0 in the previous competition, which offers highly efficient [training](https://github.com/tascj/kaggle-lmsys-chatbot-arena) and [inference](https://www.kaggle.com/code/tascj0/lmsys-0805/notebook?scriptVersionId=191245382), enabling fast full-parameter fine-tuning and inference under limited GPU resources. Kudos to @tascj0 's team for this excellent work.

# Data Processing

The `prompt`, `response_a`, and `response_b` are proportionally truncated in the middle based on length ratios. See `MIDV2ProcessorPAB` class in the inference code for more details.

# Base Models

The models use **gemma2-9b** and **ArmoRM-Llama3-8B-v0.1**, initialized with  @tascj0's trained models rather than training from scratch. The last layer of the classification head is discarded and changed to 2 (original was 3).

Named as `lmsys-gemma-pretrain` and `lmsys-llama-pretrain`.

# Final Submission 1

## Step1 Fine-Tuning Base Models

Fine-tuned `lmsys-gemma-pretrain`, `lmsys-llama-pretrain`, and `gemma2-27b` using WSDM competition data (80% training data + 20% validation set).

## Step2 Pseudo-Label Data Creation

Thanks to @nbroad for sharing three versions of datasets. We used [v1](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/552166) (8.5k) and [v2](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/556059) (13k). [v3](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/557746) was excluded due to minimal capability gaps between models, which could introduce noise during pseudo-labeling.

Averaged predictions from three models (soft labels) to create the pseudo-label dataset `hf-21k`.

## Step3 Fine-Tuning with Pseudo-Labels

Fine-tuned `lmsys-gemma-pretrain` and `lmsys-llama-pretrain` using `hf-21k` + WSDM (80%), resulting in `gemma-pseudo` and `llama-pseudo`.

## Step4 Online Inference

Used Test-Time Augmentation (TTA):

- `gemma-pseudo` predicts PAB
- `llama-pseudo` predicts PBA (swap)
  Results weighted at 3.3:1 ratio. Achieved LB score 0.709, private score 0.708.

# Final Submission 2

## Step1 Expanded Pseudo-Labels

### Data Sources

`vllm-74k` dataset. Prompts from `lmsys-chat-1m` after deduplication and removal of prompts used in Nicholas Broad's hf-v1/v2. Split by language:

- `en_424k` (English)
- `other_103k` (non-English)

Sampled datasets:

- `vllm-v1` (10k): 7k from `other_103k`, 3k from `en_424k`
- `vllm-v2` (27k): 21k from `other_103k`, 6k from `en_424k`
- `vllm-v3` (38k): 36k from `other_103k`, 2k from `en_424k`

Increasing non-English data ratio from v1 to v3 to improve multilingual performance.

### Response Pair Generation

Used vLLM inference with models:

'Qwen/Qwen2.5-72B-Instruct', 'Llama-3.3-70B-Instruct', 'gemma-2-27b-it', 'phi-4', 'DeepSeek-R1-Distill-Qwen-32B', 'Mistral-7B-Instruct-v0.3', 'internlm2_5-20b-chat', 'llama-3.2-3b-instruct', 'Hermes-3-Llama-3.1-8B', 'internlm3-8b-instruct', 'gemma-2-9b-it', 'DeepSeek-R1-Distill-Qwen-14B', 'Llama-3.2-1B-Instruct'

(DeepSeek-R1 models had "think" sections removed. vllm-v1 included some online models like DeepSeek-V3 and yi-lightning.)

### Pseudo-Label Generation

Used `gemma-pseudo` (PAB) and `llama-pseudo` (PBA) weighted 3.3:1. Removed data with `abs(winner_model_a - winner_model_b) <= 0.03`.

## Step2 Expanded Pretraining

Combined pseudo-labels: 95k (`hf-21k` + `vllm-74k`). Retrained `lmsys-gemma-pretrain` and `lmsys-llama-pretrain` to avoid pseudo-label noise, resulting in `gemma-pseudo-v2` and `llama-pseudo-v2`.

## Step3 Final Training

Fine-tuned `gemma-pseudo-v2` and `llama-pseudo-v2` using full WSDM training data (100%).

Split WSDM data:

- English subset (90% EN data)
- Multilingual subset (10% EN + other languages)

Trained for 2 epochs:
1st epoch on English subset, 2nd epoch on multilingual subset.
Resulting in `gemma-final` and `llama-final`.

## Step4 Inference

Same TTA as before:

- `gemma-final` predicts all data (PAB)
- `llama-final` only predicts data where `abs(winner_model_a - winner_model_b) < 0.8` (PBA)
  Weighted 2.5:1. Achieved same LB 0.709 but improved private score to 0.716.

# Other Attempts (Not Effective)

- Qwen2.5-72B distillation: Limited by time/GPU, only one fine-tuning cycle (lacking pretraining)
- Pseudo-labeling with Skywork-Reward-Gemma-2-27B-v0.2
- Replacing `lmsys-llama-pretrain` with Skywork-Reward-Llama-3.1-8B-v0.2
- Using Skywork-Reward-Preference-80K-v0.2 dataset
- BT loss, multi-loss weighting


**Final Inference code**：https://www.kaggle.com/code/zhudong1949/lmsys-0201
**training code**：https://github.com/zhudongwork/wsdm-cub-2nd-place-solution.git
