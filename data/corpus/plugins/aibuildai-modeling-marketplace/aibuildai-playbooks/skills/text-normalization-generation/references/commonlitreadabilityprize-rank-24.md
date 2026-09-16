# #24 Solution

Competition: commonlitreadabilityprize
Rank: #24
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257795

Firstly, thanks to the hosts and Kaggle for hosting this interesting competition. Also, great thanks to torch( @rhtsingh ) for share his/her experience, which helps me break 0.485 and could get better score.

# Summery
My final model consist of 2 kind of model and totally 17 models.
1. single model: **Input**: encoded row text. **Model**: roberta/deberta/bart last_hidden_states with mean pooling and attention head.
2. pair model: **Input**: baseline encoded text and encoded row text. **Model**: same architecture for baseline text and row text, then concatenate (or minus).

*As hosts say, every score is a compare of one text to baseline text.* [here](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236403)

# Fine-tune
My final models are fine-tuned with almost same fine-tune strategy. That is:

- Epochs: 5 for base mode and 3 for large model
- Batch Size: almost maximize the GPU memory. (4 - 24)
- Gradient Accumulate: 1-2
- Optimizer: AdamW + Lookahead
- LR Scheduler: None for base model and cosine_schedule_with_warmup for large model (warmup steps > 0)
- LLRD: use torch's shared settings and set base_lr=5e-5
- Random initial top layers: 1-2 for base model and 4-6 for large model
- Evaluating for every n steps: 10 to 60 steps

# Scoring
Just report my best score of the same base transformers.
| Model | Pair(1/0) |  CV  | Public | Private |
| --- | --- | --- | --- | --- | 
| Roberta-large | 0 |  0.47835 |   0.466 | 0.470 |
| Roberta-large | 1 | 0.46694  |   0.460 | 0.459 |
| Roberta-base | 1 |  0.47325 | 0.475  | 0.477 |
| Deberta-large | 0 | 0.46440  |  0.460 | 0.461 |
| Deberta-large | 1 |  0.46673 |  0.462 | 0.464 |
| Bart-large | 0 | 0.46184  | 0.462  | 0.464 |
| Bart-large | 1 |  0.46401 | 0.463  | 0.463 |
| Bart-base| 1 | 0.47103  | 0.470  | 0.472 |
| Xlnet-large-cased | 1 | 0.47427  |  0.475 | 0.465 |
| Electra-large | 1 | 0.47100  | 0.467  | 0.472 |
| xlm-roberta-large | 1 | 0.48622  |  0.477 | 0.478 |
| luke-large | 1 | 0.47148  | 0.471| 0.465 |
| funnel-large-base | 1 | 0.47117  |  0.469 | 0.468 |

Could not make other models work, and could not get a score below 0.5 even for fold 0.

# Ensemble
Simple `BayesianRidge` regression. Best score: CV is 0.447 and Public LB 0.451 Private LB 0.452.

Thanks!
