# 12th Place Solution

Competition: wsdm-cup-multilingual-chatbot-arena
Rank: #12
Source: https://www.kaggle.com/c/wsdm-cup-multilingual-chatbot-arena/discussion/567614

Firstly, I would like to express my gratitude to the organizers of this competition and all participants who generously shared their great ideas. I thoroughly enjoyed participating and learned a lot during the process. It was an extremely valuable and educational experience.

## Training

### Model

#### Base Model

- I selected **Qwen2.5-14B-Instruct** as the base model.
  - It performed better than gemma2-9b-it-fp16 and DeepSeek-R1-Distill-Qwen-14B in my experiments.

#### Supervised Fine Tuning with QLoRA

- Hyperparameters:
  - learning rate: `2e-4`
  - lora\_r: `8`
  - num\_epoch: `1`
  - max\_length: `2048`

### Two-Stage Training

1. **1st Stage:** Pre-training was conducted using large external datasets that were presumed to effectively capture general human preferences.

   - Datasets used:
     - `dpo_44k (44K)` (\*1)
     - `ut_157k (157K)` (\*1)

2. **2nd Stage:** Fine-tuned on datasets similar to the WSDM competition data, starting from the checkpoint obtained during the first stage.

   - Datasets used:
     - `WSDM official (48K)`
     - `LMSYS official (39K)`
     - `add_23k (15K)` (\*1)

(\*1) External dataset: [https://www.kaggle.com/datasets/shelterw/wsdm-extra-dataset/data](https://www.kaggle.com/datasets/shelterw/wsdm-extra-dataset/data)

### Snip

- I inserted a placeholder `(snip 123 chars)` in the middle of prompts and responses to indicate that 123 characters were omitted ("snipped") from the original text. Including the number of omitted characters helped the model better handle partial information, thus improving accuracy.

### Response Swapping

- I also trained the model using data with responses A and B swapped.
- Although training time doubled, accuracy slightly improved.
- It was crucial to ensure that swapped samples appeared within the same batch for this approach to be effective.

### Cross-Validation and Full Training

- Evaluated using 5-fold cross-validation (only fold\_id=0 for quicker evaluation).
- The final model was retrained on the entire dataset without any evaluation splits, using the settings found optimal during cross-validation.

### Classification Head

- In addition to the primary binary classification task, an auxiliary task to classify the model name of responses A/B was added (weight = `0.1`).
- I also tested language ID classification but did not adopt it, as it provided no improvement.

## Inference

### First Round Inference

- Accelerated inference by sorting all samples in descending order of token count, then alternately assigning them to two GPUs.

### Second Round Inference

- Selected the top 40% of samples with prediction probabilities closest to 0.5 from the first inference round. For these samples, I averaged predictions with swapped responses A and B.
- This method improved both efficiency and accuracy compared to applying TTA across the entire dataset.

## Additional Notes

- I participated solo in this competition but discussed progress and strategy nightly with ChatGPT (o1).
- o1 served as an excellent sounding board for ideas. Additionally, it helped identify several bugs in my source code.

---

Thanks again to the organizers and fellow participants. Looking forward to seeing you in future competitions!
