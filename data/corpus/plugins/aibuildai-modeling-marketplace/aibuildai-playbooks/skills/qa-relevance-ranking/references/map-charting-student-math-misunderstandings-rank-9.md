# 9th Place Solution

Competition: map-charting-student-math-misunderstandings
Rank: #9
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/9th-place-solution

Many thanks to the organizers and Kaggle staff, and congratulations to all the prize winners.
This competition taught me a lot — and I’m happy to have earned my first gold medal!

---

## Overview

* **Method**

  * Used 32B Causal LM with vLLM for inference: 5-fold × 2 models = 10 total.
* **Scores**

  * Public LB: **0.950**, Private LB: **0.948**
* **Models**

  * `Qwen/Qwen2.5-32B-Instruct`
  * `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
* **Inference time**

  * 460 minutes

---

## Dataset

* Competition train data 
  * Split into 5 folds with StratifiedKFold  based on QuestionId, Category, Misconception.

---

## Training

* **Reference implementation:**
  [https://www.kaggle.com/code/sinchir0/lb-0-942-train-fullft-qwen3-8b-by-sfttrainer](https://www.kaggle.com/code/sinchir0/lb-0-942-train-fullft-qwen3-8b-by-sfttrainer)
* The 65 labels were converted into special tokens.
* Used **SFTTrainer + AutoModelForCausalLM** with **QLoRA** for fine-tuning.
* **Hyperparameters**

  * **Qwen/Qwen2.5-32B-Instruct**

      * `epoch=2, lr=1e-4, per_device_batch_size=8, gradient_accumulation_steps=2`
      * `r=32, α=64, lora_dropout=0.01, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], modules_to_save=["lm_head"], trainable_token_indices={"embed_tokens": target_label_ids}`
  * **DeepSeek-R1-Distill-Qwen-32B**

      * `epoch=2, lr=8e-5, per_device_batch_size=8, gradient_accumulation_steps=2`
      * `r=64, α=128, lora_dropout=0.01, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], modules_to_save=["lm_head"], trainable_token_indices={"embed_tokens": target_label_ids}`
* **Environment**

  * Google Colab (A100 80GB)
* **Training time**

  * 220 minutes / 1 fold

---

## Preprocessing

* Fixed known annotation errors and unified duplicate Misconception labels.
* The unified labels were redistributed in post-processing based on the corresponding QuestionID.

```python
def wrong_corrections(df: pd.DataFrame) -> pd.DataFrame:
    false_to_true_ids = [12878, 12901, 13876, 14089, 14159, 14185]
    df["MC_Answer"] = np.where(
        df["row_id"].isin(false_to_true_ids),
        df["MC_Answer"].str.replace(r"\( 6 \)", r"\( 9 \)"),
        df["MC_Answer"]
    )

    true_to_false_ids = [14280, 14305, 14321, 14335, 14338, 14352, 14355, 14403, 14407, 14412, 14413, 14418]
    df["MC_Answer"] = np.where(
        df["row_id"].isin(true_to_false_ids),
        df["MC_Answer"].str.replace(r"\( 9 \)", r"\( 6 \)"),
        df["MC_Answer"]
    )
    return df

def replace_duplicate_misc(df: pd.DataFrame) -> pd.DataFrame:
    df["Misconception"] = df["Misconception"].replace({"Wrong_Fraction": "Wrong_fraction"})
    return df
```

---

## Prompt Optimization (for Accuracy Improvement)

* Simplifying the prompt slightly improved CV scores.
* It also helped reduce training and inference time.

```python
prompt_format = """Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}
Label: """
```

---

## Inference

* Used **vLLM + AWQ** for inference.
* Retrieved logits for the added special tokens, and averaged the scores (simple addition mean) across all models.

---

## Results (MAP@3)

| Model / Setting                                | Valid    | Public LB | Private LB |
| ---------------------------------------------- | -------- | --------- | ---------- |
| Qwen2.5-32B-Instruct (fold0)                   | 0.950182 | 0.946     | 0.944      |
| Qwen2.5-32B-Instruct (5-fold)         |          | 0.950     | 0.946      |
| DeepSeek-R1-Distill-Qwen-32B (fold0)           | 0.947956 | 0.948     | 0.945      |
| DeepSeek-R1-Distill-Qwen-32B (5-fold) |          | 0.950     | 0.946      |
| 5-fold × 2 ensemble                            |          | 0.950     | **0.948**      |

---

## Attempts That Didn’t Improve Results

* **Reranker (pairwise / pointwise)**
  Implemented for top1/top2 reordering, but performance did not surpass the base results.
* **70B / 72B models**
  CV was better than 32B, but inference time (~150 min) was too long for final pipeline.
* **Non-Qwen2.5 models**
  Tried Llama 3.3-70B, gpt-oss-20b, Mistral, Hermes, Gemma 3, Qwen 3, etc.
  None outperformed Qwen 2.5 models in our situation.
* **Soft labels**
  Re-trained Qwen-32B using logits produced by another Qwen-32B model trained on the same dataset.
  No improvement was observed on the public leaderboard, but the private leaderboard score increased slightly (+0.001).
* **LoRA Merge**
  Merging LoRA adapters trained on five folds resulted in lower performance compared to using a single model.
* **Training only on [Category: Misconception] and excluding True/False samples**
  Excluding True/False samples and training only on the Misconception category did not outperform the standard category classification model.

---

## References

https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

https://www.kaggle.com/code/sinchir0/lb-0-942-train-fullft-qwen3-8b-by-sfttrainer
https://huggingface.co/docs/peft/main/en/developer_guides/lora#efficiently-train-tokens-alongside-lora
