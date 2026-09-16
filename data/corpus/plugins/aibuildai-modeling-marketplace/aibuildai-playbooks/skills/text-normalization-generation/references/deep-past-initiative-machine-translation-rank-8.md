# 8th Place Solution: 2-Stage Fine-tuning + High Quality Data Extraction

Competition: deep-past-initiative-machine-translation
Rank: #8
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/8th-place-solution-2-stage-fine-tuning-high-qua

First off, a bittersweet feeling wrapping this up. This was my first Kaggle competition, and landing a gold medal on the first attempt is something I am genuinely proud of. At the same time, three months of hard work and finishing just outside the cash prizes stings a bit. Regardless, congrats to all the winners and huge thanks to DPI and Kaggle for putting together such a unique and meaningful challenge.

## Model Selection

Larger the model, better the score. I experimented with NLLB and mT5 as well, but ultimately landed on ByT5-XL. The score difference between the models was in the range of 1 to 1.5 GM. ByT5's byte-level approach suits this task well since Akkadian script does not tokenize cleanly with subword tokenizers.

## 2-Stage Fine-tuning

This was the biggest single contributor to my score.

### Stage 1: Intermediate SFT on ~350k pairs

Noisy, unclean data consisting of Akkadian-English pairs, single-word dictionary entries, partial sentences and so on. The volume helped the model build a broad prior over the translation task.

Training config:
- 3 epochs
- Global batch size: 128 (bf16 , per device batch size 16 , gradient accumulation 4 , num devices 2)

### Stage 2: SFT on ~65k high-quality pairs

High-quality curated Akkadian transliteration-to-translation pairs. Trained for exactly 1 epoch in fp32. Anything more led to model collapse and LB score drops.
The combination of noisy broad pretraining in Stage 1 followed by clean targeted fine-tuning in Stage 2, paired with ByT5-XL, is what pushed me past 40 on the LB.


## Data Sources

The hosts recommended sticking to their provided sources rather than pulling from other Akkadian corpora available online, since those come from entirely different time periods and dialects. I mostly stuck to what the hosts gave. There were two main sources.

### 1. Publications

My entire Stage 1 intermediate SFT data came from the host-provided publications. At the very beginning of the competition I ran these through Mistral 14B, chosen purely from a multilingual capability standpoint since most of the translations in these publications were in English, French, German and Turkish and needed to be normalized to English. This gave me the ~3.5L pairs which mostly consisted of noisy pairs, single-word dictionary entries and such.

Later in the competition I re-engineered the same setup for GPT-OSS-120B in fp4 quantization with more sophisticated few-shot prompting, along with a sliding window approach which I will get to in the PDF section below.

### 2. PDFs

The PDFs had two forms of layout, one with a columnar layout and the other with a regular text layout. Initially I thought running OCR and passing it through the same pipeline as the publications would yield good quality data, but LB scores dropped when training further on this data. There was significant misalignment and multiple OCR errors across the board, so I engineered two separate extraction pipelines, one for each PDF layout type.

#### 2a) Regular Text Layout

The problem with the regular text layout was that the translations of the current page's transliterations were actually on the next page for at least 40% of the Akkadian documents.

**The Pipeline:**
* **OCR:** Performed using **GLM-OCR**.
* **Context Window:** A sliding window approach feeding (previous, current, next) pages into **GPT-OSS-120B**.
* **Prompting:** Few-shot prompting to guide the alignment.

**Note:** The sliding window was necessary to bridge page boundaries and correctly pair transliterations with translations that spilled onto the following page.
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  
  
</div>

#### 2b) Columnar Layout
For the columnar PDFs the page misalignment issue was not present, since the transliteration and translation pairs sat in adjacent columns on the same page rather than being split across pages. The problem here was OCR itself. It did not extract the columns properly and sentences from two separate columns were coming out merged into a single line.

I switched to a VLM-based approach for this. Initially Qwen3.5-VL 35B  yielded good results compared to other VLMs I tested, but I was not able to get high throughput with vision inputs, so I had to pivot to GLM4.7V-Flash which was the next best option for throughput. All the LLMs I chose across these extraction tasks were selected through vibe-testing rather than benchmarks, since I had no clean ground truth to evaluate against at that point.

Later when I managed to get Qwen3.5-VL 35B running at acceptable throughput with vLLM , I reran the VLM pipeline and retrained, but the results were actually worse on the LB, so I stayed with the GLM4.7V-Flash outputs.


## What Did Not Work for me

1. **Dictionary and RAG-based reconstruction** for pseudo-labeling samples I lacked ground truth for. This probably did not work out for me because I was using open source LLMs only.
2. **Name and commodity augmentation**, both online and offline variants, neither worked out.
3. **Sentence alignment** via automated methods. I could not find a method that consistently aligned sentences, so I tried doing it manually on the train set, but that did not yield good results either.
4. **Training for more than 1 epoch** in Stage 2 consistently led to overfitting and model collapse.
5. **LLM post-processing**, consistent with what others reported in public notebooks and discussions.
6. **RL-based methods**: I tried DPO, minimum risk training, PPO and GRPO adapted for Seq2Seq. None of them improved the LB score.
7. **Cross-validation**: I could not find a consistent correlation between my CV and LB throughout the competition. I eventually just took the first 500 pairs from the train set as my CV. My CV scores were consistently 42 to 43.5 across all my 40+ LB scoring models, which made it unreliable for model selection.


## Final Notes

100% open source stack throughout. No commercial models or APIs were used at any point. The full pipeline is replicable on a single A100 instance or a 2x RTX 6000 rig.


- Kudos to everyone!!
- Happy to answer questions in the comments.
