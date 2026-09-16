# [DPC 1st] Data Quality Dictates Everything

Competition: deep-past-initiative-machine-translation
Rank: #1
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/dpc-1st-data-quality-dictates-everything

# Acknowledgments

Thanks to the hosts and Kaggle staffs, especially Adam Anderson. In the early stages of the competition, there were significant data issues that caused a lot of confusion, but Anderson worked diligently to resolve the most critical problems, which greatly reduced the difficulty of optimizing the models.

Thanks to my teammates mutian, zhengru, and yueting. Mutian and I have participated in many competitions on other platforms, and finally, in the year before starting our master's degrees, we had the time to compete on Kaggle, which is a much tougher challenge.

Datatech Club is a student club founded by mutian at his university (ShanghaiTech University), and zhengru and yueting are both members. The club received computing resource support from the School of Information Science and Technology at ShanghaiTech University, and we are extremely grateful for their strong support.

# TL;DR

The best submission (which we did not select) had a public score of 41.5 and a private score of 43.2. You can find the code and models in the end of this article.



The best submission is an ensemble of 11 `byt5-xl` models. Each model generates 10 candidate translations using beam search and sampling at different temperatures, and finally, the best translation is selected via a community-contributed MBR selector.

These 11 `byt5` models were trained on different combinations of datasets. 10 of them used the basic `Seq2SeqTrainer`, `bs=64`, trained for 3 epochs; 1 model used a custom loss weighted by data quality, `bs=48`, trained for 4 epochs.

Single-model inference for these models can easily reach the gold medal zone, and the best ckpt could have directly won 1st place as a single model. Below are some single-model inference submission scores (which do not fully comprise the final models used).



The `byt5-xl` models were pre-compiled and quantized to "int8-float32"—this feature is implemented by the `ctranslate2` library. The script performed data-parallel inference on 2*T5 GPUs, and we almost used up the entire 9-hour time limit.

My thoughts on pre-processing and post-processing: Pre-process the training data as carefully as possible, and post-process the model output as little as possible. This is because we can always ensure the transliteration format is uniform, but relying on the Public LB for post-processing brings the risk of overfitting and shake-up.

One of our selected submissions added a cleaning step for repeated n-grams (n=2-7) on top of the best submission. Its public score was 41.6, and private score was 42.8. (I always believed post-processing only increases overfitting risk, so I didn't conduct any post-processing experiments. But a day before the deadline, I had no other ways to continue improving the score, so I ultimately chose this submission which gave me a higher public LB rank. The fact that it wasn't the best submission exactly proves my judgment was correct 😀)



Another selected submission did not use `ct2` or quantization. It was a conservative ensemble of the top 4 ckpts on the LB. Its public score was 41.2, private score was 42.4.



# Data is the Most Important

## 1. OCR Data: Extracting Transliteration-Translation from Books

We extracted training data from PDFs provided by the hosts and collected by ourselves (we completely discarded `train.csv`).

For well-structured books (with fixed chapter formats and clear transliteration-translation alignments), such as AKT series, ICK_4, Larsen_2002, Michel_2020, POAT, we extracted data in batches using a structured approach.

- Step 1: Using GLM-OCR for layout_analysis, supplemented by regex for chapter names, we extracted a `chapter_dict` (tablet/chapter name -> page range) for each book. Using the tablet rather than the page as the basic unit preserves information more completely.
- Step 2: Utilizing the multimodal capabilities of Gemini-3.0-pro-preview, we sent the prompt, tablet name, and page images of the tablet to Gemini to extract sentence-level transliteration-translation pairs. For German/Turkish translations, we further asked Gemini to generate English translations.

For poorly-structured books, especially some short open-access texts scattered across the internet, we used a frontend created via vibe coding for extraction. The frontend allowed us to manually take screenshots, paste them into the window, select pre-set prompts, create queries, and call the API in batches.



Our prompts were written in Chinese, and here are some key points translated in English by LLM:

1) Segmenting by Logical Meaning

```text
1. Strip away "physical line breaks" and pursue "logical sentence meaning"
- No punctuation in ancient times: Akkadian transliterations on tablets strictly follow the physical lines of the clay tablet (words at the end of a line are broken directly, e.g., A-šùr-ta-ak-lá-/ku).
- Use English translation as an anchor: English translations by modern scholars have already sorted out the logical syntax and added modern punctuation. Therefore, when segmenting sentences, you must never map them line-by-line. Instead, you must first understand the complete, long English sentence, and then look for the corresponding start and end points in the transliteration.
```

2) Correctly Understanding the Tablet

```text
2. Beware of cross-page and cross-section continuations caused by "3D writing"
- A clay tablet is a three-dimensional writing medium. After filling the obverse, scribes would continue to the lower edge (l.e.), then flip to the reverse (rev.), and finally might write on the upper edge (u.e.) and left edge (l.e.).
- Difficulty: A complete clause often spans across these areas. During alignment, if you find the second half of a sentence is missing, you must immediately look for the connecting point in the following lines.
```

3) Understanding Grammar

```text
3. Adapt to the "spatiotemporal dislocation" of grammar (SOV vs SVO)
- Akkadian is SOV (Subject-Object-Verb): Verbs are usually placed at the very end of the sentence (e.g., I - silver - gave).
- English is SVO (Subject-Verb-Object): When translating, the predicate verb is moved forward. This means that a verb at the beginning of an English sentence might only be found in the last few lines of that large transliteration paragraph on the left. Sentence segmentation requires a "global view" rather than fixating on individual words.
```

4) Locating Special Vocabulary

```text
4. Make good use of "anchor" words for precise positioning
- In the dense transliteration, several types of words make excellent "anchor points":
- Sumerian logograms (remember to write them in uppercase letters in your returned results): e.g., KÙ.BABBAR (silver), AN.NA (tin), TÚG (textile/garment), GÚ (talent), GÍN (shekel).
- Numbers: Ancient merchants' accounts were extremely precise. Finding numbers (e.g., 1/2, 10, 15) is the fastest way to match bilingual texts.
- Personal and Divine Names: e.g., A-šùr (Ashur), Šu-Nu-nu (Shu-Nunu). These proper nouns usually have highly consistent spellings in both the transliteration and translation.
```

## 2. LLM Labeled Data: Generating Translations for Untranslated Transliterations using LLMs

The official OARE website provides a great database containing a massive amount of untranslated data. Based on tablet names and multi-sequence matching, we excluded the data that had already been OCR'd. Combining dictionary lookups and similar transliteration-translation example queries, we called Gemini to generate translations.

## 3. LLM Synthetic Data: Generating Completely Fictional Transliterations and Translations using LLMs
    
1) Dictionary vocabulary synthetic data (synth1). This module first conducted a coverage gap analysis between the dictionary and the training set to filter out words truly missing from the training set. Then, it used the LLM to generate transliteration-translation pairs containing these words. The LLM input included 100 example sentences and 5 dictionary words, outputting 5-10 data pairs. Personal names and words lacking translations were excluded from the dictionary words.
    
2) Free synthetic data (synth2). Based on 100 randomly selected sentences from the training set, it generated 3-7 sentence pairs.

## Dataset usage for model training

Next, we introduce the datasets used for training:

**data1 (OCR data)**: Completed on March 8, containing 29,908 sentence-level data pairs (4,472 tablets). All sourced from LLM-extracted book data, without any manual cleaning.

<p align="center">
  
  &nbsp;
  
</p>

**data2 (OCR data)**: Completed on March 14, containing 30,931 sentence-level data pairs (4,476 tablets). Based on data1, we trained models using 4-fold CV, and each fold's model inferred on its respective training fold. Since the training set should be well-fitted, if the inferred translation was far from the training translation, the sample was considered an error/hard sample. Filtering by a cross-fold average `geo_metric < 20`, we identified 740 tablets/chapters containing such sentence-level samples. We re-extracted these tablets using the LLM and supplemented with manual verification.

<p align="center">
  
  &nbsp;
  
</p>

**data3 (OCR data)**: Completed on March 19, containing 34,146 sentence-level data pairs (4,837 tablets). We further supplemented some extraction results on top of data2. Additionally, transliteration-translation length mismatches and overly long transliterations can cause model misunderstandings. Filtering by a character length difference > 100 and transliteration character length > 500, we identified 131 tablets containing such samples. We re-extracted these tablets using the LLM and supplemented with manual verification.

<p align="center">
  
  &nbsp;
  
</p>

**llmlabel (LLM Labeled Data)**: Completed on March 20, containing 21,759 sentence-level data pairs (3,946 tablets). Training with only this dataset yielded a public score of 35.0 and a private score of 37.1.

<p align="center">
  
  &nbsp;
  
</p>

**synth1 (LLM Synthetic Data)**: Completed on March 8, containing 9,685 sentence-level data pairs.

<p align="center">
  
  &nbsp;
  
</p>

**synth2 (LLM Synthetic Data)**: Completed on March 20, containing 4,972 sentence-level data pairs.

<p align="center">
  
  &nbsp;
  
</p>

The final ensemble of 11 models is basically a combination of the data above:

| model name | data1 | data2 | data3 | llmlabel | synth1 | synth2 | all/split | note | Public Score | Private Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  --- | --- |
| A | ✅ |  |  |  |  |  | all |  | 39.6 | 40.4 |
| B | ✅ |  |  | ✅ |  |  | all |  | 40.1 | 41.2 |
| C |  | ✅ |  |  |  |  | all |  | 39.7 | 40.9 |
| D |  | ✅ |  | ✅ |  |  | all |  | 40.3 | 41.3 |
| E |  |  | ✅ |  |  |  | all |  | 40.1 | 40.6 |
| F |  |  | ✅ | ✅ |  |  | all |  | 40.1 | 41.6 |
| G |  | ✅ |  | ✅ | ✅ | ✅ | all |  | 40.1 | 41.3 |
| H |  |  | ✅ | ✅ | ✅ | ✅ | all |  | 40.0 | 40.6 |
| I |  | ✅ |  |  |  |  | split | 9:1 train valid split | 40.0 | 41.2 |
| J |  |  | ✅ |  |  |  | split |  | 40.4 | 41.1 |
| K |  | ✅ |  |  |  |  | split | weighted loss by data quality | 39.1 | 40.0 |

*Note: these socres are tested on ct2 int8_float32 model*

For comparision, some float16 transformers models results are as follows:

| model name | data1 | data2 | data3 | llmlabel | synth1 | synth2 | all/split | note | Public Score | Private Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  --- | --- |
| byt5-xl-0317-ocr_akka2en/1308 |  | ✅ |  |  |  |  | split |  | 40.5 | 41.0 |
| byt5-xl-0319-ocr_akka2en/1563 |  |  | ✅ |  |  |  | all |  | 40.1 | 40.9 |
| byt5-xl-0321-data2_llmlabel/split/2238 |  | ✅ |  | ✅ |  |  | split |  | 40.5 | 41.4 |
| byt5-xl-0321-data2_llmlabel_synth2/all/2703 |  | ✅ |  | ✅ |  | ✅ | all |  | 40.4 | 42.0 |
| byt5-xl-0322-data3_llmlabel/split/2364 |  |  | ✅ | ✅ |  |  | split |  | 40.7 | 41.0 |

# Regarding Model Training and Inference

1. Beware of the model over-memorizing specific formats

    Use `eval_loss` for evaluation: Since there is inevitably a distribution and style inconsistency between the local validation set and the hidden test set, continuous improvements in `eval_bleu`/`chrf` after 3 epochs are untrustworthy. Therefore, we selected the 3-epoch checkpoints based on the validation set's `eval_loss` rather than `eval_bleu`/`chrf`.

    

2. Larger models are better

    About 10 days before the competition ended, we started switching from `byt5-base` to `byt5-large`, and then further to `byt5-xl`. The score increase on the public LB was significant.

    However, the prerequisite for this conclusion might be: Having sufficient and clean training data, so that larger models do not fall into overfitting.

3. MBR

    Referring to [open-source MBR strategies](https://www.kaggle.com/code/ngyzly/lb-35-5-better-candidate-diversity-on-public-model), during the inference stage, beam search (`beam_size=8`) selected 4 candidates, and sampling at three temperatures `[0.60, 0.80, 1.05]` yielded 2 candidates each. After unified post-processing, we combined chrF++, BLEU, Jaccard, and length rewards to use weighted MBR to select the final translation that had the highest overall consistency with other candidates.

4. Some necessary inference details
    - Sorted `test_df` in descending order by `char_length`.
    - More quantized models are better than fewer full-precision models.
    - Pre-compiling `ct2` models saved 30 mins; otherwise, an 11-model ensemble would have timed out.
    - Fully utilized 4 CPU cores, performing MBR in parallel, which also saved time.

5. Things we tried but eventually abandoned (also listing related community writeups to compensate for our insufficient experiments)
    - Using decoder-only models, such as Qwen, translate-gemma. [This 25th solution](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/25th-post-training-qwen2-5-32b-and-72b-with-gemi) provided proof that fine-tuning Qwen can achieve great results.
    - Using CPT (Continual Pre-Training). I found the quality of the pre-training corpus to be quite poor and hard to clean. [This 10th solution](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/10th-place-solution-seq2seq-cpt-pseudo-label) and [23th solution](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/23rd-place-cpt-grpo-no-sft) illustrated feasible paths for CPT.
    - Multilingual training, distinguishing akka->en/de/tr using prefixes.
    - Using context. [This solution](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684270) mentioned context could be an effective approach.
    - Test Time Augmentation (TTA), using `byt5` outputs on the test set as data for further fine-tuning. [This solution](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/llm-online-learning-is-all-you-need) pointed out that heterogeneous models should be used for augmentation.
    - In pre-processing: Standardizing Sumerian spelling, standardizing determinative formats, stripping supplementary text enclosed in parentheses in translations.
    - In post-processing: Standardizing names and places according to `onomasticon.csv`, cleaning repeated n-grams.

### Computing Resources Used

In the early stages of training the `byt5-base` models, we used 1 x A100 for training.

In the later stages of training the `byt5-xl` models, we used 8 x H20 for training.
