# 11th place solution

Competition: bengaliai-speech
Rank: #11
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/447986

Hi, all the fellow Kagglers! 

I'd like to first give a massive gratitude to Kaggle, and Bengali AI on behalf of my team. Besides, I want to thank lots of fellow competitors for providing inspirational insights. Thanks to authors of reposLastly, thanks to my teammates, particularly @rkxuan for spotting the repo that forms the basis of the punctuation model. 

---

## Model Architecture:

- **ASR model:** Wav2vec2 CTC model
- **Ngram:** Kenlm
- **Punctuation Model:** xlm-roberta-large

## What Worked:

- Fine-tuning on @umongsain filtered Common Voice dataset. Public LB: `0.439 -> 0.428`
- Fine-tuning on Openslr 37 resulted in: Public LB `0.428 -> 0.414`
- Incorporating Oscar corpus into ngram
- Data augmentation with [audiomentations](https://github.com/asteroid-team/torch-audiomentations).
- xlm-roberta-large configuration from this [repo](https://github.com/xashru/punctuation-restoration).
- Optuna search for decoding hyperparameters, as demonstrated in this [notebook](https://www.kaggle.com/code/royalacecat/lb-0-442-the-best-decoding-parameters).

## Challenges:

### Datasets:

- **Openslr 53:** Voluminous and might've led to overfitting during training. It being crowdsourced could be a factor, especially when compared to the more refined Openslr 37.
- **Fleurs:** Plagued with inconsistent quality. A consistent sharp noise mars the dataset.
- **Competition ds:** Exhibits quality diversity.

### Punctuation Restoration:

- Difficulties in restoring five punctuations: **!**, **,**, **?**, **।**, and **-**.
  - Notably, hyphens appear to be tricky. Maybe isolating it for training could help.

### Audio Augmentation:

- Might've overdone with BGMs. Modulating pitch could potentially be more effective.

### Speech Enhancement:

- Both FAIR Denoiser and Nvidia CleanUnet fell short of expectations. It's perplexing, but perhaps they inadvertently degraded human voice quality.

## Pro-Tips:

- Prefer ARPA over BIN. Kenlm seems to lose unigram post-conversion, but this trick can amplify the score by `0.007`. Mind the 13GB RAM constraint on Kaggle. I eventually settled with a trimmed 4gram, approximately 12GB.

## Observations:

- Local CV, grounded in annotated OOD examples, aligns well with the public LB. This might explain the negligible shake-up in the end.
- Oscar primarily features formal **articles**. So, the ngram, despite its magnitude, might overlook the niche vocabularies in OOD test sets, especially with the high OOV observed.

---

Feel free to share your thoughts and experiences!
