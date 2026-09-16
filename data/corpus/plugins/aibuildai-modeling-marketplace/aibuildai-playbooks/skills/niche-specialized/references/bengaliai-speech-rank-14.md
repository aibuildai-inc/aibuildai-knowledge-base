# 14th Place Solution for the Bengali.AI Speech Recognition Competition

Competition: bengaliai-speech
Rank: #14
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/447965

Thanks to the organizers and Kaggle staff for holding the competition, and congratulations to the winners!

# Context
Business context: https://www.kaggle.com/competitions/bengaliai-speech/overview
Data context: https://www.kaggle.com/competitions/bengaliai-speech/data

# Overview of the Approach

My approach addressed two main challenges:

**Challenge:**

1. The need for robust speech recognition capable of handling diverse speakers.
2. The requirement to restore punctuation in transcriptions.

**Approach:**

1. Fine-tuning the `indicwav2vec_v1_bengali` model using competition data.
2. Leveraging the Punctuation Restoration tool from https://github.com/xashru/punctuation-restoration.

This approach led to a Public Leaderboard score of 0.38.

I began with the foundation provided by @ttahara’s [notebook](https://www.kaggle.com/code/ttahara/bengali-sr-public-wav2vec2-0-w-lm-baseline).

# Details of the submission

**Diverse Speaker Recognition:**
The test audio data comes mostly from YouTube, which means that the speakers' identities are often unknown. To create a versatile model, I trained it on diverse audio data.


(dataset paper: https://arxiv.org/abs/2305.09688 C.1.1. Data Scraping Roadmap & Prerequisites)

**Punctuation Restoration:**
It's known that the labels are normalized, which implies that punctuation is preserved, as mentioned [here](https://www.kaggle.com/competitions/bengaliai-speech/discussion/432305#2400110).
Predicting punctuation during transcription is difficult, and leaving it out would result in word errors.

By restoring punctuation, Word Error Rate (WER) can be reduced:

- (label) hello. how are you?
- (predict) hello how are you
→ wer: 0.5
- (restore) hello. how are you.
→ wer: 0.25

While the public notebook appends periods at the end of sentences, the test data has an average of 34.42 words per sample and a Macro Train/Validation set with averages of 8.42/9.21. This suggests that multiple sentences may exist in one audio file, making it necessary to restore punctuation at points other than sentence endings.


(dataset paper: https://arxiv.org/abs/2305.09688 Table 1: OOD-Speech Dataset Statistics)

## Models

1. Wav2vec2CTC model
    - Fine-tuned Wav2vec2 `ai4bharat/indicwav2vec_v1_bengali`
2. Language model
    - KenLM `arijitx/wav2vec2-xls-r-300m-bengali`
3. Punctuation Restore model
    - XLM-RoBERTa-large from https://github.com/xashru/punctuation-restoration

## Training
[Github](https://github.com/Neilsaw/kaggle_Bengali.AI_ASR_16th_solution)

Fine-tuning Wav2vec2CTC with Transformers involved choosing datasets based on Yellowking’s WER, CER, and MOS_PRED metrics, as outlined in this [notebook](https://www.kaggle.com/code/imtiazprio/listen-to-training-samples-data-quality-eda). Two dataset splits were used:

1. **Easy Data:** audio samples where inference was straightforward.
    - YKG WER < 0.1
2. **Hard Data:** audio samples where the character content was correct but the WER was high.
    - 0.3 < WER < 1.5
    - CER < 0.15
    - MOS_PRED > 3

The first dataset helped the model adapt to a variety of voices, while the second dataset allowed it to handle audio with higher WER.

Additionally, the inclusion of white noise during training led to a slight improvement in the Leaderboard score by 0.001.

## Validate
I only used LB for Validate.
I couldn't rely on local cross-validation because the domain shift between the training data and the test data was too significant.

## Inference

[inference notebook](https://www.kaggle.com/code/neilus/16th-solution/notebook)

improved things from public notebook

- Punctuation restoration (-0.023)
  - for using [xashru/punctuation-restoration](https://github.com/xashru/punctuation-restoration), we need to change transformers==2.11.0.
  - so after LM inference, pip install transformers==2.11.0 and execute punctuation-restoration  on command line for reset import packages.

- using unigrams.txt for KenLM ( -0.005)

```jsx
with open(LM_PATH / "unigrams.txt", encoding="utf-8") as f:
    unigram_list = [t.lower() for t in f.read().strip().split("\n")]

decoder = pyctcdecode.build_ctcdecoder(
    list(sorted_vocab_dict.keys()),
    str(LM_PATH / "5gram.bin"),
    unigram_list,
)
```

- beam width 1500 (-0.002 ~ -0.001)


## Results

|  | Public| Improvement ||Private|
| --- | --- | --- | --- | --- |
| Baseline | 0.471 |  ||0.564|
| Training (Easy Data) | 0.425 | -0.046 ||0.508|
| Beam width 1500 | 0.423 | -0.002 ||0.506|
| Punctuation restoration | 0.400 | -0.023 ||0.488|
| Using unigrams.txt | 0.395 | -0.005 ||0.48|
| Training (Hard Data) | 0.380 | -0.015 ||0.458|

## Things didn't work for me 
- NER (Named Entity Recognition)
  - For restore "-" to NER, but using NER for ASR output sentences occur a lot of  False detection and LB down.
- create own LM
  - since i only used MaCro Train data, may be too small sentence.
- fine tuned punctuation model
  - same reason LM. only used MaCro Train data.
- denoiser (https://github.com/facebookresearch/denoiser)
  - LB and CV down. so I didn`t use.
- add various noise while training
  - only White noise was work.

# Conclusion

At the beginning of the competition, I tried adding noise to adapt to different domains, but it didn't improve the LB. From this, I thought that there were other challenges to address besides noise.

This competition challenged participants to achieve generalization and deal with label noise (punctuation). It required addressing the question of how much generalization is necessary and how to handle punctuation noise effectively. 

I am grateful for the opportunity to learn from this competition.

# Source
- https://www.kaggle.com/code/ttahara/bengali-sr-public-wav2vec2-0-w-lm-baseline
- https://www.kaggle.com/competitions/bengaliai-speech/discussion/432305#2400110
- https://github.com/xashru/punctuation-restoration
