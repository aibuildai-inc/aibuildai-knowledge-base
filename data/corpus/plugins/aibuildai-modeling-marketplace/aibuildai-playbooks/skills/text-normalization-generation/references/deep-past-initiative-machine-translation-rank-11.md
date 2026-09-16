# 11th Place Solution

Competition: deep-past-initiative-machine-translation
Rank: #11
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/11th-place-solution

First of all, I'd like to thank the Deep Past Challenge team at Kaggle for hosting this fascinating competition. I also found many publicly available notebooks and discussions invaluable throughout the process, and I came away having learned a great deal.

## Overview

1. I pre-train ByT5-large on multilingual translation pairs extracted from `publications.csv` to help the model understand Akkadian vocabularies and grammar
2. I fine-tune the model on `train.csv` to align it with the competition format and domain knowledge
3. I ensemble three separately fine-tuned ByT5-large models by pooling their outputs and selecting the best candidate via MBR decoding

## 1. Pre-training

I found that `page_text` in `publications.csv` contains many translations of Akkadian transliterations into multiple languages, including English, German, and French. I used an LLM to extract these pairs from the raw text, converting the unstructured PDF-extracted content into a clean, structured JSON format. I allowed the LLM to output **null** for cases where no corresponding translation exists, which I believe improved the overall quality of the dataset.

```
### Output Format (Strict JSON)
Return ONLY a JSON array of objects with the following structure. Do not include markdown headers or extra text.
{
  "items": [
    {
      "akkadian_transliteration": "string",
      "translation": "string or null",
      "language": "string or null",
    }
  ]
}
```

The extracted data covers many languages (with some potential hallucinations from the LLM). The histogram below shows the language distribution in my dataset. I decided to use only English, French, German, and Turkish for pre-training, as these had sufficient data volume.

[histogram_of_languages]

Training uses `Seq2SeqTrainer` and takes approximately 20 hours on a single A100 GPU in Colab.
I preprocessed the data following the [dataset instructions](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview) and [valuable discussions](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/668402).

Hyperparameters:
| Parameter | Value |
| ---------------------- | :---: |
| epochs | 3 |
| batch size | 8 |
| learning rate | 1e-4 |
| max length | 512 |
| label smoothing factor | 0.1 |
| EMA decay | 0.999 |

## 2. Fine-tuning

The training data contains translations aligned at the document level, whereas the test data is aligned at the sentence level. To bridge this gap, I used an LLM to segment the Akkadian transliterations and their English translations into sentence-by-sentence pairs.

Since I was unsure about the typical length of test examples, I analyzed `test.csv` and found that transliteration lengths range from 129 to 267 characters. Based on this, I applied a length constraint of 100–400 characters per segment. The key instructions I used are as follows:

```
Maintain Sequence: Do NOT change the original order of sentences in either language.
Length Constraints: Each Akkadian segment must be between 100 and 400 characters long.
   - If a segment is **too short** (< 100 chars): merge it with the adjacent clause.
   - If a segment is **too long** (> 400 chars): split it at a natural boundary such as a new clause, conjunction (ù, -ma), or topic shift.
```

I also noticed mismatching translation pairs in `train.csv`. The histogram below shows the distribution of translation-to-transliteration length ratios.



I replaced outliers with LLM-generated translations, using `0.66` and `2.0` as the lower and upper thresholds. I provided few-shot examples to guide the correction:

```
# Task
Improve and correct the following English translation of an Akkadian text, using the original translation as a reference base.

...

# Examples
Akkadian: 0.3333 ma-na 2.5 GÍN KÙ.BABBAR 20 NINDA i-ṣé-er tù-wa-ra-a-aḫ-šu a-lá-ḫu-um i-šu
Original: Ali-ahum has 0.3333 mina 2.5 shekels of silver and 20 loaves of bread from Tuwar-ahšu.
Corrected: Tuwar-ahšu owes 0.3333 mina 2.5 shekels of silver (and) 20 loaves of bread to Ali-ahum.

...

# Input
Akkadian: {ak_input}
Original: {original_translation}
Corrected:
```

I also generated translations for transliterations in `publication_texts.csv` using a similar prompt for use in pre-training.

Finally, I fine-tuned the pre-trained ByT5-large on the refined `train.csv` using the same training code as pre-training, with `epochs=2` and `lr=1e-5`.

## 3. Inference

I ensembled three ByT5-large models fine-tuned on different pre-training datasets. To evaluate the impact of MBR decoding, I ran experiments on the 100 hardest validation samples (selected by val score) to keep the process manageable. I compared four candidate generation methods: normal beam search, group beam search, nucleus sampling, and epsilon sampling. The best result came from using normal beam search for candidate generation and group beam search for the candidate pool. I used the competition metric directly as the utility function for MBR decoding. Applying a soft margin penalty based on the translation/transliteration ratio, along with PN/GN lemmatization, provided a small additional boost.

```python
def calculate_soft_margin_penalty(ratio, low=0.9, high=2.0, sigma=0.2):
    distance = max(0, low - ratio, ratio - high)
    return np.exp(-(distance**2) / (2 * (sigma**2)))
```

## Model scores and final submission

The val scores were not well-correlated with the public leaderboard scores, which was puzzling. I believe the main cause is that my validation set was constructed by segmenting `train.csv` with an LLM and replacing mismatched pairs with LLM-generated translations. This made the val set share the same distribution and style as the training data, inflating the val scores and causing them to diverge from the actual test set distribution. That said, I'm relieved the final submission turned out reasonably well.

D1: `published_texts.csv` D2: `publications.csv` D3: `train.csv`
| pre-training dataset | pre-training steps | val (w/o ft) | val | val hard | public | private |
| ----------------------- | :------------: | --------- | ------ | ------- | --- | --- |
| D1: `37910` D2: `79806` | 42000 | 36.2 | 41.7 | 23.8 | 39.0 | 38.9 |
| D1: `37910` D2: `126439` D3: `3987` | 38000 | 45.8 | 47.9 | 29.6 | 38.0 | 38.3 |
| D1: `37910` D2: `18834` | 30000 | 45.5 | 46.6 | 28.6 | 38.2 | 37.8 |
| (final submission) | - | - | - | 30.14 | 39.4 | 39.8 |

## Things that didn't work

- PN/GN post-processing and normalization using `OA_Lexicon_eBL.csv`
- Pre-training on dictionary data from `eBL_Dictionary.csv`
- Minimum Risk Training for calibrating the model with the competition metric
