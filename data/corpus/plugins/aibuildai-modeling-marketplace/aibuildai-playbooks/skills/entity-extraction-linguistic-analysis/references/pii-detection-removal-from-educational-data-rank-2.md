# 2nd place solution

Competition: pii-detection-removal-from-educational-data
Rank: #2
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497352

Let me start by thanking Kaggle and the host of the competition for the efforts they put to run it. And congratulations to everyone who managed to succeed in it! Let me briefly describe my solution, which in the end was not very complicated.

**TL;DR:** a bag of deberta-v3-large models with pre/post processing


**1. Pre-processing**

The provided dataset was already split into substrings, primarily individual words and punctuation marks, but also whitespaces of different sizes. For the tokenization I used these lists of substrings instead of original texts (is_split_into_words=True HF option) and took averaged output probabilities when a word was represented by multiple tokens. It is important to highlight that whitespace substrings were ignored by the tokenizer, leaving predictions to be O by default.

Next step was removal of B- and I- prefixes from the targets to continue with only 7 classes to predict. The whitespaces and other words always separate the consequent PII items in the texts, so one can convert the targets and predictions back to BIO format precisely.

I also used the generated [dataset](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221) from @nbroad, a big shoutout to him for preparing and sharing it! I've mixed it together with the training data with a lower weight of 0.5.


**2. Models**

In the end only deberta-v3-large backbones were used as others performed worse on CV and LB. The final submission is a bag of 6 models, trained on the full data with slight variations of hyperparameters:
- max length 512, 1024 and 2048
- stride 32
- batch size 8
- learning rate 2e-5 over 4 epochs with cosine annealing scheduler, 1 epoch of warmup
- AdamW optimizer

Other tricks, like augmentations, though seemed promising on public LB and CV, didn't contribute to the top scored submission.

**3. Post-processing**

I used 2 post-processings. The first one is specific for class NAME_STUDENT. Once a substring was classified as NAME_STUDENT, all other occurrences  of this substring in the document were relabeled to NAME_STUDENT. However, there were cases when a single "." was predicted as NAME_STUDENT and then propagated through the entire text. To fix that, substrings of length 1 and those that are not title cased, were relabeled to O.

The second post-processing is about the "\n" whitespace substrings that occurred only twice in the entire dataset and both times were parts of STREET_ADDRESS. I believe it was shown that the PII data was artificially modified in the data, so my guess is that the generated addresses contained "\n", while accidentally the rest of the data never did. With the approach I took it caused the sequences of addresses predicted by the model to have holes, because "\n" was not tokenized and (as mentioned above) predicted as O by default. So simply force predicting "\n" as STREET_ADDRESS fixed 2 predictions per address (the "\n" itself and the following B-STREET_ADDRESS to get converted to I-STREET_ADDRESS).

Conversion of probabilities to hard predictions was done via argmax after scaling down class O probabilities. The scaling coefficient was tuned on public LB scores, but was consistently 0.02-0.03.

UPD: inference notebook is published now https://www.kaggle.com/code/dott1718/piid-2nd-place-solution
