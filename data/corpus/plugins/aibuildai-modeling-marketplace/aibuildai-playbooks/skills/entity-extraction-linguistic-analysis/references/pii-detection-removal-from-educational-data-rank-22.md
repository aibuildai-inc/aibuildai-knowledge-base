# 22th place solution

Competition: pii-detection-removal-from-educational-data
Rank: #22
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497180

Hello Kagglers. 

First of all, we would like to thank the hosts for organizing such a great competition. Thanks a lot.

Also thanks to @takuji, @rheinmetall for the nightly discussions. Thanks a lot.

Shake Down is very sad, but I enjoyed this competition very much.

# Solution



Main point
 - 12 models ensemble
 - Folds are random for all models.

| backbone         | training dataset                | fold | CV     | public LB    |
|------------------|---------------------------------|------|--------|--------------|
| deberta-v3-large | train.json, pjmath, nbroad      | 0    | 0.971  | 0.964        |
| deberta-v3-large | train.json, pjmath, nbroad      | 1    | 0.953  | not submitted|
| deberta-v3-large | train.json, mpware(no-i-user)   | 0    | 0.963 | 0.941        |
| deberta-v3-large | train.json, mpware(no-i-user)   | 1    | 0.944      | not submitted|
| deberta-v3-large | train.json, mpware(no-i-user)   | 2    | ?      | not submitted|
| deberta-v3-large | train.json, mpware(no-i-user)   | 3    | ?      | not submitted|
| deberta-v3-large | train.json, mpware              | 0    | 0.961  | 0.953        |
| deberta-v3-large | train.json, mpware              | 1    | 0.970      | not submitted|
| deberta-v3-large | train.json, tonya               | 0    | 0.879(maybe bug)      | 0.959        |
| deberta-v3-large | train.json, tonya               | 1    | 0.941      | not submitted|
| deberta-v3-large | train.json, nbroad              | 0    | 0.949  | not submitted|
| deberta-v3-large | train.json, nbroad              | 1    | 0.961  | not submitted|


Dataset Citation:
pjmath: https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt
nbroad: https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated
mpware, mpware (no-i-user): https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays
tonya: https://www.kaggle.com/datasets/tonyarobertson/mixtral-original-prompt

## How to choose validation
- All texts were classified according to whether each label appeared or not(pattern), and all data were stratified split by pattern.
- train : valid = 0.5 : 0.5
  - By setting the ratio of train to valid to 50/50, CV and LB were correlated to some extent.

## Successes
- If the probability of "O" was less than 0.60, the label with the next highest probability of "O" was output.
- "\n" was added to Tokenizer.
 - CV: +0.030, LB: +0.009
 - In train.json, the token "\n" is all "I-STREET_ADDRESS"!
- Deleted "O" only data from train.json.
 - CV: -0.030, LB: +0.006
- The tokenizer truncation was set to False.
 - CV: +0.020, LB: +0.008
- The folds of each model were randomized.
 - The fold was not fixed and the split was changed each time the model was created.
 - LB: +0.008 (Difference from when fold was aligned across models)
- Post processing for BIO Prefixes
 - O, I-[PII]  ->  O, B-[PII]
 - B-[PII], B-[PII]  ->  B-[PII], I-[PII]
 - I-[PII], B-[PII]  ->  I-[PII], I-[PII]
 - LB: +0.005
- Post processing for impossible prediction pattern
 - e.g
     - if predicted NAME STUDENT is Non-camel case, change "O"
     - if predicted ID-NUM contains special character, change "O"( `r'[!@#$&%?^+=*<>]'` )
     - After much error analysis, we found above pattern
- Models ensemble
   - LB: +0.005 (1 model -> 3 models)
   - LB: +0.004 (4 models -> 8 models)
- ONNX
   - Maximum number of models that can be submitted: 10 models -> 12 models

## Didn’t work
- Correct label order as TokenClassification
 - e.g
     - B-[PII], B-[PII], I-[PII]  ->  B-[PII], I-[PII], I-[PII]
- Generate a dataset with additional non-student names and add them to the training
- Tuning thresholds for each PII type
- Second Stage by LightGBM
  - Use predictions from DeBERTa models as features
  - Convert probability from DeBERTa Token to Spacy Token
- Pseudo Labeling for test.json
