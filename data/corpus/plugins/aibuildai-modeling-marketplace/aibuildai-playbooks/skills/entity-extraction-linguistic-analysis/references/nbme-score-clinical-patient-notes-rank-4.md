# 4th place solution: Deberta models & postprocess

Competition: nbme-score-clinical-patient-notes
Rank: #4
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322799

First of all, thanks to competition organizers for hosting this competition and great teammates ( @takoihiraokazu, @shuheigoda, @copasta ).
Thanks also to the community for sharing many ideas in Notebook and Discussion.

# Summary
We ensembled 4 token classification models and set threshold for each case_num and did postprocess.


# Models
We trained following 4 token classification models and used them for final submission.
- token classification Deberta-v3-large 4fold
  - MLM(0.15)
  - finetune
    - SmoothFocalLoss
    - or replace augmentation for feature text
    - lower text
    - ~~3 times pseudo labeling~~
    - 2 times pseudo labeling (This is the one for final submission but 4 times pseudo labeling was better on Private LB)
    - Public LB: 0.891 with pp
    - Private LB: 0.892 with pp
- token classification Deberta-v3-large 5fold
  - MLM(0.15)
  - finetune
    - SmoothFocalLoss
    - mask augmentation
    - pseudo labeling
    - Public LB: 0.891 with pp
- token classification Deberta-v2-xlarge 5fold
  - MLM(0.15)
  - finetune
    - SmoothFocalLoss
    - mask augmentation
    - pseudo labeling
    - Public LB: 0.890 with pp
- token classification Deberta-v2-xxlarge 4fold
  - MLM(0.10)
  - finetune
    - SmoothFocalLoss
    - mask augmentation
    - pseudo labeling (Deberta-v2-xlarge)
    - Public LB: 0.892 with pp (5fold)

We also trained 1 char classification model and ensemble of 3 token classification models and 1 char classification model is the best private submission but couldn't select it for final submission.
- char classification Deberta-v3-large 4fold
  - MLM(0.15)
  - finetune
    - BCELoss
    - GRU Head(4 layer)
    - pseudo labeling
      - char classification Deberta-v3-large 4fold
      - char classification Deberta-xlarge 4fold
      - char classification Deberta-large 4fold
  - Public LB: 0.890 with pp

# Ensemble
Weighted Blending of 4 models based on char probabilities.

# Threshold
We set threshold for each case_num.

# Post Process
Here is our postprocessing code.
```
def postprocess(texts, preds):
    from nltk.tokenize import word_tokenize
    preds_pp = preds.copy()
    tk0 = tqdm(range(len(preds_pp)), total=len(preds_pp))
    for raw_idx in tk0:
        pred = preds[raw_idx]
        text = texts[raw_idx]
        if len(pred) != 0:
            # pp1: indexが1から始まる予測値は0から始まるように修正 ## +0.00123
            # if index starts with 1, replace it to 0
            if pred[0][0] == 1:
                preds_pp[raw_idx][0][0] = 0
            for p_index, pp in enumerate(pred):
                start, end = pred[p_index]
                if start == 0:
                    break
                # pp2: startとendが同じ予測値はstartを前に1ずらす ## +0.00012
                # if start == end, start should be start - 1
                if start == end:
                    preds_pp[raw_idx][p_index][0] = start - 1
                    break
                # pp3: 始点が改行の場合始点を1つ後ろにずらす ## +0.00032
                # if text[start] == '\n', start should be start + 1
                if text[start] == '\n':
                    preds_pp[raw_idx][p_index][0] = start + 1
                    start = start + 1
                # pp4: 1-2などは-2で予測されることがあるので修正 ## +0.00001
                # [digit]-[digit] sometimes predicted as -[digit], if so start should be start - 1
                if text[start-1].isdigit() and text[start] == '-' and text[start+1].isdigit():
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                if text[start-1].isdigit() and text[start] == '/' and text[start+1].isdigit():
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                # pp5: 67などは7で予測されることがあるので修正 ## +0.00001
                # [digit][digit] sometimes predicted as [digit], if so start should be start - 1
                if text[start-1].isdigit() and text[start].isdigit():
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                # pp6: 2文字前が記号のやつに対するpp ## +0.00001
                # To fix some letter sign tokenization problem
                if text[start-2] == ',' and text[start-1] != ' ':
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                if text[start-2] == '-' and text[start-1] != ' ':
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                if text[start-2] == '\"' and text[start-1] != ' ':
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                if text[start-2] == ':' and text[start-1] != ' ':
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                if text[start-2] == '(' and text[start-1] != ' ':
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                if text[start-2] == ')' and text[start-1] != ' ':
                    preds_pp[raw_idx][p_index][0] = start - 1
                    start = start - 1
                # pp7: heart -> h + eart となっているようなものを汎用的に修正する ## +0.00050
                # To fix some tokenization problem like heart -> h + eart
                try:
                    text_token = word_tokenize(text[start-1:end])
                    text_token = [text for text in text_token]
                    first = text[start:end].split()[0]
                    if first not in text_token:
                        for t in text_token:
                            if (first == t[-len(first):]) and (t[0].isalpha()):
                                sub = len(t) - len(first)
                                preds_pp[raw_idx][p_index][0] = start - sub
                                start = start - sub
                                break
                except:
                    None
                # pp8: endの修正 ## +0.00001
                # fix end
                if text[end-1:end] == '.':
                    preds_pp[raw_idx][p_index][1] = end - 1
                    end = end - 1
                if text[end-1:end] == '-' and text[end:end+1].isnumeric():
                    preds_pp[raw_idx][p_index][1] = end + 1
                    end = end + 1
    return preds_pp
```
