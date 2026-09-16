# 8th place solution

Competition: nbme-score-clinical-patient-notes
Rank: #8
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322962

First of all, I would like to thank the organizers for hosting this competition, and our awesome teammates @goldenlock @syzong @leolu1998 for their continuous efforts in the last two months. (BTW, we are the champion in the number of submissions, lol)

We ended up in 8th place, with scores of 0.893 for both Private LB and Public LB, which is relatively stable. But we moved up a lot in private LB, from 16th place in Public LB to 8th place in Private LB.

### Training and Inference

We mainly refer to the notebook:
https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train
https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-inference

Big thanks for @yasufuminakama

### ITPT

The pn_history data in the patient_notes.csv is utilized to In-Task Pre-Training based on deberta-v3-large and deberta-xlarge. The data is processed to remove the line breaks, since the pre-training data is based on each line as a sentence. The code is as follows:

```
def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    return txt
```

Parameters:

```
mlm_probability=0.15
num_train_epochs=30
per_device_train_batch_size=4
per_device_eval_batch_size=8
learning_rate=1.5e-5 #（xlarge)
learning_rate=3e-5 #（v3-large）
gradient_accumulation_steps=8
```

For the comparison test of the mlm_probability, we tried 0.1, 0.2 and 0.5, and none of them worked as well as the default parameter (0.15). The improvement from ITPT is obvious, from 0.883 to 0.886 on a single model (5 folds).

### Adversarial Training (FGM)

CV is improved about 0.0005 after adding FGM at fine-tunning based on deberta-v3-large, but it didn’t work on deberta-xlarge.

### Post-processing
In the process of analyzing the data, we found a number of cases with unrecognized first letter of the word, as shown in the figure below:



```
def get_results_pp2(char_probs, th=0.5, texts=None):
    results = []
    for idx, char_prob in enumerate(char_probs):
        text = texts[idx]                            # 对照文本
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        temp = []
        for r in result:
            start = min(r)
            end = max(r)
            if start <= 1:           # 修复丢失文本第 0 个字符的情况
                start = 0
            elif start == end:
                start -= 1
            elif re.match(r'^[a-zA-Z0-9]$', text[start - 1]) and (text[start - 2] in ['\t', '\n', '\r', ',', '.', ':', ';', '-', '+', '"', '(', '/', '&', '*']):   # 修复没有空格而丢失单词第一个字符的情况
                start -= 1
            else:
                pass
            temp.append(f"{start} {end}")    
        temp = ";".join(temp)    
        results.append(temp)
    return results
```

This post-processing can improve 0.002 on a single model and about 0.001 after model ensemble.

### Pseudo-Labeling

We first ensemble the 5-fold deberta xlarge and v3-large model with post-processing, and got 0.891 in Public LB, we further used this model to pseudo-labelling the unlabeled data in patient_notes.csv. (sample 2000 unlabeled pn_num)

This practice of pseudo-labeling without separating folds can lead to overfitting offline scores (0.90+), refer to discussion:

- https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315321
- https://www.kaggle.com/c/google-quest-challenge/discussion/129840

Separate multi-fold to make pseudo-label: 

train_data_fold0 -> models_fold0 -> pl_fold0 ... models_fold4 -> pl_fold4
and then train_data_fold0 + pl_fold0 -> models_fold0_pseudo
(each fold using different 1000 pn_num, total 5000 pn_num)

CV and LB score are relatively consistent, CV 0.890~0.891 (LB 0.890) and CV 0.891 (LB 0.892). We also tried the soft label through pseudo-labeling, which is a char prob floating-point number instead of hard label (0,1), and hard label works better on LB. Public LB score reached 0.892 (single model) after pseudo-labeling.

### Ensemble

We using char prob blending.

We were the first team to exceed 0.893 on LB (based on overfitting pseudo-label model: 5 folds deberta-xlarge+5 folds deberta-v3-large, which scored 0.891 on PB)
In the last month we were stuck in a bottleneck of LB 0.893.
We trained (using separate multi-fold pseudo labeling) 5 folds deberta-xlarge and 5 folds deberta-v3-large two weeks before the competition deadline, and also achieved a LB score of 0.893. (Which also scored 0.891 on PB)
Finally, we ensemble the two 0.893 models as the final submission (2 * 5fold deberta-xlarge, 3 * 5fold deberta-v3-large)

### Threshold

Several thresholds 0.43, 0.44, 0.45, 0.48, etc. were tested on the Public LB, and the best result was 0.44. But after the Private LB was released, we found that 0.45 was a better threshold than 0.44. Unfortunately, we did not select it.
Meanwhile, we set the threshold based on case_num and feature_num separately, but it does not reach the expected result (may be our threshold range was not well constrained), refer to the 4th place’s sharing:
https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/322799

### Others

Some experiments not work (on Public LB):

| Model | Method |	CV	| Public LB| Private LB|
|--------|---------|------|----------|------------------|
|Deberta-v3-large(3 fold)|	smoothFocalLoss |	0.880|	0.880|	0.883|
|Deberta-v3-large(1 fold)|	+LSTM|	0.876|	0.874|	0.878|
|Deberta-v3-large(5 fold)|	+WeightedLayerPooling|	0.882|	0.884|	0.880|
|Deberta-v3-large(5 fold)|	+Stage 2 : LGBM|	0.949|	0.883|	0.885|
