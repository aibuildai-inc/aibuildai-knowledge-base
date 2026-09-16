# 7th place solution: Get 0.892 in just 10 minutes

Competition: nbme-score-clinical-patient-notes
Rank: #7
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322829

First of all, I would like to thank competition organizers for hosting this interesting competition. And 
 thanks to my great teammate @zzy990106 , we discuss and work hard for the whole last month to explore new methods..  
And also thank to the community of great notebooks and discussions.

## Summary
- Baseline
- Models
- Pseudo labeling
- Postprocess
- New pipeline (Get 0.892 in just 10 minutes)
- Submission

## Baseline

We used the baseline based on the great notebook of [this one](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train).

## Models

We used the following models:

- deberta-v3-large
- deberta-large
- roberta-large
- deberta-v2-xlarge

Before training with baseline, these models were pretrained on unlabeled data, it improves both cv (+~0.005) and lb.

## Pseudo labeling

The PL were based on ensembling of models mentioned above. To avoid cv leakage, we used single fold to PL. You could see detail [here](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315321).

For training process, we randomly sample 100k PL data and combine with train data on each epoch.

## Postprocess

When we did bad case analysis, we found such mismatched case:

[[Mismatched case]](https://postimg.cc/753jL58N)

`j` was lost in prediction. This only happened the words not begin with a space. So we changed the postprocess code:

```python
def get_results_with_text(char_probs, texts, th=0.5):
    results = []
    for char_prob, text in zip(char_probs, texts):
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        
        tmp_result = []
        for r in result:
#             print(r)
            min_r = min(r)
            max_r = max(r)
            
            if min_r > 0 and text[min_r - 1] != ' ':
#                 print(text[min_r])
                min_r -= 1
                
            tmp_result.append(f"{min_r} {max_r}")
        tmp_result = ";".join(tmp_result)
        results.append(tmp_result)
#         result = [f"{min(r)} {max(r)}" for r in result]
#         result = ";".join(result)
#         results.append(result)
#         break
    temp = []
    for pred in results: 
        if len(pred)>0 and pred[0:2] == '1 ': 
            pred = '0' + pred[1:]
#             print(pred)
        temp.append(pred)

    results = temp
    return results
```

It could boost both cv (+0.001~0.002) and lb (+0.001). 

## New pipeline

In the last period of competition, we thought that the baseline we used, not concerned about relationship of `features` in the same `case num`. After discussion, @zzy990106 proposed that we could use multi-class classification. So we coded a new pipeline with 143 (143 features) classification head.

For the new pipeline, we regarded each `pn_num` text as a sample. As we have PL extra data (About 40000 samples), so we could make models train better. Otherwise, the train dataset only contains 1000 samples (1000 `pn_num`)...

The inference for this new pipeline is really fast, only takes ~10mins for deberta-v3-large (5 folds), ~15mins for deberta-v2-xlarge (5 folds).

And the cv and lb are pretty high:

- deberta-large (cv: 0.895, LB: 0.892, PB: 0.891)
- deberta-v3-large (cv: 0.894, LB: 0.891, PB: 0.892)
- roberta-large (cv: 0.894, LB: 0.891, PB: 0.892)
- deberta-v2-xlarge (cv: 0.8952, LB: 0.891, PB: 0.892)

Maybe we could ensemble more than 20 models using this new pipeline...

## Submission

The submission we select based on 8 models ensembling, but this was not our best pb. We still have 6 higher pb submissions not to choose... They all have very high cv.

We found that the variance of LB was pretty high,  and there could be a shakeup. So we trust more on our cv but not LB.
