# 9th Weight search and threshold modification

Competition: nbme-score-clinical-patient-notes
Rank: #9
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322891

# 9th Solution

First, great thanks to Kaggle and NBME for hosting such an amazing competition, and congrats to all winners!

thanks to my great teammate @yanbojwang @qinhui1999 @yankuoaaagmailcom @alex821 

## Summary
We ensembled 6 token classification models and set threshold for feature_num with large changes and did postprocess.
## Models
We trained following 6 token classification models and used them all for final submission(29 models).
### Here is Public LB
- token classification Deberta-v3-large 5fold: 0.887 with pp
- token classification Deberta-v1-large 5fold: 0.887 with pp
- token classification Deberta-v2-xlarge 4fold: 0.886 with pp
- token classification Deberta-v1-xlarge 5fold: 0.890 with pp
- token classification funnel-transformer 5fold: 0.883 with pp
- token classification electra 5fold:0.886 with pp
### Deberta-v3-large, Deberta-v1-large, Deberta-v2-xlarge,Deberta-v1-xlarge Method used:
- MLM(0.30)
- FocalLoss
- 2fold pseudo labeling
- FGM
- multiple-dropout
### funnel-transformer, electra Method used:
- MLM(0.30)
- BCE
- Special char (add ‘\n,\s,\t’)
- No pseudo labeling
- FreeLB
- SMA


## Threshold
According to the text length of different features and the change trend of F1, we select some features to set the threshold separately. Instead of selecting more settings in private, we chose a small number. However, setting more values can get higher private scores.
## Tricks
We achieve super acceleration by sorting multiple times with different tokenizer(30 models in 9h).
Because our pseudo tag online performance is not friendly enough, we only took a few fold of data for pseudo tag training, and added the final fusion to prevent jitter. According to CV, we equipped each model with weight to obtain a more friendly LB.
By setting thresholds for some features, it can increase 0.001-0.002 in PB   
Weight search can improve ~0.001
## Post Process
if len(result) > 0:  
    for start,end in result:  
        if start < len(valid_text):  
            if valid_text[start-1] != ' ':  
                start -= 1  
            if valid_text[start] == '\n':  
                start += 1  
Our score in the private list can actually reach 0.894，We expect a big shake in this game, so we choose to be too conservative
