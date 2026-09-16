# 5th place solution

Competition: nbme-score-clinical-patient-notes
Rank: #5
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322875

First of all, I would like to thank competition organizers for hosting this interesting competition. And
thanks to my great teammates @bestpredict  @lxf615712 ,  thanks for more than a month of hard work. And also thanks to the kaggle community, I had learned a lot from notebooks and discussions.Thanks to the excellent [baseline ](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train/notebook) and [experiment results](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315707), [case5](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315314)

### **method**
- MLM(20epoch)
- Stratified  and greater learning rate ( large 2e-5, xlarge 1e-5, xxlarge 5e-6)
- pseudo labeling ( 2000-4000 nums,[5folds](https://www.kaggle.com/c/google-quest-challenge/discussion/129840)，different models take different samples, resampling of Insufficient case_num， some models not used pseudo labels,  lb is better,  but pb is not robustness)
- multidrop
- fgm
- [deberta v2/v3 tokenizer](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330)
           
### **Models**
- Deberta- v1-large  
- Deberta- v1-xlarge  
- Deberta-v2-xlarge  
- Deberta-v2-xxlarge
- Deberta-v3-large  


### **Ensemble**
Blending of 5 models based on char probabilities.In the last few days, we add v2 xxlarge, it's cv884 and lb883(2folds) is not good, but model emsemblings is better at lb, worse at pb. We had not selected 30+ better pb submissions.**Trust your CV**

### **Postprocess + Threshold**

Space migration + oof set([word probs](https://www.kaggle.com/code/abhishek/two-longformers-are-better-than-1))  , We set thresholds for some features(lower than all best   case_num threshold cv on pb) .



`
def convert_offsets_to_word_indices(preds_offsets,texts, case_nums, feature_nums, th=0.5):

      predicts = []
      for text,preds,case_num, feature_num in zip(texts,preds_offsets, case_nums, feature_nums):

            encoded_text = tokenizer(text, add_special_tokens=True,max_length=CFG.max_len,padding="max_length",return_offsets_mapping=True)

            offset_mapping = encoded_text['offset_mapping']
            sep_index = encoded_text["input_ids"].index(tokenizer.sep_token_id)
            result=np.zeros(len(preds))
            
            results = np.zeros(sep_index)
            for idx, (offset, pred) in enumerate(zip(offset_mapping[:sep_index], preds)):
                    start = offset[0]
                    results[idx] = preds[start]
            sample_pred_scores = results
            if str(feature_num)[-1] == '3' and (str(case_num)=='0' or str(case_num)=='3' ):
                result = [1 if s >= 0.54  else 0 for s in results]
            elif    str(feature_num)[-1] == '3' and (str(case_num)=='1' ):
                result = [1 if s >= 0.45 else 0 for s in results]
            elif    str(feature_num)[-1] == '3' and   str(case_num)=='6':
                result = [1 if s >= 0.52 else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '503' ):
                result = [1 if s >= 0.49  else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '504' ):
                result = [1 if s >= 0.4  else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '508' ):
                result = [1 if s >= 0.49 else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '509' ):
                result = [1 if s >= 0.4 else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '510' ):
                result = [1 if s >=0.55 else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '511' ):
                result = [1 if s >=0.55 else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '512' ):
                result = [1 if s >=0.45 else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '513' ):
                result = [1 if s >=0.59 else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '514' ):
                result = [1 if s >=0.49 else 0 for s in results]
            elif str(case_num)=='5' and ( str(feature_num) == '516' ):
                result = [1 if s >=0.41 else 0 for s in results]
            elif str(case_num)=='7' and ( str(feature_num) == '702' ):
                result = [1 if s >=0.56 else 0 for s in results]

            else:

                result = [1 if s >= 0.47  else 0 for s in results]
            preds = result
            
            sample_text = text
            sample_input_ids = encoded_text["input_ids"]

            sample_preds = []

            if len(preds) < len(offset_mapping):
                preds = preds + [0] * (len(offset_mapping) - len(preds))
                sample_pred_scores = list(sample_pred_scores) + [0] * (len(offset_mapping) - len(sample_pred_scores))

            idx = 0
            phrase_preds = []
            while idx < sep_index:
                start, _ = offset_mapping[idx]
            #     print(start,idx)
                if preds[idx] != 0:
                    label = preds[idx]
                else:
                    label = 0
                phrase_scores = []
                phrase_scores.append(sample_pred_scores[idx])
            #     idx += 1
                while idx < sep_index:
                    if label == 0:
                        matching_label = 0
                    else:
                        matching_label = 1
                    if preds[idx] == matching_label:
                        _, end = offset_mapping[idx]
                        phrase_scores.append(sample_pred_scores[idx])
                        idx += 1
                    else:
                        break
                if "end" in locals():
                    phrase = sample_text[start:end]
                    phrase_preds.append((phrase, start, end, label, phrase_scores))
            temp = []
            for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
                newlabel = ''
                word_start = len(sample_text[:start].split())
                word_end = word_start + len(sample_text[start:end].split())
                word_end = min(word_end, len(sample_text.split()))
                ps = " ".join([str(x) for x in range(word_start, word_end)])

                if label != 0:
                    if sum(phrase_scores) / len(phrase_scores) >= th and word_end !=0:
                        if start!=0 and sample_text[start]==' ':
                            start+=1

                        temp.append(np.array([start,end]))
            if len(temp)==0:
                temp = [list(g) for _, g in itertools.groupby(temp, key=lambda n, c=itertools.count(): n - next(c))]
                temp = [f"{min(r)} {max(r)}" for r in temp]
            #                 print(temp)
            else:
            #                 print(temp)
            #                 print(type(temp[0][0]))
                temp = [f"{min(r)} {max(r)}" for r in temp]
            #                 print(temp)
            predict = ";".join(temp)
            predicts.append(predict)
    return predicts`
