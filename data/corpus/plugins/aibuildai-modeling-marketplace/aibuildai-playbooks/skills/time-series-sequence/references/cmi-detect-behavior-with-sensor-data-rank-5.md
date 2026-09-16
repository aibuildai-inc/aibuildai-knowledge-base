# 5th place solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #5
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/5th-solution

First, congrats for all winners and thanks to the organizers for hosting this interesting competition, the community-we learn a lot from your works and sharings. Thanks to my teammates(Rib~) for the excellent teamwork.

# Tricks
It is obvious that this competition involves some tricks, and our solution mainly focuses on the following two points.
1.Figure out how to use the other sequences' information with the same subject. For convenience in subsequent descriptions, we will refer to this model as the subject-based. We group data from the same subject into a single batch, extract sequential information using CNN, RNN or transformer, and then add a Transformer layer along the subject dimension to capture inter-sequence relationships. During offline validation and online inference, you need to accumulate each subject’s historical sequence information before making predictions on the current sequence.


Since the evaluation API provides data in batches of single sequence rather than all sequences from the same subject, and additionally shuffles the order of sequences for each submission, this approach leads to some instability. We address this issue in the following ways:
- During the training process, the sequences from the same subject is sampled at different ratios(0.5~0.8), and this process is repeated multiple times.
- We have observed that the effectiveness of this solution is limited when historical data is insufficient. Therefore, we still need the sequence-based model to correct predictions under conditions of inadequate historical data, that's the second part of our tricks.

2.We prepared 4 types of models: 
| model | training approch |
| --- | --- |
| imu-only_subject | use imu features only, train subject-base |
| imu-only_sequence | use imu features only, train sequence-base |
| all_subject | use all features, train subject-base |
| all_sequence | use all features, train sequence-base |

We adopt the following strategies for model selection and ensemble:
- When the proportion of missing values exceeds 50%, the imu-only model is selected. Otherwise, the all model is chosen.
- The imu-only model can further enhance the performance of the all model. Therefore, we apply a weighted fusion of the two.
- When historical data is insufficient, the sequence-based model is used for ensemble, with its weight adjusted based on the amount of accumulated records.
```
df_test = df_hist[subject]
n_hist = len(df_hist[subject])
w_imu_tf = min(n_hist,15)/15
w_all_tf = min(n_hist,30)/30
n_samples = sequence.shape[0]
if sequence['thm_1'].null_count()/n_samples>=0.5:
     subject_predictions = inference(imu-only_subject_model) 
     sequence_predictions = inference(imu-only_sequence_model)
     predictions = w_imu_tf*subject_predictions + (1-w_imu_tf)*sequence_predictions 
else:
     imu_predictions = inference(imu-only_subject_model)
     all_subject_predictions = inference(all_subject_model)
     all_sequence_predictions = inference(all_sequence_model)
     predictions = w_all_tf*all_subject_predictions + (1-w_all_tf)*all_sequence_predictions
     predictions = predictions*0.8+imu_predictions*0.2
```

# Others
1.  Features & Data augment: we use the features and augments mainly from the public codes, thanks again to those contributors.
2. Validation: a little different with the public ones, we just split by subject. 5folds cv: imu-only: 0.855, all: 0.898.
3. The subject-based model improved the performance by ~0.02 compared to the sequence-based model, while the ensemble (average) of both of our individual solutions further increased the performance by ~0.01.
4. Our sequence-based model may be relatively weaker compared to those of other top teams, which could be one of the reasons for the performance gap.
5. Due to the special API mechanism I mentioned earlier, our approach results fluctuate with each submission. Therefore, we finalized our solution three days ago and used the last 15 submission opportunities to repeatedly evaluate and select 2 of them.
