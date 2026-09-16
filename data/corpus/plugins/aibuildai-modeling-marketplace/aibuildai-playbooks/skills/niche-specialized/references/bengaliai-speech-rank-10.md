# 11th place solution

Competition: bengaliai-speech
Rank: #10
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/448126

# 11th Solution

The solution consisted of a single whisper medium model using a beam decoder with a size of 4.

###### Training
- Like other solutions, the most important step is to use cleaned data. I use following rules based on metadata shared by host:

```python
cond0 = train_df[
    (
        (train_df.ykg_wer < 0.6) | (train_df.ggl_wer < 0.6)
    ) & (
        (train_df.total_wer_by_client_ykg < 0.7) |
        (train_df.total_wer_by_client_ggl < 0.7)
    ) & (train_df.mos_pred > 1.5)
]
```

- To speed up training, about 80% of the dataset is presented to the model in combinations of two audios, showing only single audios to prevent hallucinations. This also helps to reduce the impact of possible bad annotations.

- SpecAugment, SpecAugment++, CutOut were used.

###### Inference

- For inference, the most important thing is to handle correctly audios longer than 30 secs and sentences longer than 448 tokens. This made an improvement on LB from 0.43 to 0.38.

- Inference code: https://www.kaggle.com/code/themadrambito/11th-place-whisper-inference
