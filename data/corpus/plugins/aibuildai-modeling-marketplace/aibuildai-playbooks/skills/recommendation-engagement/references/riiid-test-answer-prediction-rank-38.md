# 38th place solution: Single SAINT+

Competition: riiid-test-answer-prediction
Rank: #38
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209689

First of all, thank you to the host and Kaggle team for the competition. I think it is a very useful challenge for those who want to try transformer models outside of the nlp. I would also like to thank everyone who shared their ideas for this competition. I can say that it contributed to my learning journey in many ways.

Thank you to Tito and his great [notebook](https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering) about loop for feature engineering
Thank you to Mark and his great [notebook](https://www.kaggle.com/markwijkhuizen/riiid-training-and-prediction-using-a-state) for the harmonic mean idea. 

Results
- 
My local train/val AUC: **0.804** / **0.806**
I already knew the difference between my public and private lb score, because of the private leak.
As a CV strategy, I used a similar but not the same as tito.
Model details
- 
The model structure is very much the same as [SAINT+](https://arxiv.org/abs/2010.12042) paper, except below parameters.
- 2 encoder - 2 decoder layers
- 256 embed dim
- 0.1 dropout for attention layers, 0.2 dropout for fully connected heads at the end of the encoder-decoder layers
- 100 seq length

**Encoder embeddings (Known features)**
- Pos embedding (Categorical)
- Question id embedding (Categorical)
- Bundle id embedding (Categorical)
- Part embedding (Categorical)
- Question encoded tag embedding (New number for each unique value) (Categorical)
- Question attempt embedding (Categorical)
- Lag time embedding (current task timestamp - prev task timestamp) (Categorical)
- Harmonic mean embedding (Continuous)

**Decoder embeddings (Future features)**
*All features shifted/adjusted*
- Pos embedding - same embedding layer for encoder (Categorical)
- Answer embedding (Categorical)
- Elapsed time (Categorical)
- Answer bundle id - same embedding layer for bundle id (Categorical)

Training:
- 
- 25 epoch, train almost took 7-8 hours on GPU, inference 3-4 hours
- Adam with 1e-3 lr for first 20 epoch, then 1e-4 last 5 epoch
- 128 batch size

Notebook [link](https://www.kaggle.com/fomdata/38th-solution-saint-model)
