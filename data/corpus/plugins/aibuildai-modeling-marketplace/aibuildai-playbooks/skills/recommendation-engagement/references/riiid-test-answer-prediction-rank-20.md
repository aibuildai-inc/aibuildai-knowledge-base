# 20th solution, transformer encoder only model (SERT) with code and notebook

Competition: riiid-test-answer-prediction
Rank: #20
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209587

Thanks to the organizers for hosting a comp with very little shakeup. This is always nice to see. I wanna thank @wangsg and @leadbest for the starter kernels. Without those I would not have known what to do in this difficult comp.


# Architecture

I use the transformer encoder only SERT (SIngle-directional Encoder Representation from Transformers), to make predictions just one linear layer after the last encoder layer. This is probably a mistake

# Embeddings

1. Question_id
2. Prior question correctness
3. Timestamp difference between bundles
4. Prior question elapsed time
5. Prior question explanation
6. Tag cluster thanks to @spacelx 
7.  Tag vector
8. Fixed pos encoding, same as in Attention is All You Need

# Key modifications
1. encoder only, this is kind of stupid and a mistake. I think this cost me a few places
2. fixed pos encoding, which allows retraining the model with longer sequences
3. layer norm and dropout after embedding layers
4. Loss weight favoring later positions, np.arange(0,1,1/seq_length)*loss

# Mistakes I couldn't resolve
1.	Intra-bundle leakage, I made a nice task mask implementation, but it only made my score worse, probably because of the lack of a decoder. I ended using just an autoregressive mask
2.	First few positions during inference and training may have wrong timestamp difference since i simply do t[1:]-t[:-1]

# Code and submission kernel
https://github.com/Shujun-He/Riiid-Answer-Correctness-Prediction-20th-solution

https://www.kaggle.com/shujun717/fork-of-tag-encoding-with-loss-weight?scriptVersionId=51272583

## Feel free to ask questions. This is a very brief write-up
