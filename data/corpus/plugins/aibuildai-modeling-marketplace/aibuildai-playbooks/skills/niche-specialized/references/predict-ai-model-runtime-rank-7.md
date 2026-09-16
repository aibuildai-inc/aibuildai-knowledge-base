# 7th Place Solution Write-up

Competition: predict-ai-model-runtime
Rank: #7
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456673

Thanks for hosting this fascinating competition, and congratulations to the winners! I really learned a lot through this competition.

**Tile Part:**
My approach to the Tile part was quite similar to the official code. I also check the public tile codes and find the validation score was already quite high , and ensembling them didn't lead to significant improvements, so I didn't invest too much time in this section.

**Layout Part:**
For the Layout part, I primarily referred to the gst code [https://github.com/kaidic/GST](url). However, I encountered some challenges in making it work efficiently due to GPU memory constraints. Eventually, I used a sampling method to reduce GPU usage. 

**Sampling Method:**
The raw data consumed too much memory, so I decided to sample only 500 or 1000 configurations for each sample. This significantly reduced training time and GPU memory requirements.

**Training Strategy:**
1. Training them all.
2. Training seperately based on the edge & node shapes. While the test data didn't explicitly specify the model type, we could infer it based on the edge & node shapes.
3. Using different kinds of parameters, such as graph conv type, learning rate, batchsize, layers number and hidden size. 

**Ensembling**
Ensembling above models proved to be effective in improving the results. Every time I trained a new model, I found that ensembling it with existing models contributed to score improvements.

**Regarding the Public Score:**
I did identify the fact mentioned in a post [https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456083](url) but couldn't fully understand why it worked. I exercised extreme caution in utilizing it during the private phase since I considered it very risky.
