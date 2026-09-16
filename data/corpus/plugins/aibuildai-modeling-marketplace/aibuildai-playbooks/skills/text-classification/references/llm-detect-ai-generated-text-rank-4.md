# [4th Place Solution] A Summary of Combined Arms Approach

Competition: llm-detect-ai-generated-text
Rank: #4
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470179

Firstly, I would like to thank kaggle, the hosts, and everyone who participated in this enjoyable competition. I learned a great deal along the way, and it was quite refreshing during this emerging era of large language models. 

I would also like to extend my gratitude to my teammate, @chasembowers. Although we merged in the last day of team merging, the integration process was quite smooth and productive. We were able to quickly sync our approaches, utilizing the strong points of both pipelines together. 

I am also sorry for those who worked hard during the competition but were affected by the final shakeup. I wish you better luck next time!

**Overview**

Since it was really hard to implement a reliable CV scheme —although we had separate CV's for each piece— we wanted to have a generalized model as possible. Developing a generalized model in uncharted waters presented its own challenges, as we were uncertain which approaches would be successful. With our custom approach, we could have a resilient final model where each piece is strong in some aspect of the data and could help each other where another one fails. You can refer to this as a simple ensemble, somewhat of a mixture of experts, or a combined arms approach if you prefer.

So I'd like to explain each important piece individually:


**Classical ML Approach:**

This is a section where many of you are already familiar, as it is similar to my public notebook with only minor parameter adjustments, such as:

- Wider ngram range (3,7)
- Limited feature space, such as max_df and max_features...
- Extra tokenizer preprocessor steps like normalizers etc.
- The use of the DAIGT V2 training dataset, for which I am grateful to @thedrcat, with some post-processing.
    - Instead of fixing/removing typos, we embraced them. But they were causing some degree of overfitting between human and LLM-generated essays, so we decided to replicate artificial typos. 
    - At first, we believed these obfuscations followed certain patterns, such as keyboard typos where a typo character is typically near its neighbors. However, after further analysis, we discovered that character-level obfuscations were completely random. So we chose that approach as well and inserted/replaced artificial typos randomly to 30% of the generated data.
- This piece itself was an sub-ensemble of different linear models and gradient boosting models. 

In the end, the solution was not significantly different from what I had shared a few weeks prior.  It was a flagship piece and alone it was a public lb gold zone solution for a long time. We utilized this piece in other pieces too as I'll explain in next section. However it was a risky pipeline alone in terms of overfitting. It needed more generalized models around it to mitigate this effect...

**LLM Approach:**

I will briefly discuss this aspect, as my teammate @chasembowers explained the details in depth [here](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/471934). Essentially, we developed mistral-7b based features (think of token probabilities, features over logprobs, perplexity etc.), to train a discriminator and make predictions with it.

But we took it further by pseudo-labeling the test set using the previous TF-IDF approach and retraining a bunch of ML models based on these labels using the mistral features, such as linear and gradient boosting models. This helped us find a good balance between the LLM's generalization capabilities and TF-IDF's token based, task-specific strong indicators.

**Transformer-1: Longformer**  

In this complimentary experimental piece, we trained a longformer model, but instead of predicting binary outcome, we set the training set source as the target. Then we merged these predictions with previous predictions to create a feature space and trained another set of models over them.

**Transformer-2: Deberta Large:**

In this final piece, we trained a DebertaV3 model using a vast merge of publicly available data, amounting to around 700k instances. This model served as our regularizer and a fallback option for instances when TF-IDF based models were uncertain, such as cases involving data drift in private data.  It also assisted in softening overly confident predictions.

We trained this model for 1 epoch (it was already achieving 0.999 CV early on) with 1024 sequence length, without early stopping and with minimal or no learnable parameters in the first layers of the model.

**Final Ensemble:**

In final part of ensemble each model had it's own weight, but most importantly we didn't gave weights over their public lb scores since it was too risky, we used common sense and balanced  approach while manually setting the weights since they all had perfect CV scores.

Here's the training(most of it) and inference [notebook](https://www.kaggle.com/code/chasembowers/4th-place-solution/notebook). [It should be public pretty soon.]

**Final Thoughts:**

I truly enjoyed this competition, particularly the collaborative spirit of sharing ideas, sources, and working together. Surviving this intense shakeup is truly rewarding, as is sharing our findings in discussions and notebooks.

Throughout the competition, as a team and individually, we explored a vast number of ideas, and while many did not work out, the final week was particularly exhausting as we wrapped up and finalized our work.

I've tried to keep this post simple, and I apologize if I missed any crucial details. Please feel free to ask any questions you may have in this thread, and I will be more than happy to provide answers when I have time.
