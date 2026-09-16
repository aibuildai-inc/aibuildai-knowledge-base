# 16th place solution

Competition: commonlit-evaluate-student-summaries
Rank: #16
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446872

Thanks, Kaggle and @cookiecutters for this interesting competition. Congratulations to all winners, and gold medalists. I would like to give credit to my talented and hard-working teammates @conjuring92 and @syhens.

Kudos to @tsunotsuno for their fantastic work on LGBM trick, and all participants of this competition for their sharing and hard work.

# First words:
- We think this competition is a perfect example of out-of-distribution forecasting, therefore, we don’t fully trust either CV or LB. Instead, we judge the models by our sense of training stability, partly CV, partly LB, and diversity;
- Therefore, we try to think of as many model architectures, and model training techniques as possible;

# Model 1 - Transformer + Catboost
### Transformer model
- CV/LB/PB: 0.552/0.488/0.556
- Input: text + [SEP] + prompt_question
- Loss: MCRMSE 
- Backbone: deberta-v3-large
- Hyperparams:
    * max_length: 256
    * pooling: attention
    * random-reinit: 1
    * differential learning rate factor: 2.6
    * lr: 1e-5
* epochs: 4
* evaluate at the end of each epoch
- add additional 2x full-fit checkpoints, it helps to gain 0.003 in PB

### GBDT - Catboost
- CV/LB: 0.513/0.443/0.484
- Hyperparams:
    * learning_rate: 0.05
    * Cat_features = [“summary_n_sentence”]
- Features:
    * summary_length, summary_n_sentence, mean_edit_distance of summary sentences, …
    * remove many features like spell errors, POS tag features, …
    * removing prompt_length improves CV a bit, but no help in LB/PB
- Total inference time: 32mins

# Model 2 - Span Model

- Reformulated the problem as a span-regression task by randomly grouping N summaries associated with a particular prompt and predicting their scores together 
- Predictions were conditioned on a context that consisted of the top 8 sentences from the prompt text as ranked by their importance. 
- The importance value of a sentence from the prompt text was determined as the average cosine similarity score between the sentence and all summaries for the prompt
- Used MSE Loss on the context and wording targets
- Discretized the content and wording scores into 16 bins to create auxiliary targets and used ArcFace loss on aux targets
- Used ranking loss among summaries in each example span
- During inference, the random grouping of N summaries provided a TTA-like impact
- Private LB by using 2x full-fit checkpoints: 0.473

# Model 3 - Search Approach

- Input text is comprised of a context and the student summary
- Context:
    * Retrieved the top 3 sentences from the prompt text based on cosine similarity scores with the summary. 
    * Concatenated prompt title and prompt question 
- Trained the model using BCEWithLogitsLoss
    * Scaled the scores between 0-1 
    * Used BCEWithLogitsLoss to compute the loss between scaled scores and logits
- Private LB by using 2x full-fit checkpoints: 0.476

# Model 4 - Long Context

- Remarks:
    * max_len: 1024
    * lr: 2e-5
    * weight_decay: 1e-2
    * n_epochs: 3
    * AWP and EMA after 1 epoch
    * evaluate 10 times/epoch
    * Inference time: 205 mins (4 folds)
    * CV/Public LB/Private LB: 0.499/0.438/0.462
    * Backbone: OpenAssistant/reward-model-deberta-v3-large-v2
    * We have another model with this structure and its CV/Public LB/Private LB: 0.494/0.448/0.459. Unfortunately, we didn’t include this model in the ensemble
- Problems:
    * Inference time is too long due to the long context → not good for ensemble
- Code:
    * Training: https://www.kaggle.com/datasets/shinomoriaoshi/commonlitsummaryv5e
    * Inference: https://www.kaggle.com/code/shinomoriaoshi/commonlitsummary-v5e-infer 

# Model 5 - Dual Encoder with Transfer Learning

- Model Ideas:
    * Instead of concatenating text and prompt info (prompt text, question, title) before passing through the backbone, we can concatenate them after the transformer. One benefit is that many summaries share the same prompts, we can compute them only once and put them into a hashing table, once the embedding of a prompt is computed, we can just query it instead of computing it again.
    * However, I think this way is not good in terms of performance because there is no connection between text and prompt (they’re processed separately in spite of being processed by the same backbone). To overcome it, I load the trained backbone from the long-context model (trained with concatenated text and prompt, with a long sequence length), freeze the top 12 layers, and train the second stage model on top.
- Remarks:
    * max_len context: 1536
    * max_len text: 512
    * n_epochs: 1
    * AWP and EMA enabled after 50 steps
    * Use LGBM trick
    * Inference time: 39 mins (4 folds)
    * CV/Public LB/Private LB: 0.500/0.445/0.470
    * Backbone: OpenAssistant/reward-model-deberta-v3-large-v2
- Code:
    * Training: https://www.kaggle.com/datasets/shinomoriaoshi/commonlitsummaryv6c 
    * Inference: https://www.kaggle.com/code/shinomoriaoshi/commonlitsummary-v6c-infer 

# Final ensemble
We make ensembles of these above models and the public models by @tsunotsuno (we actually train it by ourselves rather than get the model from them directly). We observe a super messy correlation between CV and Public LB here, at the ensemble phase. For example, we simply took the average of the component models, the weights were optimized with some constraints to deal with overfitting.

# What didn’t work
- We tried to generate synthetic data from LLMs (T5 and Llama), but it didn’t work at all;
- Training with LoRA for target-wise models worked and got CV 0.53+, but didn’t help in the local ensemble;

# Team member
- @conjuring92
- @syhens
- @shinomoriaoshi

