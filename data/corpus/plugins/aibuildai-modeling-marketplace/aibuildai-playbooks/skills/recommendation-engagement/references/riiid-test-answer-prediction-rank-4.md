# 4th place solution : Single Transformer Model

Competition: riiid-test-answer-prediction
Rank: #4
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210171

Hi everyone,
It has been a real pleasure competing in this wonderful challenge.  Thank you Kaggle and the Competition Host for making it possible.

I'm happy to share here my solution which got me to the 4th place. It is a single transformer model inspired from previous works (like *SAINT, SAKT*) very much discussed in this competition. 

# The model 
[Riiid 4th solution model architecture].png?generation=1610229212586102&alt=media)
I hope my figure is straightforward. Below are key features of the model I'd like to explain more:

### Input sequences
I tried to include all data available from the train table and metadata tables. I also add *time lag*, which is the delta time from the previous interaction (questions in the same container have the same timestamp so they share the same *time lag*). 
Also, on the question table, I added 2 features : difficulty level (*correct response rate* of each questions), popularity (*number of appearances*), which are computed from the whole training data.
### Input embeddings
Same size of embeddings for all inputs, the embeddings are then concatenated and go through linear transform to feed to the first encoder and decoder of the transformer.

Embeddings of continuous features (*time lag*, *question elapsed time*,  *question difficulty*, *question popularity*) are computed using a *ContinuousEmbedding* layer.  The idea of *ContinuousEmbedding* is to sum up `(`weighted sum`)` a number of consecutive embedding vectors (from the embedding *weight matrix*). 

This way we have a "smooth" version for the embeddings of the continuous variable: 2 values very close together should have similar embeddings. 

### The transformer
Input of the encoder are embeddings of all input elements. Input of the decoder doesn't not contain user answer related elements.
Encoder and decoder layers are almost the same as in the original paper ([Attention is all you need](https://arxiv.org/abs/1706.03762)).
One key difference is of course the causal masks to prevent the current position from seeing the future. The other is a feature that I add to improve the performance and convergence speed : a kind of time aware weighted attention. The idea is to decay the attention coefficient by a factor of \\(dt^{-w}\\) where \\(dt\\) is the difference in timestamp of a position and the position it attends to and \\(w\\) is a trainable parameter constrained to be non-negative ` (`one parameter per attention head`)`. This is pretty easy to implement: compute the timestamp difference matrix in log scale, multiply it with the parameters \\(w\\) and subtract it from the attention logits (*scaled dot product* output of the attention layer).
  	
# Training
I use the cv method https://www.kaggle.com/its7171/cv-strategy (thanks @its7171). The model was implemented in Tensorflow and trained on TPU with Colab Pro.
Sequences are randomly cut and padded to have the same length and all parts are kept for training.
The final version of my model has embeddings size of 128, model size of 512, 4 encoder layers and 4 decoder layers. 
It was trained with the sequence length of 1024 for about 36000 steps (*warmup* 4000 steps then cosine decay) and with batch size 64. Training took about 4-5 hours.
On the submission kernel I had to reduce sequence length to 512 due to resource limit.

#Some observations
- Input embeddings: concatenation is better than sum
- Longer sequence `(`for both training and inference`)` improves the performance
- Model size also matters: bigger model size generally improves the performance but going beyond size of 512 and 4 layers does not improve much. 

#Why not ensemble
I didn't have much time toward the end of the competition. When I still made improvement on my single model I made the choice of staying on that rather than spending time on making ensemble of smaller models. I'm not sure it was a good choice,  but it got me this far so I'm still happy. 

# Code
[Here](https://www.kaggle.com/letranduckinh/riiid-model-submission-4th-place-public-version) is the submission kernel that I made public. You should find all my code source training log in the kernel. 
As you can see it scores 0.8180 on valid set, 0.815 on public LB and 0.817 on private LB.
My top scored submission has the same model and training configuration but was trained on the whole training set, which did not improve much.

[Here](https://github.com/dkletran/riiid-challenge-4th-place) is the source code on github including all steps to reproduce the solution.

###Best regards to all
