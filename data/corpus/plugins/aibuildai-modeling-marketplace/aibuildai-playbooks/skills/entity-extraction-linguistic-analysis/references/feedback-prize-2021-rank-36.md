# 36th place - deberta-large-1024/longformer-large-1536 ensemble with thresholding + code

Competition: feedback-prize-2021
Rank: #36
Source: https://www.kaggle.com/c/feedback-prize-2021/discussion/313452

First of all, I'd like to thank organizers for creating this nice competition and the whole kaggle community for sharing a lot of fruitful ideas during the competition. I learned a lot about modern NLP techniques used for analysis of large texts. 

In particular, I'd like to thank @cdeotte for his comprehensive [starter kernel](https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-633), which really helped me to get going, @abhishek for a strong [baseline](https://www.kaggle.com/abhishek/two-longformers-are-better-than-1), which was probably used by majority of the top teams in this competition, as well as @hengck23, whose [model ensembling](https://www.kaggle.com/hengck23/1-birdformer-1-longformer-one-fold/) approach I used and who also generously shared a lot of results from his experiments. Please, give their kernels an up-vote!

My solution is using an ensemble of 5 models: 2x deberta-large-1024, 2x deberta-v3-large-1024 and longformer-large-1536 trained on 5 folds each.
- I used Abhishek's 5-fold split and trained all the deberta-large models using max_len=1024, since due to GPU memory restrictions I was not able to go any higher and deberta models do not seem to support gradient checkpointing out of the box. I used max_len=1536 for longformer-large.
- OOF validation scores showed good correlation with the public leaderboard scores, which was very nice to see and gave me confidence in my modeling approach.
- **Deberta-large-1024 models gave the best performance. I did not verify this with a LB submit, but I think a simple ensemble of two deberta-large models using vanila Abhishek's code with properly tuned thresholds would be enough to get a silver medal. I estimate that one would get a public LB score of 0.702-0.703 and 0.713-0.714 on the private LB.**
- Adding 2x deberta-v3-large-1024 and longformer-large-1536 allowed me to get ~+0.004 boost and the public LB score of 0.707 (or 0.719 on the private LB). This ensemble appeared to be fairly robust as I picked up 11 positions on the LB after the shake up.
- I trained the models using Abhishek's code with batch_size=1, where I added gradient_accumulation=4 (so effectively I used batch_size=4) to fit them on my 15GB GPU. I used half a cycle cosine schedule with warm up of 0.1 and 5 epochs and AdamW optimizer. Training converged after 3-4 epochs for the most part. 
- I observed that slightly larger hidden_dropout=0.15 gives better validation scores than hidden_dropout=0.1, although I included both cases into the final ensemble for deberta. I also used LR=2e-5, however there were a couple of folds, when training did not converge in which case I dropped LR to 1e-5 on a subsequent attempt.
- For ensembling different models, I used @hengck23's idea of predicting probabilities for every symbol in the text, then averaging probabilities of different models before predicting discourse.
- I tuned the probability and length thresholds for different discourse classes using OOF data and computed average thresholds using the 5 folds, which gave me +0.004 boost compared to the thresholds posted in public kernels.
- Inference of the 5 models, having 5 folds each, took a little over 8 hours in the Kaggle kernel.
- I tried different ensembles by adding BigBird-Large, Funnel-Large, and medium-size deberta models to the mix, however this did not improve the score.
- I also tried applying ranking to the probabilities before averaging to correct for any systematic model prediction shifts, however that made the scores worse.
- Unfortunately I ran out of time and did not get a chance to try to build a better 2nd level post-processing model myself, but I am currently looking at the amazing solutions from the top teams. Thanks for sharing!

My training code: https://github.com/akuritsyn/feedback-prize-2021
Submission kernel: https://www.kaggle.com/akuritsyn/feedback-model-ensemble

Good luck everyone!
