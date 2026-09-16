# 20th solution - 2 models, various embeds, mixed loss

Competition: quora-insincere-questions-classification
Rank: #20
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80527

I was shocked when I saw the final standing. We never passed the 0.7 baseline on public LB and it was really frustrating. I basically gave up and just prayed this competition was about CV instead of public LB. It turned out to be true. I teamed up with [YangHe][1] in the last week and we decided to make two submissions: one focusing on CV and one focusing on public LB. I was responsible for the CV one. My final submission reached CV 0.698 and public LB 0.699.

Anyway, this is my [kernel][2].


**Pre-processing:**

Basically the public kernel, with some bug fixed (order of punc clean/contraction clean) and more contraction cleaning. I also used multiprocessing to speed things up. I met a bug when using Keras Tokenizer with PyTorch model: I couldn't set num words=None in the Tokenizer. It would run into some CUDA error during the training phase. So I fitted the Tokenizer locally and set num words = len(tokenizer.word_index) in the kernel.


**Model:**

I built two models:

1. concat(GloVe, FastText) embedding + LSTM + TextCNN with kernel size [1, 2, 3, 4] + 2 dense layers, with some batch normalizations and dropout layers

2. mean(GloVe, Para) embedding + LSTM + GRU + concat(GlobalAvgPool, GlobalMaxPool) + 2 dense layers, with some dropout layers

We noticed the [bug][3] in the embedding dropout after the submission deadline.


**Training:**

I split the training data into 4 folds.

Loss: BCE + soft F1 loss. I changed from BCE loss to this mixed loss on the last day and it gave me 0.003 boost on public LB and 0.002 boost on CV. It gave a stabler threshold v. F1 curve at the optimal point and I believe this granted us the 20th position. I also tried BCE + Lovasz, BCE pretrain and Lovasz fine-tune, BCE pretrain and soft f1 fine-tune, etc. Some of them didn't improve the model, others didn't converge at all. The model didn't converge when I was using pure soft F1 loss. This might be due to the imbalance of the label. Oversample might be needed when using soft F1 loss, but I didn't have the time to try. 

I used consine schedule with max LR = 0.003, and trained each model 4 epochs. I think consine schedule is better than step schedule and it is my favorite scheduler of all time. Notice that overfitting the training set a little would give a stabler threshold v. F1 curve. That's why all 0.7 public kernels overfit, 
I also tried AdamW with weight_decay = 0.0001, and it indeed gave better result. I didn't use it since it took more time to run.


**Post-processing:**

Average of all 8 classifiers and set the threshold based on oof prediction. I have made an all positive submission to figure out there were 3376 insincere questions in the public test data. I noticed that a lot of solutions to the past competition set the threshold so that the ratio of predicted label in test set is the same as training set. However, I didn't do that because I felt that it would be dangerous to use the same strategy in a binary classification problem.


**Some Takeaways:**

1. Stability is the key. You want threshold as insensitive as possible. 

2. Model is not the most important thing. The major variation is in the embedding layer.

3. Read discussion, read public kernels, read solutions to similar past competitions, read solutions to different past competitions.

4. When you fork someone's code, read it! It might not be bug-free!

5. Don't give up! The shakeup is REAL!

  [1]: https://www.kaggle.com/kukicap
  [2]: https://www.kaggle.com/jihangz/20th-solution-4-folds-2-models-mixed-loss
  [3]: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/79911
