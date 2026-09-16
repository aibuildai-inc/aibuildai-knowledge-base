# 11th place solution

Competition: gendered-pronoun-resolution
Rank: #11
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90483#latest-523417

## Base model
I have 2 base model all coming from public kernel. 

- [Taming the BERT](https://www.kaggle.com/mateiionita/taming-the-bert-a-baseline) -&gt; `Model 1`
- [Bert + Score Layer ](https://www.kaggle.com/chanhu/bert-score-layer-lb-0-475) -&gt; `Model 2`

Thanks for their selfless sharing.

## Modification
1. Using large-bert instead of base-bert(~0.04).
2. Concat last 8 layers instead of last layer(~0.1).  Here I made some attempts. If I increase the number of layers after the 8th floor, it will not be much better.
3. Concat uncased-bert and cased-bert(~0.04). In fact, they can also be used separately in different models and then merged, but doing so will increase the number of models, I don't have time to adjust them one by one, so I used them directly together. If you want to go deeper on ensemble, use uncased model and cased model separately to provide better diversity. Or instead of using the concat method but using the averaging method. But based on my experience in `Quora` and `Jigsaw`, this method usually performs slightly worse, at least on local-cv.
4. Using 10fold-cv instead of 5fold-cv(~0.015). The promotion is much bigger than I expected. Maybe it’s because the training data is too small, I guess A-A.
5. Adding  `A*P, B*P`  (~0.05), only for `Model 1`. I tried some combinations like `A + B`, `A - B`, etc. In the end, only this set of features has an effect. This finally stabilized the gap between `Model 1` and `Model 2` at 0.02.
6. Tuning mlp and ffnn(~0.01). In fact, this step is very important, and it is executed earlier than other steps. It ensures that each fold can converge normally and avoid over-fitting problems caused by excessive use of bert. 

## Ensemble
I chose four models for ensemble: `Model 1` * 1, `Model 2` * 3(Architecture is slightly different). These four models are handed over to lgb and lasso for ensemble. The final result is as follows:
- LGB - CV 0.3059 - LB 0.2047
- Lasso - CV 0.3074 - LB 0.2336

## What didn't work
- Fine-tuning. Until now, I don’t know the correct posture of fine-tuning, it’s very awkward.
- Correct the wrong label. 
- Elmo.
- Siamese network. Later I found out that it is helpful to solve this problem, but it requires some skill when using it.

## Written at the end
- Thanks to Google AI and Kaggle for organizing this research competition giving everyone the opportunity to try the LM. During Quora, I often wondered how good it would be if I could use bert. The only pity is that I had a lot of time at the time but the rules didn't allow it, but this time I didn't have a lot of time to try LM.
- I think augmentation is also important in addition to fine tuning. I am even curious as to whether the promotion brought by fine tuning will actually be larger than some of the structures or features specifically designed for this problem.
- I think the name of my team is really good, isn't it?


See you in Jigsaw. Have a good time.
