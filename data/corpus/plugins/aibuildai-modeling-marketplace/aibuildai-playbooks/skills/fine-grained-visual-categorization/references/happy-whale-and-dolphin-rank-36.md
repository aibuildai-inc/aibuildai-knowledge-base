# #36 place solution

Competition: happy-whale-and-dolphin
Rank: #36
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/319812

Congratulations to all the winners. Thanks Happywhale and Kaggle for hosting this competition.

Many thanks for the sharing of Lex Toumbourou and Jan bre, I started from their significant work of [Happywhale - Effnet B6 fork with Detic crop] and the dataset of backfintfrecords.
Many experements in the competition had been tried and finally only three significant improvements in model.

1.**Dynamic margins**: publicLB +0.02 (Thanks Landmark team to share the solution, paper is here: https://arxiv.org/abs/2010.05350): 
  Tested and found the best margins range is 0.05-0.6. but training became difficult( loss is easy to Nan) and need to reduce LR.
2.**Batch normalization**: publicLB +0.02 (Thanks Heng, discussion is here: https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/315129 )
  add Batch normalization layer before dense layer, add L2 normalization for dense:
3.**FreezeBN**: publicLB +0.01(Thanks Balaji, discussion is here: https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/309582 )
  it works for EfficientB6 and B5, but not works for B7(loss always to nan)

here is a training notebook included above 3 implementations
https://www.kaggle.com/code/liuzhangzhen/happywhale-bodydataset-training

finally, two ensembles help to improve public LB to 0.858
	Concatenate 7 model embeddings: improve from LB0.815 to 0.835(body dataset)
	Combine the two datasets(body and fin) test neighbor and take the max confidence: improve from LB0.835 to 0.858 
	
Here is my final solution:
· Dataset: fin and body, use yolov5x6 to train 10folds, test by b0rev256 then pickup best 5 folds, choose the best 1 for train and all others for TTA.(Result: huge effort but slight improvement, maybe should be more carefully to check the dataset and remove noise)
	
· Split: random choose 500 2 pictures and 55 pictures as new whales(about 11% just same percentage as the LB) as validation dataset and use all others for train. (Result: validation dataset is proved to co-related with LB and useful)

· Pseudo: take 4000+ with confidence > 0.95 and all other targets confidence < 0.65

· Augmentation: base notebook + 50% rotation, shear, shift, zoom.(Result: slight improvement)

· Model: dynamic margins, batch normalization layer before dense layer, add L2 normalization for dense, FreezeBN(Result: significant improvements)
	· Model candidate: (final 7-8 models for fin and body)
		· Effv1:  b7 768 Adam(use TPUv3-8), b7 608 Adam(Colab), b6 640 Adam(Colab),  b5 640 Adam(Colab),  b6 480 Adam(Colab), + some SGD model (ex: b7 640)  

· Ensemble
		· TTA:5 fold embeddings and take the mean (small improvement)
		· Embedding concatenate for body and fin dataset(Large improvement)
		· Merge fin and body by max conf, leverage by the LB score(use confidence * (LB score**2))(Large improvement)
