# 18th place Solution -Thank You !

Competition: learning-equality-curriculum-recommendations
Rank: #17
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394910

Many thanks to organizers for setting up this competition which I believe not many competition we will find of this kind.  It was amazing to see 1st rank solution based on mere traditional machine learning quite a number of things to learn from  . 
I thank all my team members working hard on this competition @iafoss  @rohitsingh9990  specially @evgeniimaslov2  whose last 2 weeks efforts put us in top 20. Lot more new things as take away from other Top solution
Below is the outline of our solution
We used 2 stage solution approach with channel level  5 fold StratifiedKfold (Alighned quite well with LB)
**Stage1**
1) 64 seq length Paraphrase mpnet v2  Trained on Similarity loss (With different Seq l & Pos only +Pos/Neg combination)
2) 128 seq L Roberta L   Trained on Arcface Loss (Pos only samples) 
Max pos score of all our models on whole dataset was 0.95+ (Top 50) , while Validation fold was 0.85+
**Stage2**
Ensemble of 
1) 64 seq Length 5 fold Paraphrase Mpnet v2  Pretrained on stage1 with Top 50 Neighbors  Trained on Contrastive loss  ( This gave significant boost compared to ReRanker based public approach) ( This boosted score from 0.57 to straight away 0.657

2) 64 Seq length  5 fold Roberta L as classifier trained on same contrastive loss   Top 64 Neighbors from corresponding Roberta model of stage1 
  
3)  256-352 Seq length 5 fold ReRanker trained using public kernel approach Reducing  some of FPs count of Model 1  (This boosted the score by  0.01) from 0.657  to 0.66-0.67 series

4) Finally Light GBM  This took the score to 0.68 series ( a final boost)

**Micro level Approaches**
1) Quite a number Top N selections  made for stage2 based on similarity ranking/ CV boost for Stage2 using these approaches  @evgeniimaslov2 can throw some more light on this
2) Light GBM based on prob features (Ranking based)  and other Train features like Kind, category etc  after stage2 classification . This gave boost of around 0.02 to land us to our current score. using light gbm our CV reached to 0.74 to get private lb of 0.721 quite aligned.

All in all it was progressive move using above approaches during entire period of competition.

Regards
Jaideep
