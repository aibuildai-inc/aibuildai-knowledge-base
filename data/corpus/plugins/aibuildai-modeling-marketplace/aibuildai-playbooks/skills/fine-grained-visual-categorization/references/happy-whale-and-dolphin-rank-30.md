# 30th Simple Solution

Competition: happy-whale-and-dolphin
Rank: #30
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320205

I would like to thank the organizers for hosting the great competition.
also, I congratulate all the top players.
I was very honored to become a teammate with @hwigeon 


##Data
We started the original dataset + detic crop at first.
We doubled dataset with original dataset + detic crop in the same training phase to make our arcface learn a margin well. 
but finally, We manually made a cropped full-body dataset itself and replace this with the detic crop.


##Models
Efficientnet 5,6,7+Arcface or Dolg,  Nfnet, Nfnet+hybridVit, Nfnet+dolg were used.  
Arcface margin = 0.3 is chosen.
The size range is [768~1024]
RAdam+Lookahead or Madgrad is chosen as optimizer. 
Original dataset + our manually cropped dataset reached public LB 0.83x easily
After Pseudo labeling, we could have scored public LB 0.858, private LB 0.826  



##Augmentation
Flip augmentation for Nfnet networks.
Flip + Hsv augmentation for Efficientnet.  



##Validation
A valid dataset that only has 2 individuals are chosen as holdout.  
We didn't care about new individuals since we believed that the top1 score matters most.
the demerit of our validation set is that it can't post-process our results to predict new_individuals.
Our top1 cv reached 0.89x  


##Postprocess
to use KNN, we concatenated all the embeddings to axis =-1
after We used KNN=1 and gather nearest 880 individuals.
new_individual threshold = 0.375,0.4,0.425,0.45 and voted five submissions using 1:1:1:1:1 weights.
we can get value of cosine inner product output from arcface output and used it for reranking, but it doesn't help much. 

## What didn't work
Convnext, efficinentnetv2, swin transformer(384size) didn't work well for us. 


##Regret
We missed backfin while most of the competitors use backfin.
Many reports that using backfin is crucial for this competition.
We used a little bit of a backfin, but it didn't seem so good, so we moved on right away.  


We did our best while we were very busy with work and school, but we are sad to receive the silver medal. I will continue to make a lot of effort.  


Thank you for reading this!
