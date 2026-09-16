# 15th solution (wavenet only)

Competition: liverpool-ion-switching
Rank: #15
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153727

I finished my best solution(I thought)with unexpectedly good score in last day , but no time to finetune.....


The following model got the best score for me but it is not "beautiful" :(.

               
OS: windows 10
                                         
framework : tensorflow 1.4/keras 2.2
                                         
dataset : 'clean-kalman/train_clean_kalman.csv' and remove batch8
          
model:modified wavenet based on 
https://www.kaggle.com/siavrez/wavenet-keras
              
optimizer : adam (adam &gt; ranger &gt; sgd)
                                                                       
loss : bce + weighted macro_double_soft_f1 (bce &gt; ce = focal loss &gt;macro f1)
         
batch : 16
                  
feature : **rfc /lag_with_pct 123/ roll_stats 5 /gradient1-4**   added in the beginning
          **abs(sig)  , exp(sig) , sig^^2**  added in model layer (after augmentation)
           
augmentation : flip / gausision noise +-0.0075/ random shift 5
            
post processing : flip TTA+ Group 5 fold averaging
    
*key in early stage without rfc feature : use batch 1-4

*key2 : train with low lr and without earlystop 
              reduce_lr = ReduceLROnPlateau(min_delta = 0.000001,cooldown = 3,factor=0.3, patience=15, min_lr=1e-6, verbose=2)
  
  
early_stopping = EarlyStopping(patience=1000, verbose=2)
   
*key3 : loss function
