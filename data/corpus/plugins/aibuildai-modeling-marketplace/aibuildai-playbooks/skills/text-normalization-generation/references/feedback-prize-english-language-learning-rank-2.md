# 2nd solution (back-translation & rank-loss)

Competition: feedback-prize-english-language-learning
Rank: #2
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369369

This is a fun game, thanks for the organizer and all pepole who share here. 
- CV  
I used abiheshark's cv strategy with 5 folds and seed 42.
My pb best choosen one is with **pseudo train and optuna tune CV 44494**  PB 43363
(but this **cv is not accurate** and over optimized since optuna tune for all train OOF, I should have used OOF agian for optuna based CV, unfortuantely I trust LB here and it happen to improve my LB but hurt PB a little or just by chance)  
My pb best one is without pseudo train and no optuna tune(hand tune rule by per model per target cv) **CV 44543**  PB 433541
But after game I tested pseudo+hand tune rule the results is 43380 so hand rule is also not very stable and has chance to overfit as you tune weight on local OOF cv.
TODO: Try Rige, lasso, hill clmb，Nelder-Mead, negative weight of model methods to improve ensemble results
**Best single modelsingle model is cv around 449** without pseudo train, using backtrans pretrain or feedback2 pretrain. 
**Best single model with pseudo is cv 4469** , for late submission I find better one with cv **4456** pb **434726** see table below 


Notice above image, epoch n means first(n + 1) models mean. 

Single model performance (5folds + 1 full train):
| model | cv  | lb | pb | 
| --- | --- | -- | -- | 
| base(dev3-large maxlen1280) | 4514  |  442013 |  438818 |  
| base+rank_loss | 4505 | 438912 | 437175  | 
| base+rank_loss+trans-nl pretrain | **4498** | 440225 | **435711**  |
| base+rank_loss+feedback2 pretrain | **4488**  | **438583**|  **435814** | 
| base+feedback1 pseudo train only | **4469** | **438601** |  **436144** | 
| base+rank_loss+feedback1 pseudo pretrain | **4497**  | 440576 | **435144** | 
| base+feedback1 pseudo&feedback3 train | **4468**  | **438063** | **434809** | 
| base+rank_loss+feedback1 pseudo&feedback3 train | **4456**  | **438084** | **434726** | 

Model with pretrain or pseudo or back-trans(**model with more backgound knowlege**) work better for both CV/PB and some models for CV/LB/PB.And these models tend to have better cv performance on vocabulary.
Rank loss always help.

- Backbones
deberta-v3-large  (lr 2e-5)
deberta-v3-base  (lr 5e-5)
Only the two work, all others much worse then deberta-v3-large
TODO: I should have add **SVR method ensemble**
SVR single model reach CV around 450 so good enough to ensemble and it could introduce more knowlege of non-deberta backbones.
I did not add it as I return to this game late and could not make it safe to add for infer, there are some OOM issues , local cv seems to improve like 445146 to 44496 without tune weight for each target, using this one https://www.kaggle.com/code/quangphm/lb-0-43-simple-ensemble-deberta-base-svr.
- Train
Text maxlen 1280, batch size 8, 4 epochs, linear lr decay, awp training for the last 2 epochs  
For back translation pretrain only 2 epochs train without awp.  

- Models
I submit 23 models * 2, 23 full train model, 23 fold(0-4) models. Cost 2h20 min using T4 * 2 infer.  

- What works
1. **Let bert deal with the target relation**
"[CLS] cohesion syntax vocabulary phraseology grammar conventions text [SEP]“  
Each target output emb with 1 unique fc added to predict target value. 
2. **Add pearson loss**
We could see bert tend to **predict all target with similar value**, in order to predict well for **hard target**, I found adding pearson loss help a lot. But it will hurt for easy target performance,  I did not find good method to deal, ensembling with person loss rate 0.1 and 0 is what I used.    
Also with rank_loss_rate = 0.1 improve single model cv from 4514 to 4505.  Below show add rank loss (name with -crank) help a lot for hard targets(score/max calc for target index np.argmax(label - label.mean())). Here also has improve space, I think maybe we can better ensemble using some specific features and do regression on OOF. 

3. **Use back translation for pretrain**
I used a lot of back translation for ensembling. They are [nl, fr, de, pt, af, cn, ru, fi, sv, ja, ko, el, hr, cy]    
Notice back translation data is only used for pretrain, so 2 steps training, the second step is the same as other models.
No back translation models:  
('score', 0.44672287515754533),
 ('score/vocabulary', 0.4109792609030708),
Add back translation models, with weight=np.array([1,1,10,1,0.25,0.25], float) * 0.8:
('score', 0.4453322618856002),
 ('score/vocabulary', 0.4086610628919421),
Back translation models only:
('score', 0.44640808445875324),
 ('score/vocabulary', 0.408889105739513),
Vocabulary improve a lot ! Actually for all backtrans models vocabulary improve so it means backtrans help on learning more vocabulary info!
Single model (see image above with name .trans-nl)  cv improve a lot 4505 to 4497 and also PB gain.
4. **Use feedback2 data for pretrain**
Just train feedback2 listwise model (remove any discourse type info) as pretrain model.  This will make model more stable and help especially for some targets like convetions and vocabulary. 
TODO: I did not use ensemble and pseudo for feedback2 data pretrain, simply using 1 single full train model on fddeback2 data only, so here might have improvement space.  
Single model (see image above with name .ft2) cv improve a lot 4505 to 4488 and also LB/PB improve.
Why use feedback2? I did not investigate but I think discourse effectiveness has relation with our 6 targets.
5. **Pseudo**
Pseudo works, pseudo+optuna is much better then no pseudo+optuna, both LB and cv and PB 433811->43363.  
Pseudo also helped produce best PB(435144) single model.  
TODO: I did not try (Mean of Train PL / Actual Labelsas) mentioned in  [1st solution](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457) it might give boost.
6. **Tune for each target**
For example back translation help a lot for vocabulary but hurt conventions as show on cv image above.    

 - Code 
https://www.kaggle.com/code/goldenlock/best-pb-online-1127?scriptVersionId=112354741  (ensemble infer, best personal PB)
https://www.kaggle.com/code/goldenlock/pb-433630-pseudo-optuna-tune (ensemble infer, selected PB)
https://www.kaggle.com/goldenlock/single-model-infer (single model infer, 5 fold + 1 full train=6 models, 20min)
https://www.kaggle.com/datasets/goldenlock/feedbacken  (train)
https://github.com/chenghuige/Feedback-Prize---English-Language-Learning (full code)
