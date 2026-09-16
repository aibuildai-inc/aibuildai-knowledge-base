# 3rd place solution

Competition: sartorius-cell-instance-segmentation
Rank: #3
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298021

Thank you very much to Kaggle and the organizers of this competition and more specifically to @christoffersartorius, the entire Sartorius team and the brave data annotators that allow us to do countless tensor multiplications on GPUs with their training data.

Congratulations also to all the participating teams and to the 1st and 2nd place teams for their great achievement.

I also thank my teammates @slawekbiel  @rytisva88 and @cpmp that were absolutely amazing during this competition. I learned a lot from all of you in the past intense weeks.

**Here is the general schema of our ensemble solution:**


The two Mask-RCNN approaches are extended versions of @slawekbiel great serie of notebooks using Detectron2:
<a href="https://www.kaggle.com/slawekbiel/positive-score-with-detectron-1-3-input-data">1) Inputs</a> ==> <a href="https://www.kaggle.com/slawekbiel/positive-score-with-detectron-2-3-training">2) Training</a> ==> <a href="https://www.kaggle.com/slawekbiel/positive-score-with-detectron-3-3-inference">3) Inference</a>
 
**1) Mask-RCNN models:**
- Resnest200-Mask-RCNN (https://github.com/chongruo/detectron2-ResNeSt)
- Weight initialization from Livecell pretraining (https://github.com/sartorius-research/LIVECell)
- Multiple size + hflip training augmentations

**A) 3 classes models**
- 5 folds
- Multiple size inference

**B) Class specific models**
- 5 folds +- full data "fold" for each class
- TTA + WBF + NMS + Mask averaging
- TTA (box inference): Multiple size, Hflip + Vflip
- TTA (mask inference): Multiple size, Hflip 
- Class specific parameters and hyperparameters (anchors size, image sizes, amount of TTA, wbf IOU)




**2) Cellpose**
Slawek detailed his great Cellpose approach <a href="https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/297984" >here </a> which was blending very well with the Mask-RCNN models.

**Final week experiments:**

**Pseudolabels**
- Labels generated from the ensemble of models for the unsupervised data
- Models trained on merged pseudolabels + training data
- These models had quite variable results both on public LB and private LB

**Ensemble variations**
- Variable experiments in adding/removing number of models in ensemble, with the allowed 9h runtime, gave +- 0.002 variation both on public LB and private LB depending on configurations.

**Submission selection**
- To decrease correlation of final submissions, we choose one ensemble with some pseudolabels models and one ensemble without any pseudolabels models. Both submissions scored 0.354 on private LB.
