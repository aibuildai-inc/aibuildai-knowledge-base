# Our solution: 9th place

Competition: recursion-cellular-image-classification
Rank: #9
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110366

Dear all, thank you for excellent spirit of competition and astonishing results! Without such favorable pressure we wouldn’t be able to achieve such scores.

I will split description into two main parts: pipeline and prediction with post-processing.

**Pipeline:**
- **6-channel images.**
- **Super Resolution.** During exploratory data analysis we suddenly realize that the size of individual cells is rather small and convolutional layer might have a hard time learning all patterns. So we use simple bicubic superresolution technique. That really boost our scores. We checked up to 960 * 960. We have tried to use GAN’s (ESRGAN in particular) for this task but its performance was worse compared to vanilla bicubic interpolation.
- **3Fold by experiments.**
- **Add all controls to train.** We tried to add controls only from train experiments, but it didn’t work.
- **Mixed Precision:** Due to the large size of images we decided to use only Mixed Precision learning because of inevitable graphics card’s memory shortage.
- **Gradual layer unfreezing:** We found out that gradual layer unfreezing was crucial condition for our models not to diverge. First 5 epochs we unfreeze 20% of layers from head each epoch. Presumably it was because of Mixed Precision - some sort of gradient explosion or something similar.
- **Metric learning - CosFace loss:** We took my favorite metric learning loss. We also tried ArcLoss and AdaCos but they fail to converge or achieve worse results in Mixed precision mode.
- **Cyclic Linear Lr Annealing.**
- **Cyclic Linear Scale annealing:** This idea is inspired by AdaCos.  https://arxiv.org/abs/1905.00292 where authors decrease scale as model achieve high scores. S_max = 64, S_min = 24.
- **Classes for distinct cell types are distinct as well:** So we predict vector of 1108 * 4 + control classes. It  improved convergence drastically.


- **Unsupervised Domain Adaptation by Backpropagation:** https://arxiv.org/abs/1409.7495. We took plate and cell type as domain label and schedule gradient reversal layer coefficient as lr and Scale for CosFace. 


- **Backbones:** seresnet50 (img_size up to 960), senet154 (img_size up to 672), xception (img_size up to 860), inceptionresnetv2 (img_size up to 512), densenet161 (img_size up to 840), densenet201 (img_size up to 720). 
- SGD optimizer with momentum. We tried Adam, Lookahead and Ranger but they didn’t improve validation score.
- Gradient normalization.
- Batch size = 16 with accumulation up to 32.
- Focal loss: gamma = 32.


**Prediction:**
- **Prediction balancing.** This technique can be used when classes are equally balanced. Excellent implementation: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73803#latest-438270.
- **Progressive prediction.** Idea is simple: for every experiment each class can be presented only once. So if we predict some class with high probability there is a little chance that this class can be predicted further with lower probability. We take 50% most probable classes, remove them from consideration and predict the next sample and so on.
- **Leak.**


**What didn’t work:**
- Mix-up
- ArcFace
- AdaCos
- Deformable ConvNet - https://arxiv.org/pdf/1811.11168.pdf
- Deep Supervision
- Weight Standardisation
- Concatenation of two sites together to make 12-channel image
