# [14th Place Notes]  Image Agumentation + Domain Adaptation + ABMIL

Competition: UBC-OCEAN
Rank: #14
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465476

This was my first Kaggle competition, an enjoyable and educational journey. 
Here are some key takeaways from my experience:

# ==================================================

#  Basic structure of my model

1 backbone **(densenet201)** for instance-level feature extraction >> 2 **ABMIL** models (TMA and WSI separated) for bag-level classification

# ==================================================

# Training procedure

- Use the **152 WSI masks** to extract tiles whose types I'm certain of for **backbone training.**
- **Lock the backbone** for feature extraction, applying **MIL training** on **513 WSI dataset** with **a light attantion model that has its own classifier**. The bag classifier's params are inherited from the backbone and is applied with a low learning rate during MIL training.
- Use TMA samples  for **unsupervised domain adaptation** training and monitoring the model's perfomance on TMA during the whole train process.

# ==================================================

# Inference procedure

- All images are patched into 224×224 tiles: 
WSI is scaled down by 0.5, TMA is scaled down by 0.25;
WSI is patched in grid of 224, TMA is patched in grid of 120;
the maximum number of tiles for one bag is set at 512 (this is for large WSIs);
- A single backbone is shared for feature extraction of both WSIs and TMAs. It also performs instance-level prediction, and only tiles that are classified as cancer types will be sent to attention models. (heathy or dead tiles are eliminated)
- Two attention models dedicated to WSI and TMA separately transform features of tiles into one bag label, a softmax confidence threshold of 0.4 is used to re-label low-confidence predicitons as "Other". 

# ==================================================

# Breakthroughs during exploration

## Image Augmentation

Learning that both TMA and WSI can vary significantly in staining, color, and clarity, I implemented extensive augmentation techniques. This included custom tools like using a circular mask to make WSI tiles resemble TMA more closely. Based on my LB perfomrance, I think Image Augmentation is a critical step for improving the models' genralization ability on TMA. 



##  Domain Adaption

Still, the model trained on WSI tiles performs worse than my expection on TMA. So I used **domain adversial training techniques**, from classic **DANN**, **heurstic domain adaptaion**, to **toAlign**. The key idea is to **use the limited TMA images to help the model extract more task-related and less domain-related features without revealing their labels**. This is the second and most critical step for my score boost on LB.

During training, I use TMA acc to actively monitoring the model's transfering ability on TMA:


***related resources:***
Domain-Adversarial Training of Neural Networks: https://arxiv.org/abs/1505.07818
Heuristic Domain Adaptation: https://arxiv.org/abs/2011.14540
ToAlign: Task-oriented Alignment for Unsupervised Domain Adaptation: https://arxiv.org/abs/2106.10812

## AB-MIL (Attention-based Deep Multiple Instance Learning)

I used **the most basic attention-based MIL model** with **a self-attention kernel** whose impact I'm unclear of. Based on my observation, if the feature extractor is trained well, the model performed adequately with basic MIL (max/mean scoring) on public LB. However, AB-MIL offered much better accuracy on my local valid dataset, therefore theoretically more stable and superior performance.

***related resources:***
Attention-based Deep Multiple Instance Learning: https://arxiv.org/abs/1802.04712
Kernel Self-Attention in Deep Multiple Instance Learning: https://arxiv.org/abs/2005.12991

# ==================================================

# Approaches that I found not quite useful

- **Feature augmentation**
I aimed to further narrow the gap between WSI and TMA after applying various image augmentation methods. Attempting to add noise directly to the extracted features, however, proved ineffective.
***related resources:***
A Simple Feature Augmentation for Domain Generalization: https://openaccess.thecvf.com/content/ICCV2021/papers/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.pdf
- **Switch backbones**
I tried different types of backbones, from classic reset to popular efficientnet, None of them perfomed better than densenet with my pipeline. 
- **Traditional anomaly detection techniques for outlier detection**
I tried Isolation Forest, DBSCAN on the features I extracted and found that these methods couldn't even tell the existing cancer types apart. I soon releazed that there was no way these methods could surpass my specially trained classifiers. To me this is an absoulte wrong path.
# ==================================================

# Potential improvements in the future

- Fundimental training techniques:
Label Smooth
Mixup and CutMix
CV for the best backbone+ABMIL combination
Median averaging for basic MIL approach instead of max/mean
Use BCEWithLogitsLoss. Unlike CrossEntropy, it computes independently for each label.
Train in 16-bit float to increase speed, usually doesn't hurt performance
- Smarter ways to disthiguish outlier:
Use Entropy to thresholding
Predict bottom 5 or 10 percentile scores as Others
Train directly with external other cancer types
synthesize images with exsiting types as "Other" data for training (I doubt its validity, but seems to work as well)
- I didn't use Model Ensemble at all. There are multiple ways to use ensemble:
Ensemble of backbones of the same structure but trained on different scales
Ensemble of backbones of different strutrues
Ensemble of backbones (like ConvNext, HoRNet, EfficientNetV1, and EfficientNetV2) predicts different sets of labels for the same input, using sigmoid activations to combine / compare  independent probabilities across models
Ensemble of attention models of different structures
- Despite my unsuccessful tests with different backbones, many teams with top LB scores credited models specifically trained on Pathology dataset, which I think should be of vital help:
CTransPath ( https://github.com/Xiyue-Wang/TransPath )
LunitDINO ( https://github.com/lunit-io/benchmark-ssl-pathology )
iBOT-ViT ( https://github.com/owkin/HistoSSLscaling )
- Try more sophisticated MIL attention models:
DTFD-MIL ( https://arxiv.org/abs/2203.12081 )
TransMIL ( https://arxiv.org/abs/2106.00908 )
CLAM ( https://github.com/mahmoodlab/CLAM )
DSMIL ( https://github.com/binli123/dsmil-wsi )
Perceiver ( https://github.com/cgtuebingen/DualQueryMIL )

# ==================================================

My notebook link: 
https://www.kaggle.com/code/yannan90/ubc-submit-att
