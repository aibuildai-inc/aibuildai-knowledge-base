# 13th Place Solution Summary - CVSSP & forecom.ai (0.527 LB)

Competition: hpa-single-cell-image-classification
Rank: #13
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238333

We enjoyed the “Human Protein Atlas - Single Cell Classification” competition very much (even though we only joined half-way through). Many thanks to the organisers, Kaggle and all participants!

In the spirit of sharing with the Kaggle community, we would like to present an outline of our best submission, which we've visually summarized here:

[Block diagram]

**The key ideas used in our pipeline are:**
- Improved cell segmentation processing with reduced processing time
- Automated detection of mis-segmented cells around the borders using a trained classifier
- Ensemble of (4 or 7) cell-level CNN’s, including:
- **1.** EfficientB4 (300×300×4)
- **2.** ECA-NFNet-L1 (320×320×4)
- **3.** Swin Transformer (384×384×4)
- Use of transformer nets
- Test time augmentation on all cell-level classifiers, 4x
- Image level classifier (Swin Transformer)
- Class-dependent blending of the image-level and cell-level decisions

**In addition, the training was supported by**
- Train time augmentation, including rare-class augmentation  
- Focal loss and hybrid-loss training and remaining for batch
- AI-based Label Boosting
- Hand-labelling of some examples for rare classes

We will be presenting a more detailed write-up of our solution shortly!

\- Sameed, Dmitry, Eng-Jon, Mirek and Mikel (anokas).
