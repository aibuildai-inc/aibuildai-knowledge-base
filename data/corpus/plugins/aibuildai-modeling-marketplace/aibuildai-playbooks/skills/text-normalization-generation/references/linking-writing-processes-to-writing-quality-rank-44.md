# [48th place solution] Multi-label K-Fold for CV Strategy

Competition: linking-writing-processes-to-writing-quality
Rank: #44
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466965

Thanks to @kaggle and The Learning Agency Lab for this amazing competition. Also thanks for kagglers who made amazing public notebooks.

Though I didn't do extensive research to prove my assumption, I think using Multilabel Stratified K-Fold [https://github.com/trent-b/iterative-stratification for](multilabel-k-fold) to make my CV strategy more robust and helped me to make a more stable solution.

**Features**

Besides the public notebook features, I used TF-IDF Vectorizer (char_wb) and Count Vectorizer (only char) to extract more features from the reconstructed essay. But the downside is this makes the feature engineering process more slow.

**CV Strategy**
I used other important features as additional labels to the score. I selected these features based on EDA and Linear Correlation Analysis.
[cv-strategy]

**Modeling**
For the modeling part, I used 6-seed 10-fold LGBM for training and averaging model results for the final LGBM prediction part.

**Weighted Sum with Public Models**

Last Prediction = 0.65 * LGBM + 0.35 * Denselight Prediction (from Public Notebook)

This pipeline scores:
- Public LB 0.576 (48th place)
- Private LB 0.567 (48th place)

**What didn't work**
- Encoder-based transformer model (used only reconstructed essays)
- LSTM with Attention mechanism (used reconstructed essay with features)
- Feature Selection Network
- K-Best Feature Selection
