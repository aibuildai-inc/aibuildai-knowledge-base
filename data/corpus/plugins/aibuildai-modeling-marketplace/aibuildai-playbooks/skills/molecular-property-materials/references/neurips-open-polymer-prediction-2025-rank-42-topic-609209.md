# PB: 42nd - LB: 30th Place Solution [LB: 0.060, PB: 0.087]

Competition: neurips-open-polymer-prediction-2025
Rank: #42
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/pb-42nd-lb-30th-place-solution-lb-0-060-pb

# **Context**
 - Business Context: [Competition Page](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
 - Data Context: [Dataset](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data)
<br>
***
  
# **Overview of the approach**
Our final model was a weighted mean ensemble of three models:
- **ChemBERTa-based regression model** (PB/LB = 0.088 / 0.067)
- **GNN-GREA ensemble model** (PB/LB = 0.093 / 0.067)
- **CatBoost** (PB/LB = 0.092 / 0.065)

Through experments, I found that the ensemble weights **(BERT = 0.33, GNN-GREA = 0.24, CatBoost = 0.43)** minimized the LB score to **0.065**
<br>
***


# **Details of the model** 
## **Inspiration**
At the beginning, I took inspiration from the 1st place solution of the *EUOS/SLAS Joint Challenge: Compound Solubility*.
- [Writeup](https://www.kaggle.com/competitions/euos-slas/writeups/olab-olab-solution)
- [Solution PDF](https://storage.googleapis.com/kaggle-forum-message-attachments/2124821/18648/olab-solution.pdf)


(The photo was cited from the PDF.)
I aimed to implement a similar idea by concatenating embeddings and predicting with an MLP.
<br>


## **GNN-GREA**
Based on the implementation in [torch-molecule repository](https://github.com/liugangcode/torch-molecule/blob/main/examples/prediction/plym_predict.py), I created training/inference pipeline and slightly modified the library to output hidden state vectors.
- Trained with **batch size = 512, 250 epochs**
- Performed hyperparameter optimization (60 trials)
- Both GNN and GREA individually achieved LB = 0.073
- Averaging them reduced the LB to **0.067**

Code: [GREA+GNN  ensemble](https://www.kaggle.com/code/nakanishiwataru/pb-0-093-lb-0-067-grea-gnn-ensemble-model)
<br>


## **ChemBERTa based model**
Based on the Implementation [reference](https://www.kaggle.com/code/defdet/polymer-bert-train), I created [code](https://www.kaggle.com/code/nakanishiwataru/pb-0-088-lb-0-067-2gpu-chemberta-train), training ChemBERTa with 2GPU.
- Training setup: 2GPUs(T4), batch size = 64, learning rate = 0.001, epochs = 80
- Trained separately for each property
- ChemBERTa achieved LB = **0.067**

Code: [ChemBERTa Inference](https://www.kaggle.com/code/nakanishiwataru/pb-0-088-lb-0-067-chemberta-inference)
<br>


## **CatBoost**
I used the [baseline](https://www.kaggle.com/code/yusuketogashi/lb-0-66-no-datasets-no-problem), which employed Mordred descriptors as features.  
I found that adding the following fingerprints improved the LB score from **0.068 to 0.065**:
- MACCS keys
- Morgan fingerprints
- RDKit fingerprints
- Atom pair fingerprints
<br>


## **Ensemble  Model**
Initially, I tried concatenating hidden state vectors from BERT and GNN, followed by training an MLP ([code here]
(https://www.kaggle.com/code/nakanishiwataru/lb0-066-gnn-bert-hvec-concat-nn)), which achieved LB = 0.066 but did not outperform.

Instead, I tried another idea, weighted average ensemble of three different model types (ChemBERTa, GNN+GREA, CatBoost).
After multiple experiments, the optimal weights were **(BERT = 0.33, GNN-GREA = 0.24, CatBoost = 0.43)**,  achieving the best LB = **0.060**.

Code: [BERT GREA-GNN CatBoost ensemble](https://www.kaggle.com/code/nakanishiwataru/pb-0-087-lb-0-060-bert-grea-gnn-gbdt-ensemble)
<br> 


## **Data Augmentation**
I used widely shared augmentation techniques throuout the competition:
- **Standardization -> enumeration, kekulization, stereoenumeration**
For ChemBERTa, augmentation was applied during both training and inference.
For GNN and GREA, augmentation was applied only during inference (TTA). which improved LB by about **0.001**
<br>


## **Post Processing**
No post-processing was applied.
<br>
***



# **Wat Did Not Work**
- **Hyperparameter optimization (Optuna) for decision trees:**
 Optimizing depth and n_estimators for CatBoost did not significantly improve LB.
- **Feature engineering for decision trees:**
Adding fingerprints to Mordred descriptors improved performance, but PCA-based dimensionality reduction actually worsened the score.
- **Other tree models:**
Unlike many top solutions, XGBoost and LightGBM did not work well in my case. TabTransformer/TabPFNRegressor also did not perform well and consumed too much memory, so I excluded them from the final ensemble.
<br>
***


# **Summary**
This was my second Kaggle competition. One key factor for success was leveraging ideas from previous top solutions in similar competitions.

As widely discussed, there was a distribution shift between public and private test data, which led to significant shake-ups and downs in the leaderboard. Interestingly, many top solutions applied post-processing, but since I did not, I expected a larger shake-down. However, my final position remained relatively stable, which was quite surprising.
