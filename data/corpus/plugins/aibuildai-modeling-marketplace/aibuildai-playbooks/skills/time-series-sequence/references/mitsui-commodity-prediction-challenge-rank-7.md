# Tatru in MITSUI&CO. Commodity Prediction Challenge

Competition: mitsui-commodity-prediction-challenge
Rank: #7
Source: https://www.kaggle.com/c/mitsui-commodity-prediction-challenge/writeups/tatru-in-mitsui-and-co-commodity-prediction

# Model Summary

## A1. Background on you/your team
- **Competition Name:** MITSUI&CO. Commodity Prediction Challenge 
- **Team Name:** Tatru  
- **Private Leaderboard Score:**  0.506
- **Private Leaderboard Place:** 7th

**Team Members:**  
- Name:  Alejandro Sierra Serna
- Location:  Dopiewiec, Wielkopolska, Poland
- Email:  asierraserna@gmail.com

## A2. Team Background
- Academic/professional background: Industrial Engineer, SAP Sales and Distribution Consultant, Specialist in Marketing Management, Developer in SAP, Cloud Computing, Open Source web, React NAtive for mobile apps and Data Science.  
- Prior experience relevant to this competition: 
  - Experience in data analysis and predictive modeling  using machine learning techniques to determine optimal prices for B2B negotiations in Manufacturing industries and Sales using SAP Sales data.
- Motivation for entering:  Exploring new challenges in the field of data science and applying my skills to real-world problems.
- Time spent on the competition:  10 days exploring, developing the model and doing hyperparameter tuning.
- Teaming up (if applicable):  just me for this competition.
- Roles and contributions:  Sole contributor responsible for data pre-processing, feature engineering, model development, and evaluation. I used it to teach my brother some basics of ML and experimentation.

## A3. Summary
The modeling process began with thorough data exploration and the development of a minimal baseline model to ensure the pipeline and submission process worked as expected. After establishing this foundation, I implemented a comprehensive hyperparameter optimization workflow, comparing three main strategies: XGBoost (for robust tabular modeling), TensorFlow deep neural networks (for capturing complex patterns), and a weighted ensemble that combined the strengths of both approaches. Three main submissions were made: one with XGBoost alone, one with TensorFlow alone, and one with the ensemble. While XGBoost performed best on local validation, the TensorFlow model alone achieved the best results on the actual Kaggle leaderboard, likely due to better generalization. Feature engineering and preprocessing were carefully designed to handle missing values, scale features, and create time-aware splits. The final model selection was based on validation performance, with the best-performing configuration submitted for the competition. All experiments and model training were conducted using Python, leveraging libraries such as XGBoost, TensorFlow, Optuna, and MLflow for tracking and reproducibility.

## A4. Feature Selection / Engineering
- Most important features (list or plot):  
  - Features related to commodity futures (e.g., LME metals), FX exchange rates, and US stock prices were among the most predictive. Technical indicators such as moving averages, volatility, and cross-asset correlations also ranked highly in feature importance analyses (see feature importance plots in the EDA notebook).
- How features were selected:  
  - Initial selection was based on domain knowledge and exploratory data analysis, grouping features by asset type and relevance. Feature importance from XGBoost and permutation importance were used to refine the set, focusing on those with the highest impact on validation performance.
- Important feature transformations:  
  - Created technical indicators (moving averages, volatility measures), market regime indicators (trend, momentum), and lagged features to capture temporal dependencies. Features were scaled for neural network models using robust scaling.
- Interesting feature interactions:  
  - Notable interactions were observed between commodity prices and FX rates, as well as between different commodity groups. Cross-asset correlations and lagged target values provided additional predictive power.
- Use of external data (if any):  
  - No external data was used; all features were derived from the provided competition datasets and engineered features.

## A5. Training Method(s)
- Training methods used:  
  - The main training methods were XGBoost (for robust, interpretable modeling of tabular data) and TensorFlow deep neural networks (for capturing complex, non-linear relationships). Both models were tuned using Optuna for hyperparameter optimization, with MLflow and TensorBoard for experiment tracking. Three main submissions were made: XGBoost alone, TensorFlow alone, and an ensemble of both.
- Ensemble methods (if any):  
  - A weighted ensemble was constructed to combine the predictions of the XGBoost and TensorFlow models, leveraging their complementary strengths.
- Model weighting (if ensemble):  
  - The ensemble weight was optimized as a hyperparameter, with the final configuration assigning approximately 0.68 to XGBoost and 0.32 to TensorFlow, based on validation performance.

## A6. Interesting Findings
- Most important trick used:  
  - Joint hyperparameter optimization of both model types and the ensemble weight, along with careful feature engineering and robust validation splits, significantly improved performance. Notably, while XGBoost was the best model on local validation and test data, the TensorFlow model alone achieved the highest score on the private leaderboard, suggesting it generalized better to the hidden test set.
- What set your solution apart:  
  - The use of both XGBoost and TensorFlow, combined with a systematic ensemble and thorough experiment tracking, allowed for flexible model selection and robust results. Automated tuning and early stopping helped avoid overfitting. The competition leaderboard was challenging to interpret during the event due to the presence of suspicious or "fake" submissions that exploited the evaluation API, making it difficult to assess true model performance until the final results were revealed.
- Other interesting data relationships:  
  - Strong relationships were observed between commodity prices and FX rates, and lagged target values provided valuable predictive signals.

## A7. Simple Features and Methods
- Subset of features for 90-95% performance:  
  - A reduced set of top 10-15 features (mainly key commodity prices, FX rates, and technical indicators) with XGBoost achieved close to 90% of the full model’s performance.
- Most important model:  
  - XGBoost was the most effective single model for the simplified approach on local validation, but the competition results showed that TensorFlow could generalize better in some cases.
- Simplified model score:  
  - The simplified model achieved approximately 90-92% of the full ensemble’s validation score (see notebook for exact metrics).

## A8. Model Execution Time
- Training time (full model):  
  - Full training (including hyperparameter tuning) took several hours on a standard GPU instance; final model retraining on all data took 1-2 hours for XGBoost and 1-2 hours for TensorFlow.
- Prediction time (full model):  
  - Generating predictions for the test set took a few minutes for each model.
- Training time (simplified model):  
  - The simplified XGBoost model trained in under 10 minutes.
- Prediction time (simplified model):  
  - Predictions with the simplified model were generated in under a minute.

## A9. References
Main references and resources:
- Kaggle discussion boards for competition rules, submission format, and troubleshooting
- Large Language Models (LLMs) for research, code review, and brainstorming, including:
  - Gemini 2.5 and 3 PRO
  - Claude Sonnet 4.5
  - GPT-4.1
- EITCA AI Academy (https://eitca.org/eitca-ai-artificial-intelligence-academy/) for foundational AI and ML knowledge
- SuperDataScience podcast, especially the XGBoost episode with Matt Harrison (https://www.superdatascience.com/podcast/sds-681-xgboost-the-ultimate-classifier-with-matt-harrison)
- Lex Fridman podcasts on Machine Learning and AI
- YouTube video courses and playlists:
  - Sentdex Machine Learning with Python: https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
  - Sentdex Deep Learning with Python: https://www.youtube.com/watch?v=BzcBsTou0C0&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh
  - Sentdex Neural Networks from Scratch: https://www.youtube.com/watch?v=nLw1RNvfElg&list=PLQVvvaa0QuDfSfqQuee6K8opKtZsh7sA9
- Official documentation for:
  - XGBoost: https://xgboost.readthedocs.io/
  - TensorFlow: https://www.tensorflow.org/
  - Optuna: https://optuna.org/
  - MLflow: https://mlflow.org/
- Stack Overflow and various blog posts for Python, pandas, and scikit-learn usage
