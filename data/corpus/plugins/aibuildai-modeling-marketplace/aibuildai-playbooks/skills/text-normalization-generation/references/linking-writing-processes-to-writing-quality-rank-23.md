# 23th Place Solution

Competition: linking-writing-processes-to-writing-quality
Rank: #23
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466771

Whoa, I was trusting the CV and expecting a shake-up but not an upper-silver. 😅

You can check the code from here: https://github.com/nlztrk/Linking-Writing-Processes-to-Writing-Quality/

Here are the things I've basically extracted and used:

**Text Features**
- Pre-processed the extracted essay and deleted recurring tabs, spaces and newlines
- Word, sentence and paraghraph based statistical aggregations
- Punctuation statistics
- Punctuation mistake statistics (dot without a space after, comma with a space before, newline without a punctuation etc.)

**Event Log Features**
- Count and nunique features for events
- Statistical aggregations for all numerical columns
- Pause-time aggregations (for all text, pause between words, sentences and paragraphs, seperately.)
- P and R burst time and key-press count features

**Training**
- Used 10-Fold Stratified
- Used CatBoost, LGBM and XGBoost
- Hyperparameter-tuned all models with using Optuna
- Tracked OOF
- Ensembled final models using Optuna-OOF-tuned weights

I am happy that my efforts weren't for nothing. I congratulate all of you!
