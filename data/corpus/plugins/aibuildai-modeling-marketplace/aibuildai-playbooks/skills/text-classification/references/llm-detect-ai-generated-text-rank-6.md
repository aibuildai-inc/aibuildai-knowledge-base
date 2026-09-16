# 6nd place solution with code

Competition: llm-detect-ai-generated-text
Rank: #6
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/471831

Many thanks to Kaggle and the organizers for creating the competition.


Link to training and inference code: https://www.kaggle.com/code/davidecozzolino/coder-one2 
Link to github repository: https://github.com/davin11/entropy-based-text-detector
Link to model summary documnt: https://github.com/davin11/entropy-based-text-detector/blob/main/Documentation.pdf


Solution:
1. A pre-trained Large Language Model (LLM) is used to compute entropy-based synthetic features.
2. Starting from feature vectors of few elements, a One-Class SVM is trained using only the human-written essays provided by the organizers as training-set.

Note:
- I used  [DAIGT-V4-TRAIN-DATASET](uhttps://www.kaggle.com/datasets/thedrcat/daigt-v2-train-datasetrl) to select the best features.
- I tried different LLMs; phi-2 proved to be the best
