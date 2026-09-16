# 42th solution: BPE + LSTM-Transformer-CNN + DistillRoberta + Superfast Unsupervised

Competition: llm-detect-ai-generated-text
Rank: #42
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470290

Here is my simple solution:

- Model1: Public 963 with some postprocessing (public LB: 0.963, private LB: 0.894)
Base on predicted probability convert top 2% to 1 and bottom 10% to 0

- Model2: LSTM-Transformer-CNN model (public LB: 0.927, private LB: 0.788)

- Model3: DistillRoberta model (public LB: 0.927, private LB: 0.884)

- Model4: https://www.kaggle.com/code/aerdem4/daigt-superfast-unsupervised-baseline (Model 4.1) + Model 3 (public LB: 0.956, private LB: 0.914, with some postprocessing can get private LB 0.917):
Use the top 9% from model 3 prediction as AI likely part of model 4.1, use the bottom 45% as the student likely part of model 4.1. Then search the top 20 student likely and AI likely to calculate the final probability.

- Final result: 0.552*Model1+ 0.08*Model2 + 0.184 Model3 + 0.184 Model4 (public LB 0.967, private LB: 0.918)
