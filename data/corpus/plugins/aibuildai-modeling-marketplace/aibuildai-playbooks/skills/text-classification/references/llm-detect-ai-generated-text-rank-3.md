# 3rd place solution

Competition: llm-detect-ai-generated-text
Rank: #3
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470333

Many thanks to the organizers for creating the competition.

Our solution is a weighted average of tfidf pipeline and 12 deberta-v3-large models.

### Transformers Ensemble

As a preprocessing step, we used the deobfuscator shared by @sorokin ([post](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/457819)), but we corrected only texts that had more than 15 errors. Also, we removed symbols that were not in the original train set and normalized the encodings of the remaining symbols.

4 models were trained on 11k selected generated/rephrased (essay-level and sentence-level)/partially rephrased essays; some of the essays are from shared datasets, and some are custom-generated using several LLMs. 
We selected training samples using the following algorithm:
* Train the initial model using @alejopaullier [data](https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset)
* At each iteration, add samples that the previous model failed to predict correctly - 500 human-written and 500 generated, with the highest distance from the true label.
* Train a new model and repeat again

We evaluated each 4-th iteration on an LB. Once LB stopped improving we took the previous best dataset. A best single model trained on this data has a 0.927 public and 0.845 private score.

Inspired by @jsday96 [post](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882) we generated continuation for pile and slimpajama datasets. We filtered out text that was too short/too long, contained code or math, non-English text, and had a high non-letters/letters ratio. Then we used ~35 different open-source models with different combinations of parameters. We split sampling parameters into 3 scenarios depending on the temperature value and used random values for top_p/min_p and presence_penalty/frequency_penalty within bounds specified for each scenario. 
We've trained 3 models using 500k, 1m, and 1.2m samples generated this way. All models were trained with default hyperparameters, max length 256 (1512 for inference), and high batch size - 48. The best single model trained with ~1m samples and has a 0.956 public and 0.967 private score.

We also finetuned 5 models on the selected 11k dataset (weights are from the models trained on 500k+). The public LB for these models was slightly higher, but private worser by ~0.005. 

### Tfidf Pipeline

We took one of the earliest public notebooks ([link](https://www.kaggle.com/code/hubert101/0-960-phrases-are-keys?scriptVersionId=153589869)) and made a few adjustments. 
* Increased catboost and lightgbm number of iterations by 250, and used weights=[0.05, 0.225, 0.225, 0.5] for voting classifier
* Added 1k pseudo from the test set to @thedrcat [dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)  - only samples in which the ensemble of transformers was most confident (probabilities lower than 0.01 or higher than 0.99)

With these changes, the public score remained the same, but the private increased from 0.893 to 0.927. 
Since it was a little gambling game, we selected both - the initial pipeline and the adjusted one, they have 0.970 and 0.974 private scores respectively.

### Final Ensemble

We used a weighted average ensemble on probabilities in two steps:
* Firstly, we weighted tfidf and models trained on the 11k dataset - only the samples there transformers predictions were lower than 0.1 or higher than 0.9; for samples in the middle we used just tfidf probs
* Secondly, we used weighted averages without any conditions for step 1 and models trained on large datasets. 

Averaging this way improved both private/public LB and local CV (but it was unreliable though).


### Postprocessing

For each prompt_id, if the number of samples there greater than 1000, we fitted umap on tfidfs (the same as in tfidf-catboost pipeline, but per-prompt), calculated distance to 7 closest human-written and 7 generated samples, and scaled predictions by the ratio human_distance / generated_distance with clipping to (0.9, 1.1). It slightly improved public and private LB.

### Acknowledgements

I want to say thank you to everyone who shared their ideas/assumptions/datasets. Especially @evilpsycho42 for your great work during this competition. 

### Links

Inference: https://www.kaggle.com/code/evgeniimaslov2/llm-daig-3rd-place-solution?scriptVersionId=160663257
Training: https://www.kaggle.com/datasets/evgeniimaslov2/llm-daig-src-code
