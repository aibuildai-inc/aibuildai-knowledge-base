# 11th Place Solution

Competition: jigsaw-agile-community-rules
Rank: #11
Source: https://www.kaggle.com/c/jigsaw-agile-community-rules/writeups/11th-place-solution

# Introduction
We (@takaito, @Muj!rush!, @Sankurero and @yukko08) would like to express our sincere gratitude to the organizers for such an outstanding competition! I also found many publicly available notebooks helpful. I'm grateful to the Kaggle community too!

# Our Solution

Our solution completely relies on train-on-test (online training) approach. We are training diverse models for various tasks. Like other teams, we introduced diversity by classifying using prompts combining elements like positive_example, negative_example, and body, or prompts using only the body, leveraging the four examples provided in the test. We incorporated contrastive learning and prioritized integrating as many diverse methods as possible. Furthermore, for the body-only classification task, pseudo labeling proved effective for both BERT and LLMs.
Below is an explanation of each part.

# BERT Sequence Classification (takaito Part)
My approach before the team merge involved training BERT-based models for body classification online and then ensembleing them. Even just ensembleing BERT-based text classification models achieved a public score of 0.926. 
A key optimization was using two GPUs to train two models in parallel online, allowing me to prepare many models.
Furthermore, since the selected models and hyperparameters could be rapidly validated using training data, I extensively explored various options.
It was surprising that the base model outperformed the large model in accuracy. Additionally, multilingual BERT proved to be very strong. Regarding hyperparameters, many models maintained high accuracy even when fixing layers close to the input, while also accelerating training, so I actively adopted this approach. Considering factors like xlarge achieving higher accuracy but slower speed, we tested various models. After team merging, we adopted only the most powerful one among them: gte-multilingual-base [1].
gte-multilingual-base (LB: 0.914, Time: 9m)
(Confirmed improvement by increasing training via late submission. LB: 0.915, Time: 12m)

# deberta-v3-base (Public Notebook)
https://www.kaggle.com/code/nahidhossainredom/deberta-v3-base-3-epochs-lb-0-906/notebook
Since the Muj!rush!&yukko&Sankurero team had been using it for ensemble even before the team merge, I kept it in the subs until the end.

# SentenceBERT (Muj!rush! Part)
We built rule-specific embedding models and performed online learning. Based on local validation, we adopted the following two model types, tuning hyperparameters per rule:
- bge-large-en-v1.5 [2]: Hyperparameters optimized to match the Public LB.
- gte-large-en-v1.5 [3]: Hyperparameters optimized for local validation scores.

For online learning, we used only the test data. Using each rule as the anchor, we conducted contrastive learning that pulls positive examples (pos example1/2) closer to the anchor and pushes negative examples (neg example1/2) farther from it. The loss function was CosineSimilarityLoss from SentenceTransformers.
At inference time, we used the trained models to obtain embedding vectors for each rule and body in the test data, and took the cosine similarity between the rule and body embeddings directly as the prediction score. Training and inference for all six rules complete in approximately 35 minutes.

# LLMs
jigsaw_pseudo_training_llama-3.2-3b-instruct)
https://www.kaggle.com/code/wasupandceacar/jigsaw-pseudo-training-llama-3-2-3b-instruct/notebook

- (Sankurero Part)
For the LLM implementation, we built upon the excellent public code by wasupandceacar (LB: 0.916). Using this implementation as the foundation, we experimented with multiple instruction-tuned language models — LLaMA-3.2-3B-Instruct [4], Qwen-3-4B-Instruct [5], LLaMA-3.2-1B-Instruct [6], and Gemma-2B-IT [7]. Based on the leaderboard results, we ultimately selected LLaMA-3.2-3B-Instruct and Qwen-3-4B-Instruct as our primary models.

- (takaito Part)
We introduced Test-Time Augmentation (TTA) by performing multiple inference passes with different combinations of positive and negative examples, instead of randomly selecting one pair.
In addition, we implemented a zero-shot inference mode by removing example data from the prompt. This configuration later proved effective in the subsequent pseudo-labeling stage.

# Pseudo Labeling (takaito Part)
We applied pseudo labeling (hard label) to both BERT and LLMs. For the test set bodies, we ensembled the current prediction values and assigned labels 1 or 0 to the top and bottom xx% of the data, respectively, to use them for additional training. Since LLMs take longer to train and tend to benefit from more training, we aimed to reduce training time by loading previously trained LoRA weights from the earlier stage.
In a post-submission analysis, we found that applying pseudo labeling improved the private score from 0.92579 to 0.92734.

# What Not Worked
- Few-shot Pseudo Labeling
We also experimented with few-shot pseudo labeling. However, since the amount of training data is important but the prompts are long, the training took considerable time, forcing us to limit the size of the training data. As a result, the accuracy improvement was not sufficient relative to the training time, so we decided not to adopt this approach.

- Use of models larger than 4B
We experimented with models such as llama3 8B[8], gemma2 9b[9], and qwen2.5 14B[10], but due to their large model size, they required too much inference time, so we decided against using them in the final solution.

- Continual Pretraining of LLMs
We hypothesized that the base LLMs had not been sufficiently exposed to the colloquial and sometimes impolite linguistic patterns characteristic of Reddit discourse. To address this, we collected approximately 38 GB of Reddit data using the official Reddit API and conducted continual pretraining on the two selected models.
Although the training exhibited reasonable convergence and stability, it did not lead to a substantial improvement on the leaderboard. Consequently, we did not adopt the continually pretrained models in our final system. We suspect that the trade-off between domain adaptation and instruction-following ability may have resulted in the latter’s degradation, offsetting potential domain-specific benefits.

[1] https://huggingface.co/Alibaba-NLP/gte-multilingual-base
[2] https://huggingface.co/BAAI/bge-large-en-v1.5
[3] https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5
[4] https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
[5] https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
[6] https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
[7] https://huggingface.co/google/gemma-2b-it
[8] https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
[9] https://huggingface.co/google/gemma-2-9b-it
[10] https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
