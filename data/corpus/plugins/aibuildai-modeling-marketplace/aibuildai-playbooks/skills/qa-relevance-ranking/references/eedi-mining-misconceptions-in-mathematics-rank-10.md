# 10st Place Solution Summary

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #10
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/551722

Thanks to the organizers and Kaggle, this is my second solo gold medal🏅, which is enough to make me a Kaggle Grandmaster. Below is a brief summary of my solution.
## External data
 7k external data using GPT-4O.
## Retrievers
Model: Qwen/Qwen2.5-14B-Instruct,Qwen/Qwen2.5-Coder-14B-Instruct
Train detail:
firstly train(lora) with  7k additional data and then load the lora weight,lastly fintune the train data.
CV: train with 5 folds and then merge using lora.
I integrated a total of three recall models, including the 14b 5-folds, 14b-coder 5-folds, and the full 14b 5-folds. The training code needs to be slightly modified in some Model classes from my previously open-sourced KDD code.
## Rerankers
Model:Qwen32b AWQ,training with lora
Generated approximately For ranking, I used a multiple-choice approach. 
The input consists of questions and answers, along with the top 25 options, and the output is the probabilities of these 25 options.
 During training, I only optimized the probabilities of these 25 characters, using the cross-entropy loss function. In the end, my leaderboard score was achieved using only the 5-folds (7k additional data + training data) of LoRA fusion (LoRA fusion remains effective in any scenario).
## Try but not worked
Distillation：I tried to distill the content of 72b, but since I didn't see any improvement offline, I didn't submit it. I might have been wrong, after all, the offline validation data is relatively limited.


The first-place solution made me realize that there is still much room for optimization in my current solution, but I am also quite satisfied～
