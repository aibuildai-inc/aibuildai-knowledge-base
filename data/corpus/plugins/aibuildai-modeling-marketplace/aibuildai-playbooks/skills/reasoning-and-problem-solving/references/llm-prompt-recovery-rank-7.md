# 7th place solution: Mastering mean prompt

Competition: llm-prompt-recovery
Rank: #7
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494650

First of all, I would like to thank the organizers for hosting this exciting competition. Also, a big thank you to my teammate @andreivanenko for his wonderful collaboration.

TL;DR: Our solution primarily builds on the mean prompt, which is slightly improved with Mistral 7B.

**Mean prompt training:**
Initially, we trained the mean prompt using a public dataset of prompts available at https://www.kaggle.com/datasets/what5up/concat-prompts. 
The main method of training was to add a new word to the existing prompt to make it more similar to the other prompts in the training dataset. The corpus of words that was used included all the words from the training prompts.
The training of the mean prompt began with the phrase "Rewrite this text" and included the following methods:
- Greedy search, Beam search: Adding the next best word to the end of the prompt.
- Inserting: Inserting new words into the middle of the prompt.
- Pruning: Deleting unnecessary words from prompts.

Through experimentation, we found that it is better to initially generate a mean prompt using beam search, and once it stopped improving accuracy, use the insertion and pruning methods.
Also, during leaderboard probing, we discovered that prompts with a length of more than 128 tokens for some reason scored approximately 0.41, so they required additional pruning.
Using all these approaches, the score was improved to 0.68.
**Reducing bias in training dataset:**
As the mean prompt became more accurate on our local dataset, we saw that the score difference from the leaderboard scores increased. We decided to make it better by regenerating it in a way that reduces the score difference between our local dataset and the leaderboard.
For this, we created a list of all mean prompts previously submitted to the leaderboard:
```
scores = [
    # 70 more mean prompts
    [0.60, "Improve the text to this."],
    [0.59, "rewrite this text tothepoint humanoid about around towards takes accompanying"],
    [0.59, "rewrite this text conveying human ensue somehow portrayal one further"],
    [0.59, "rewrite it make thee o e be how this described the text ideas lie plane ultimate"],
    [0.58, "Improve the text to this. Rewrite this text using this style."],
    [0.56, "Improve essay to this. rewrite this text using this style."]
]
```
To compare our local dataset with the leaderboard, we first evaluated our mean prompts using this dataset and obtained their scores. Then we compared these scores to the leaderboard using the cosine similarity metric. Following this, we began constructing our dataset by randomly selecting 10 prompts in each iteration and adding them to the this dataset if they improved our evaluation metric.
```
best_prompts_ids = []
best_score = 0
iteration = 0
while True:
    sample_ids = random.sample(range(len(prompts_embs)), 10)    
    candidate_prompts_ids = best_prompts_ids + sample_ids

    # Calculate the sharpened cosine similarity between the embeddings of the mean prompts and the prompts from generated dataset
    candidate_scores = (cosine_similarity(lb_prompts_embs, prompts_embs[candidate_prompts_ids, :]) ** 3).mean(axis=1).reshape(1, -1)
    # Compute our dataset evaluation metric which is the cosine similarity between scores of our generated dataset and LB.
    cos_score = cosine_similarity(candidate_scores, lb_scores)[0][0]
    
    if cos_score > best_score:
        best_score = cos_score
        best_prompts_ids = candidate_prompts_ids
        
        print(f"Iteration: {iteration}, cos_similarity: {cos_score:.6f}")

    iteration += 1
```
In this way, we created a new local dataset for training. This allowed us to improve the score on the public leaderboard from 0.68 to 0.70. This led to our best mean prompt, which is as follows:
>rewrite also key essence since its cry thine that had then expressed in improve the underlying paragraphs from it more directly but with either в similar descriptive desired statement to best how you described such text it is da ultimate involves that an human maintains retell animistic this newly eventual presented than classic adult manner due please my would just fashion the following as follows device ss plea chefs poe us da formal piece while edit out any non grand local warera band gospel virtual salt park industry flair useless question oath sherlock taker for page transcript get four empowerment discuss name and height so frame it

**Enhancements mean prompt with LLM:**
In our experiments, we discovered that adding output from the LLM slightly improved the accuracy of the mean prompt on both the local dataset and the leaderboard.
We used the Mistral 7B model to create prompts starting with "Improve this text by." The resulting text then was added to our best mean prompt, which improved results by approximately 0.005.

Code:
- [Inference notebook](https://www.kaggle.com/code/taras456/7th-place-solution-mean-prompt-llm)
- [Training mean prompt](https://www.kaggle.com/code/taras456/mean-prompt-training)
- [Dataset preparation](https://www.kaggle.com/code/taras456/dataset-preparation)
