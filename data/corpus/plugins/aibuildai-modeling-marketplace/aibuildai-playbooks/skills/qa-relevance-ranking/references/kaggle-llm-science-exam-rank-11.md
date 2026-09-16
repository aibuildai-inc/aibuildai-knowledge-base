# 12th place solution

Competition: kaggle-llm-science-exam
Rank: #11
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446660

First of all, we are grateful to the organizers of this amazing competition, which was a great learning opportunity for us. We also appreciate all the kagglers who shared their ideas and code during the competition. Thank my team members @zzy990106, @berserker408, @fankaixie for their hard work.

In short, our solution consists of diverse contexts and models.

# RAG
## Context #1:

Built wiki english faiss index, combine top5 of three recall paths as context for each question:
- bge prompt + answer
- gte prompt + answer
- gte prompt 

## Context #2:
Sentence based openbook context shared by JJ ( @jjinho )

## Context #3:
270k dataset shared by MB (@mbanaei), 270k data cohere and parsed.

## Context #4:
Built a 480k dataset using the notebook MB to build 270k dataset to cover more articles.
The 270k dataset did not cover all the 154 articles, we adjust the clustering parameters to cover all the articles with 480k.

# Models

* Deberta V3 large multiple-choice classification
   Shared by @cderotte.
* Deberta V3 large one-shot:  
Concatenete  all answers together, only inference once for each question.
Model architecture reference [here](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433)





* Llama2 7B
We trained llama2 7B with lora, it inferences 5 times for a single question, takes 3 hours using 512 seq length, llama2 model gave us 0.003 boost for ensemble.

We trained multiple checkpoints of deberta, in our final solution, we used 8 checkpoints to take different contexts. We also observed that train with better context did not improve model, so we selected the checkpoints trained with sentence-based context.
