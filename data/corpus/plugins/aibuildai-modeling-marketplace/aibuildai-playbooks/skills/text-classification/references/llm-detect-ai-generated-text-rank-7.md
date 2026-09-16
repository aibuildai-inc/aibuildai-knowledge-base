# [7th Place Solution] Generate Data with Non-Instruction-Tuned Models

Competition: llm-detect-ai-generated-text
Rank: #7
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470643

## **Forewords**
Firstly, a big thank you to Kaggle, the Learning Agency Lab, and Vanderbilt University for orchestrating this interesting competition. Hats off to you all! 🎩 This marks my inaugural Kaggle competition, and the learning curve has been quite the journey.
Early in the competition, I had reservations about how well the TF-IDF approach might translate to real-world scenarios. This skepticism led me to delve into LM-based methods and resist the urge to switch back. Little did I anticipate the leaderboard shake-up would be this significant and the outcome has genuinely taken me by surprise.

## **Final Solution**
My "magic sauce" is to use only non-instruction-tuned models to generate data. If we look at the 3 generated essays provided in this competition, they seem increasingly human-like. More importantly, none of them has the generic ChatGPT answer format. I believe this was a big hint by the host. 
After reading [James' post](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882) and [this excellent article](https://www.lesswrong.com/posts/t9svvNPNmFf5Qa3TA/mysteries-of-mode-collapse-due-to-rlhf#The_one_answer_is_that_there_is_no_one_answer) (tl;dr instruction tuning has an adverse effect on response diversity), I generated around 400k non-essay texts from slimpajama and 25k essays with persuade 2.0. Models used are vanilla Falcon-7B, Mistral-7B, and Llama2-7B with temperatures ranging from 0.8 to 1.2, top p values from 0.8 to 0.98, and frequency penalties from 0 to 0.3.
Then I simply finetuned a single DeBERTa-v3-large model with this dataset at 512 context length and inference at 1024 length. One training run takes around 8 hours on a 3090. I tried training at 1024 length, but the results were worse. I couldn't find much information on why this is the case, so insights on this matter would be appreciated.
In my final solution, predictions falling within the 40th to 60th percentile were replaced with outcomes from a basic TF-IDF + SGD model. This tweak resulted in a modest increase in my CV, but had minimal impact on LB.

| Model | Dataset | Public LB | Private LB |
| --- | --- | --- | --- |
| DeBERTa-v3-base-512 | only persuade | 0.869 | 0.875 |
| DeBERTa-v3-base-512 | persuade & slimpajama | 0.921 | 0.920 |
| DeBERTa-v3-base-1024 | persuade & slimpajama | 0.910 | 0.901 |
| DeBERTa-v3-large-512 | persuade & slimpajama | 0.942 | 0.965 |
| DeBERTa-v3-large-1024 | persuade & slimpajama | 0.922 | 0.957 |
| DeBERTa-v3-large-512 + TF-IDF | persuade & slimpajama | 0.942 | 0.965 |

## **Prompt Engineering**
My prompt for generating essays is as follows:
```md
(100 points) [essay instruction] Your text should be around 500 words.\n\n\n\n
Name: [random name based on gender]
Grade: [grade]
Date: [random date]
[essay title]
[one cleaned essay from persuade 2.0]\n\n\n\n
Name: [another random name with the same gender]
Grade: [same grade]
Date: [same date]
[essay title]
```
Grade and gender information were taken directly from persuade 2.0 dataset. I made 10 generations for each persuade 2.0 essay, and then filtered generations that are too similar to the sample human essay using embedding similarity and Levenstein distance. Generations were then further filtered by length and repetition.

## **List of Failed Ideas**
- Separate essay into list of semantic blocks, embed each block, then train a bi-LSTM on top of it. (score 0.763)
- Same architecture, but train LSTM on the differences of each pair of neighboring blocks. (score 0.685)
- [Robust Encoding](https://arxiv.org/pdf/2005.01229.pdf) to combat character-level adversarial attacks.
- Sliding window inference and averaging the logits. (score 0.929)
- MLM pretraining on the test set (10 epochs with deberta-v3-xsmall), then finetune on my dataset. (score 0.689)
- And a lot more ideas that didn't even pass local CV.

## **Dataset and Code**
### Training and Dataset Generation Code
https://github.com/Tailen/Kaggle-Detect-AI-Generated-Text-7th-Solution
### Datasets
https://www.kaggle.com/datasets/tailen/persuade-corpus-ai-generated-dataset
https://www.kaggle.com/datasets/tailen/slimpajama-ai-generated-parallel-dataset
### Inference Code
https://www.kaggle.com/code/tailen/daigt-deberta
