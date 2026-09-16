# 20th place solution

Competition: commonlit-evaluate-student-summaries
Rank: #20
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446676

Thank you to the organizers for the fun competition and everyone who participated. 
I share my solution.

# Summary
- When I added prompt_text, the local score clearly improved. However, as a result of the experiment, I found that prompt_text is very long (more than 4000) for test dataset. So  I set the max_length to 5000. This is the important point for me.
- weighted ensemble: deberta-v3-large + LightGBM = 7:3

[solution-image]

# 1. Validation Strategy
- GroupKFold : 4-fold (groups=prompt_id)

# 2. deberta-v3-large model
- input
    - input_text = answer: summary_text [SEP] title: prompt_title | question: prompt_question | text: prompt_text
    - tokenizer: max_length=5000
    - not text cleansing
- model:
    - pretrained-model: microsoft/deberta-v3-large
    - freeze: 18 layers
    - head: cls-token
    - max_position_embeddings=5000
- train: 
    - loss: torch.nn.SmoothL1Loss()
    - optimizer: Adam
    - batch_size=2
    - epoch=10
    - scheduler: cosine annealing
- predict:
    - max_token=5000 
    - batch_size=1
    - only 2-fold (When it was 4-fold, it timed out.)

# 3. LightGBM
- features: 141 (count_word, count_paragraph, count_sentence, count_space, etc.)
- train: 2 model (content, wording)

# 4. Ensemble
- weighted average (tune weight by train_oof)
- deberta-v3-large : LightGBM = 7:3

| id | model | local-cv | public | private |
|---|---|---:|---:|---:|
| 1 | Deberta-v3-large (4fold) | 0.4964 | Timeout | Timeout |
| 2 | Deberta-v3-large (2fold) | 0.5007 | 0.461 | 0.476 |
| 3 | LightGBM | 0.5532 | 0.495 | 0.516 |
| 4 | ensemble(2+3) | 0.4822 | 0.455 | 0.461 |

# Others
- pooling-layer: almost same
- N-hidden-layer: almost same
- LSTM/1dcnn-layer: not work
- Set lr for each layer: not work
- Longformer: not work
- summarizing prompt_text using Llama2: not work

When I added prompt_text, the local score clearly improved. I also considered it important to have a model that can handle the long token length of test data. For this reason, although it was a submitted file that was outside the medal range, I believed in the local score and selected this submit file. I'm glad I chose it.
However, I realized the length of the tokens two days before the end of the competition, so I regret not being able to try 4-fold model. .

Thank you for reading.
