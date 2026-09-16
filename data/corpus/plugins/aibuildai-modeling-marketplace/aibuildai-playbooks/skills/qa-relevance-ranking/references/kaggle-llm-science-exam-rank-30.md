# 31th Place Solution

Competition: kaggle-llm-science-exam
Rank: #30
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446233

# [Our Inference code](https://www.kaggle.com/code/sugupoko/llm-31th-place-solution/notebook)
# [70k Dataset](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/445438)


Thank you for host and competitors.
It is interesting competition for me.

My first NLP competition was a great learning experience for me.
I'm looking forward to the solutions from the top.

## Team Member
@shuichiurabe
co-workers!!

## Solution overview
### inference flow
We decided on a strategy to have fewer models and more variations in the data. After observing that a 270k dataset was released and the LB scores significantly increased, we guessed this approach is validated.


### model
- we trained with 60k dataset [cdeotte's discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436383)
- only changed max length to 640. Freeze 18, Freeze embedding.

## Things that didn't go well:
- Searching for similar documents. 
  - Training the Sentence Transformer.
  - Replacing the Sentence Transformer.
  - BM25 method
- Regarding answer prediction:
  - Increasing the data.
  - Spent $70 on custom data, but it was ineffective.

## Things I couldn't do:
- Searching for similar documents.
    - Expanding data sources.
    - Couldn't due to PC specs (Needed more than 200GB RAM).
  
- Regarding answer prediction (continued):
    - Making labels soft.
        - Wanted to give partial scores like 0.5 when the answer contained some correct content.

## Reference previous competition
Our many experiments are from the previous competitions
- https://www.kaggle.com/competitions/feedback-prize-2021/leaderboard
- https://www.kaggle.com/competitions/AI4Code/leaderboard
- https://www.kaggle.com/competitions/feedback-prize-effectiveness/leaderboard
- https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/leaderboard
- https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/leaderboard
- https://www.kaggle.com/competitions/feedback-prize-2021/leaderboard
