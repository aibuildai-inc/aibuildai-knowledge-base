# 13th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #13
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369440

### 14th Place Solution

First of all, I'd like to thank kaggle team, host for hosting this competition and thank many other participants to share codes, ideas and datasets.

I'm really happy to get solo gold.

### Overview

My final submission is weighted avg model which is 0.5 * 11 models nelder-mead blended one + 11 models bayseian ridge stacking one.

Each models are trained with various hyperparameter such as seed, num_fold, max_length, and so on.

[Screen Shot 2022-11-30 at 21 07 06]

My final Code is here. : 
- https://www.kaggle.com/code/kaerunantoka/fb3-private-14th-solution-code?scriptVersionId=112497983

- I'm sorry that a lot of my notebooks are public because everyone can run this code.

My training repo is here. :
- https://github.com/osuossu8/FeedbackPrize3


### What Worked
- [High Impact] Idea
    - Trust CV
        - I added new model to ensemble in case oof cv score is improved.
    - [rapids SVR](https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x)

    - [nelder-mead optimized weighted blending](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead)
        - I heard this method first time in [this previous nlp compatitions solution.](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328)

    - longer max length (1536) and use head-and-tail tokens
        - many public notebooks used 512 max_length
        - however max token length in train full_text is around 1430.
        - I decided to use longer token length for my training.

    - freeze embedding layer and first 2 layers
        - ```
            def freeze(module):
                for parameter in module.parameters():
                    parameter.requires_grad = False
            
            if self.cfg.freezing:
                freeze(self.model.embeddings)
                freeze(self.model.encoder.layer[:2])
            ```
    - evalulate twice per epoch

- [Medium Impact] Idea
    - I replaced svr with xgboost and added final model. This is contributed to my model diversity.

    - add special token 
        - I noticed full_text has many "\n\n".
        - ```
            def setup_tokenizer(CFG):
                CFG.tokenizer.add_tokens([f"\n"], special_tokens=True)
                CFG.tokenizer.add_tokens([f"\r"], special_tokens=True)
            ```

### What Didn’t Work
- Idea
    - Use my finetuned embeddings when I try rapids svr method.
    - Use Universal Sentence Encoder Embedding 
    - Use text features as meta feature when I finetuneing hugging face models and my 2nd stage stacking models.


### Additional Context:

- CV Strategy
    - I used [abiheshark's cv strategy](https://www.kaggle.com/code/abhishek/multi-label-stratified-folds) with 5, 10 folds various seed.

- Models

| Name | CV | max length | num_fold | link |
| ---- | ---- | ---- | ---- | ---- |
| deberta-v3-large  | 0.4584 | 1536 | 5 | https://github.com/osuossu8/FeedbackPrize3/blob/main/exp/024.py |
| deberta-v3-large  | 0.4585 | 1536 | 5 | https://github.com/osuossu8/FeedbackPrize3/blob/main/exp/026.py |
| deberta-large  |  0.4617 | 512 | 5 | https://www.kaggle.com/code/kaerunantoka/feedback3-n004 |
| roberta-base  | 0.4622 | 512 | 5 | https://www.kaggle.com/code/kaerunantoka/feedback3-n005 | 
| longformer-large-4096  | 0.4670 |  512  |  5  |  https://www.kaggle.com/code/kaerunantoka/feedback3-n009  |
| deberta-v3-base  | 0.4554 | 512 | 5 | [original](https://www.kaggle.com/code/batprem/deberta-layerwiselr-lastlayerreini-infer), [my copy to save oof](https://www.kaggle.com/code/kaerunantoka/fb3-public-save-oof) |
| deberta-v3-large  | 0.4670 | 512 | 10 | https://www.kaggle.com/datasets/kojimar/20221012-123357-deberta-v3-large |
| deberta-v2-xlarge-mnli  | 0.4675 | 512 | 10 | https://www.kaggle.com/datasets/kojimar/0919-deberta-v2-xlarge-mnli |
| deberta-v3-large | 0.4548 | 512 | 10 | https://www.kaggle.com/datasets/kojimar/0926-deberta-v3-large-unscale |
| pretrained hugging face embedding & rapids SVR | 0.4499 | - | 15 | [original](https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x), [my copy to save oof and models](https://www.kaggle.com/code/kaerunantoka/fb3-public-svr-024-infer) |
| pretrained hugging face embedding & XGB | 0.4552 | - | 15 | https://www.kaggle.com/code/kaerunantoka/fb3-pretrained-bert-family-xgb-train |
| bayseian ridge | 0.4446 | - | 10 | https://www.kaggle.com/code/kaerunantoka/fb3-stacking-nelder-mead |

### Important Citations:
- notebook
    - https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x
    - https://www.kaggle.com/code/batprem/deberta-layerwiselr-lastlayerreini-infer
    - https://www.kaggle.com/code/quangphm/lb-0-43-simple-ensemble-deberta-base-svr

- discussion
    - https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328

- dataset
    - https://www.kaggle.com/datasets/kojimar/20221012-123357-deberta-v3-large
    - https://www.kaggle.com/datasets/kojimar/0926-deberta-v3-large-unscale
    - https://www.kaggle.com/datasets/kojimar/0919-deberta-v2-xlarge-mnli


Thanks and Acknowledgements:

Finally, I'd like to thank Kaggle, host for hosting such an iteresting competition and all who participated in lively discussions.
I wish my solution helps you in your future NLP competitions!
