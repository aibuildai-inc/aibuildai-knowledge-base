# Team Turing: 9th Place Solution

Competition: feedback-prize-english-language-learning
Rank: #9
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369956

A big thanks to Kaggle and the competition hosts for organizing the amazing feedback series! We learned a lot during the competitions and especially enjoyed the addition of efficiency track in Feedback 2 & 3. We look forward to participating in future competitions!

## Special Mention
Congratulations to our teammate @syhens for becoming a Kaggle Competitions Master! Very well deserved and this was special after many narrow gold misses! 

## Overview

Our solution focused on building a set of diverse models to facilitate better ensemble performance. To this end, we trained the following models which were finally used in our ensemble. The actual experiments trained were a lot more :) 

| Model | Backbone | Training Approach | PL | CV &nbsp;&nbsp; &nbsp;| PB &nbsp;|
| --- | --- | --- | --- | --- | --- |
| exp009a | deberta-v3-large | PET | Yes, 10k Random Samples | 0.4464 | 0.4374 |
| exp022b | deberta-v3-large | Keywords | Yes, 4k Min-distance Samples | 0.4444 | 0.4382 |
| exp024 | deberta-v3-large | Multiscale | No | 0.4493 | 0.4399 |
| exp024a | deberta-v3-large | Multiscale, MSE + Ranking Loss | No | 0.4484 | 0.4424 |
| exp026b | deberta-v3-large | Freeze + Reinit | Yes, 4k Random Samples | 0.4447 | 0.4373 |
| exp027 | deberta-v3-large | SETFIT | No | 0.4470 | 0.4371 |
| exp030 | deberta-v3-small | MSE + Ranking Loss | Yes, 4k Min-distance Samples | 0.4490 | 0.4396 |
| exp120 | deberta-v3-small |2 Stage PL + Loss Function | Yes | 0.4586 | 0.4467 | 
| exp121 | luke-large | Same as above | Yes | 0.4567 | 0.4462 |
| exp132 | roberta-large | Same as above  | Yes | 0.4492 | 0.4438 |
| exp203b | deberta-large | Prompt + Smooth L1 Loss | Yes, 4k Min-distance Samples | 0.448 | 0.4380 |
| exp207a | deberta-v3-large | PET | Yes, 4k Min-distance Samples | 0.4468 | 0.4381 |
| exp208 | deberta-large |Prompt + Smooth L1 Loss | Yes, 4k Min-distance Samples | 0.4501 |  |
| exp300 | deberta-v3-base | Clean Text MCRMSE Loss | Yes, 4k Min-distance Samples | 0.4553 | 0.4460 |
| exp302 | deberta-large | GRU Head. <br> Mean Max Pool.<br>  Smooth L1 Loss| Yes, 4k Min-distance Samples | 0.4549 |0.4460 |
| exp303d | deberta-v3-large | LSTM Head. Mean Max Pool. | 2 Stage PL | 0.4464 | 0.4366 |
| exp310 | deberta-v3-large |  | 2 Stage PL | 0.4508 | 0.4387 |
| exp320 | deberta-v3-large |  | No| 0.4512 | 0.4387 |
| SVR | SVR Models trained on embeddings| | No | 0.45 | 0.4431 |



## What Worked

- **[High Impact] Model Diversity**
    - **PET Based Approach**
        - Use the Pattern-Exploiting Training (PET) that reformulates input examples as cloze-style phrases to help language models understand a given task.
        - Pattern:
        
        ```
        Student Essay Evaluation:
        
        Performance on cohesion (essay organization; transition; logical sequencing)? [MASK]. Performance on syntax (sentence structure and formation; word order)? [MASK]. Performance on vocabulary (word diversity; topic related terms)? [MASK]. Performance on phraseology (phrases; idioms; collocations)? [MASK]. Performance on grammar? [MASK]. Performance on conventions (spelling; capitalization; punctuation; contractions)?[MASK]. [SEP]
        
        Essay: {full_essay}
        ```
        
        - Verbalizer
            - Positive Tokens List: `['Excellent', 'Good', 'A', '5', '10', '100', 'excellent', 'good', 'High', 'high', 'Strong', 'strong']`
            - Negative Tokens List: `['1', 'C', 'D', 'F', 'Bad', 'Limited', 'limited', 'poor', 'Poor', '0', 'Low', 'low']`

        - Additional Pattern/Verbalizer
         - Pattern: Example for cohesion here. 
        
        ```
         Rate the following essay for cohesion (good text organization, transitional, overlap) with a score between 1 (bad) and 5 (great). The cohesion score for the following essay is 1 or 2 or 3 or 4 or 5? The score is [MASK]. {full_essay}
        ```
        
         - Verbalizer
            - Positive Tokens List: `['3', '4', '5']`
            - Negative Tokens List: `['1', '2']`


    - **Keywords Approach**
        - Create a set of keywords (words / phrases) by studying Feedback 3 that carries important signal regarding the different targets
        - Use the ****`Finding Dataset Shortcuts with Grammar Induction`** to identify the keywords
        - Pre-process the essay by prefixing the identified keywords to the essay. For example:
        
        ```
        thomas jefferson, much we, always doing, this statement, it is always, a sense, it makes, makes you, more than, what you, you can also, the way, the first reason, thing is, it gives you, you worked, your school, real world, do more, i di, with my, my life, could make you, want to do, such a, give you, opportunity to, you may be, able to, how to, finishing school, you will take, that you learn, our life, in life, also a, trying to accomplish, you become, better person, in conclusion, it helps you, can alway [SEP] [SOE] Thomas Jefferson once states that <……> makes you feel accomplished in life, and shows that you can always strive to do more in you life [EOE]
        ```
        
    - Multi-scale approach
        - Predict target using different granularities of the essay
            - Scale 1: mean pooling over entire essay (standard)
            - Scale 2: mean pooling over paragraphs, followed by LSTM and mean pooling transformation
            - Scale 3: mean pooling over sentences, followed by LSTM and mean pooling transformation
        - Mean of 3-scale predictions to get the final score
    - SETFIT approach
        - Two step process
            - Step 1: Fine-tuning of a sentence-transformer in contrastive manner
            - Step 2: Training a regressor using embeddings from step-1 fine-tuned sentence transformer
        - Reference: [https://huggingface.co/blog/setfit](https://huggingface.co/blog/setfit)

- **[High Impact] Ensemble**
    - Result : Filter N models down to a manageable list keeping diversity in mind. 
    - Reasoning:  Given the large number of diverse experiments trained by our team, it was important to find a subset of diverse models that helped the overall CV. Hill climbing was the perfect approach for this. 

- **[High Impact] Pseudo Labelling**
    - We carefully curated an unlabelled essay corpus and subsequently sampled essays from that corpus such that PL distribution matches the training distribution as closely as possible. For details,  please refer to our 1st place efficiency solution here: https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369646
    - Result: CV improved by around 0.003
    - Reasoning: PL helped to distill knowledge from our best ensemble into the low capacity model used for efficiency. PL also has regularizing effect and often leads to better generalization.

- **[High Impact] 2 Stage Pseudo Labeling**
  - Result: CV improved 0.002 to 0.005
  - Reasoning and Context:  There were 2 sets of pseudo labeling(PL) for the team
    * Based on FB1 + FB2 - FB3 unlabeled data - Two-stage training works best. 
    [stage 1] Train a model with PL only, validate 5 times an epoch and save the checkpoint. 
    [stage 2] Load the checkpoint from stage 1 and fine-tuning on the train set with relatively smaller lr. 
    *  Min-dist PL - The only difference is save the last checkpoint of stage 1

- **High Impact] Freezing & Re-initializing**
    - Freeze bottom 6 layers of the model and re-initialized the top layer of the transformer backbone.
    - Result: CV around 0.001 better
    - For this task, it was helpful to freeze bottom transformer layers to keep intact lower level syntactic patterns learned by the backbone during its LM pre-training. On the other hand, re-initializing a top layer gave more flexibility to learn task specific signals.

- **[High Impact] AWP & EMA**
    - Adversarial training via AWP helps the model to become more robust by inducing a flatter weight loss landscape. CV also improved by around 0.001
    - EMA helps with generalization by keeping track of running average of model weights (similar to SWA)
    
- **[High Impact] LSTM + MeanMax Pooling**
    - Reasoning and Context: Tried different combinations of pooling and heads but LSTM + MeanMax pooling worked.

- **[High Impact] 2 Stage Loss and PL Training**
  - This methodology was used by some models to increase diversity.
  - Stage 1: Use loss function X with Pseudo Labels and train for one epoch
  - Stage 2: Use stage 1 checkpoint. Use another loss function Y for original train data and train for more epochs. Same loss function across stages gave worse CV vs different loss functions. 

- **[Medium Impact] Grouped differential learning rate**
  - Grouped differential learning rate: This is very similar to standard LLRD. Split backbone's encoder layers into 3 groups, set lr per group. [e.g. sSet 5e-5/2e-5 as base_lr for high level group, base_lr / 2.6 for middle level group, base_lr / 2.6 ** 2 for low level group; for head and LSTM + MeanMax pooling, set 5e-3 or 1e-3]
​
- **[Low Impact] MLM**
  - Result: -~0.001
  - Reasoning and Context: MLM seemed to only work with deberta-large and FB2 dataset. 
- **[Low Impact] Adding a new token for `\n\n`**
- **[Low Impact] Ranking Loss**
    - Adding ranking loss to standard MSE loss made aware of relative ranking of training examples within a batch

## What Didn’t Work

- Mask Augmentation
- Random Augmentation
- Auxiliary Target Training  -  add POS tagging, Statistical feature, ... as auxiliary target 
- Stacking - Add statistical and readability features to apply stacking with OOFs 
- Paragraph Random Shuffle 

## Important Citations

- Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference ([https://arxiv.org/abs/2001.07676](https://arxiv.org/abs/2001.07676))
- Improving and Simplifying Pattern Exploiting Training ([https://arxiv.org/abs/2103.11955](https://arxiv.org/abs/2103.11955))
- SetFit: Efficient Few-Shot Learning Without Prompts ([https://huggingface.co/blog/setfit](https://huggingface.co/blog/setfit))
- On the Use of Bert for Automated Essay Scoring: Joint Learning of Multi-Scale Essay Representation ([https://aclanthology.org/2022.naacl-main.249/](https://aclanthology.org/2022.naacl-main.249/))
- Finding Dataset Shortcuts with Grammar Induction ([https://github.com/princeton-nlp/ShortcutGrammar](https://github.com/princeton-nlp/ShortcutGrammar))
- SVR https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x

## Thanks and Acknowledgements
Thanks to the organizers and Kaggle for a very exciting competition!
Many thanks to my teammates (@conjuring92, @syhens, @harshit92) for such an amazing collaboration and perfect teamwork! I am sure we will be teaming up again for future competitions! 

## Team Members

Raja Biswas @conjuring92
Yao He @syhens
Harshit Mehta @harshit92
Trushant Kalyanpur @trushk
