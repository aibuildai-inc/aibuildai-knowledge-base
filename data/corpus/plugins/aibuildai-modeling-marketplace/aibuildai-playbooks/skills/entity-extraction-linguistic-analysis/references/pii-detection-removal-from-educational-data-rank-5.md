# 5th place solution

Competition: pii-detection-removal-from-educational-data
Rank: #5
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497306

# [Introduction]
First of all, I would like to express my gratitude to the host and Kaggle staff for hosting this competition. Also, I would like to thank  the Kaggle community for sharing various datasets and insights. Especially, I am grateful to @nbroad, @pjmathematician, @valentinwerner and @mpware as their datasets and notebook greatly contributed to our solution. Thank you so much! 
And, I am grateful to my teammates, @takai380 and @minfuka. (Congratulations to @takai380 on competition master title!)

# [Ensemble Part]
| | max_length(train/inference) | weight | cv | public | private |
| --- | --- | --- | ---  | --- | ---  |
| ryotak12 | 128/128 | 10 | 0.979 | 0.973 | 0.960 |
| takai380 | 512/512 | 13 | 0.977 | 0.975 | 0.965 |
| minfuka | 1536/4096 | 8 | None | 0.972 | 0.967 |
- Our solution consisted of an ensemble of 12 deberta-v3-large models.
- After performing ensemble methods such as voting or averaging within each member's models, we conducted voting between the members' ensembled predictions.
- Simple voting was the best method in our cv and public lb.
- To increase the number of models used in the ensemble, I utilized AMP for predictions, which reduced inference time.
- As one of our strategies, each member trained models with different max_lengths to achieve diversity in the ensemble.
    -  While this approach worked well for cv and public lb, we found, however, that longer max_lengths performed better on the private lb. (@minfuka's ensemble trained longest max_length achieved a gold medal on its own.)

# [Ryota Part]
- I created a total of 6 models for the final ensemble.
### Common for all models
- backbone : deberta-v3-large
- task : token classification
- full data training
- external dataset
  - @nbroad (https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated)
  - @mpware (https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays)
  - @pjmathematician (https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt)
- max_length : 128
    - train_overlap=96, eval_overlap=64
- positional features
    - Included two types of position information: absolute position and related position.
- EMA
- layer freezeing
    - Freezing the backbone for the first epoch led to faster convergence.
    - From the second epoch, no layers were frozen.
- loss weight
    - “O” : 1, Other : 10
- optimizer : AdamW
- scheduler : Cosine Annealing

### Model Specifics
- with/without prefix
   - Four models were trained using labels with prefixes, while two models were trained using labels without prefixes.
- Additional Training
    - I basically trained the models using the nbroad 's dataset as additional data.
    - However, for some models, I first trained models for a few epochs using the competition data combined with datasets from nbroad, mpware, and pjmathematician. Then, I retrained these models for a few more epochs using only the competition data and nbroad 's dataset.
- max_length
    - I basically trained the models with a max_length of 128, but for the sake of diversity, I trained one model with a max_length of 512.

### Post Process
Before ensemble, I applied the following post-processing steps to each single model
- Removed whitespace
- Ensured prefix consistency
- Removed false positives using regular expressions tailored to each PII type

### Did not work
- 2nd-Stage Model (for removing FP of NAME_STUDENT)
    - Masking the tokens predicted as NAME_STUDENT and using binary classification to predict whether the masked tokens are truly NAME_STUDENT based on the surrounding context.
    - However, this only yielded a 0.0005 improvement in CV, so I did not use it to avoid the complexity of the pipeline.
- AWP
- Label Smoothing, Online Label Smoothing
- GRU (following DeBERTa)
- Layer Reinitialization
- etc…

# [takai380 Part]
### Model1: deberta-v3-large (cv/public/private → 0.974/0.970/0.961)
* max_len=512, stride=128
* Datasets: 
 * @mpware (https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays)
 * @pjmathematician (https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt)
 * @nbroad (https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated)
* Weight of non-'O' losses increased by 5 times
### Model2: deberta-v3-large+lstm (cv/public/private → 0.965/0.972/0.955)
* max_len=512, stride=128
* Dataset: nbroad
### Model3: deberta-v3-large (cv/public/private → 0.973/0.967/0.969)
* max_len=512, stride=128
* Dataset: nbroad
* Weight of non-'O' losses increased by 5 times 
### Model4: deberta-v3-large (cv/public/private → 0.970/?/?)
* max_len=1024, stride=256
* Dataset: nbroad
* Add lstm layer following deberta-v3-large
* Weight of non-'O' losses increased by 5 times
* (This model was not used in the selected private best model.)
### Common for all models:
* B- and I- tags are not used (These tags are rule-based and not data-driven; therefore, they should not be learned in a data-driven manner.)
* Basic rule-based processing:
 * **`NAME_STUDENT`**: Rejects if not starting with a capital letter followed by lowercase letters. If one character, remove.
 * **`URL and EMAIL`** formats must be adhered to, or they are rejected.
* Strict rule-based processing:
  * If an article precedes the prediction of **`NAME_STUDENT`**, it is removed unless "'s" follows, in which case it is not removed.
  * Examples:
            - Articles ('The', ' the', 'a', 'and') often precede non-name terms. If 'at' is before, it's likely a location and is excluded.
            - **`'The', 'Marias', 'Gamesa', 'took'`**
        - However, there are exceptions when followed by "'s" as it may not fit the rules as intended:
            - **`~ the Takai's house ~`**. In such cases, the rule is not applied.
    
This is a personal observation, but I am surprised to see that the scores(0.969) achieved with a single model, either similar to or higher than those obtained by ensemble methods combining CV and public LB scores. Does this suggest that the labels already detectable by a single model are achievable, while others pose more difficulty for machine learning models to detect?
    

# [min fuka Part]
    
### Common for all models
* Base-Model
 * deberta-v3-large(Freezing embeddings , Freezing first 6 layers)
  * max_len=1536
  * Datasets
      * training_data : competition training data (4/5）+ external data
      * validation_data : competition training data (1/5）
 * TrainingArguments
      * fp16=True,
      * learning_rate=2e-5,
      * num_train_epochs=3,
      * per_device_train_batch_size=8,
      * gradient_accumulation_steps=2,
      * gradient_checkpointing=True,
      * logging_steps=50,
      * evaluation_strategy='steps',
      * eval_steps=50,
      * save_strategy="steps",
      * save_steps=50,
      * load_best_model_at_end=True,
      * lr_scheduler_type='cosine',
      * metric_for_best_model='f5',
      * greater_is_better=True,
      * warmup_ratio=0.1,
  * loss_function
      * CrossEntropyLoss(weight=class_weight)
      * class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]　←Only Lable 'O' is 0.1
 
### External data　(Thanks for all exernal-data author)
* I use various datasets to generate diversity
 * Model1 using @mpware 's dataset(https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays)
 * Model2 using dataset in @valentinwerner 's notebook(https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset/output)
 * Model3 using "moredata" in @valentinwerner 's notebook (https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset/output)
   
### Rule-based processing
* Nothing
    
### Inference
* Notebooks I referred to:
 * https://www.kaggle.com/code/valentinwerner/945-deberta-3-base-striding-inference (Special thanks @valentinwerner)
* Difference from trainig
 * INFERENCE_MAX_LENGTH=4096
 * STRIDE=384

### Ensemble
* ensemble (use models mentioned above: Model1, Model2, and Model3.)
  * ensemble_pred = (Model1 + Model2 + Model3) / 3 ← before softmax
 * Above inference score is Public:0.972 Private:0.967
 * (Add takai-rule-based processing, score is Public:0.971 Private:0.968)URL and EMAIL formats must be adhered to, or they are rejected.Examples:
            - Articles ('The', ' the', 'a', 'and') often precede non-name terms. If 'at' is before, it's likely a location and is excluded.
