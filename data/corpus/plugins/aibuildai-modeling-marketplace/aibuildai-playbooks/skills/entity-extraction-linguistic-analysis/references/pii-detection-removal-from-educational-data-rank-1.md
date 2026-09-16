# 1st place solution - Ensemble of diverse Deberta architectures and postprocessing

Competition: pii-detection-removal-from-educational-data
Rank: #1
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497374

First of all, thanks to Kaggle and The Learning Agency Lab from hosting this fun and very real-world applicable competition!

Thanks @gauravbrills @ympaik @drpatrickchan @evgeniimaslov2 for the wonderful collaboration!

Inference code: https://www.kaggle.com/code/yeoyunsianggeremie/pii-1st-place-solution

Full code: https://github.com/bogoconic1/pii-detection-1st-place

**Context**
===
Business context: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data
Data context: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data

**Overview of the approach**
===



**Datasets**
@nbroad https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated (by far the most useful!)
@mpware https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays
With the help of [this notebook](https://www.kaggle.com/code/minhsienweng/create-ai-generated-essays-using-llm) by @minhsienweng, we also created our own dataset of 2k samples and used it for some training runs. Our created external dataset is available [here](https://www.kaggle.com/datasets/ympaik/pii-1st-solution-datasets) (external_data_v8.json - note this includes nbroad dataset)

**Modelling**
Each experiment was trained on 4 folds (document % 4) then 1 or 2 times full-fit. 
- 13 labels were used for all the training runs. 
- The models are trained on a max length of between 1600-2048 for 3-4 epochs.
- LR 1e-5
- No downsampling of essays with no PII
- o_weight = 0.05

We tried various models as well as architectures, as Deberta was fast in inference and good at both LB and CV we included variations of Deberta architecture in our ensemble. Some variations we tried.

**Multi-Sample Dropout Custom Model**: Helped to train with a better-fitted train curve and less of spikes we saw with training compared to just a normal model.
**Bilstm layer Custom Model:** This introduces a bilstm layer just before the classifier which was tried by some public notebooks. This helped in diversity and was smoother in training and the scores correlated well with LB and CV. Bilstm we did have issues when training with Nan loss. So initializing the layer and also saving with `save_safetensors=False,` helped there. 
**Knowledge Distillation:** As we had some models trained on disparate datasets we thought of an idea to try to use them as teachers for a student trained on different datasets with the same validation folds. This worked great and and improved CV and LB by about 0.005-0.01. We also tried to have multiple teacher models but in the end, were bit short on time. 
**Augmentation:** We tried to train one run with this for ensembling. For some essays, swap the first and last names before training the model (e.g. document 7779 where Leroy can both be a B-NAME_STUDENT and an I-NAME_STUDENT)

Credits to @gauravbrills for points 2 and 3 above, these improved diversity of our ensembles.

For the **Knowledge Distillation** task, the loss function looked like
```py
class CustomTrainer(Trainer):
        def __init__(self, *args, class_weights=None, teacher_model=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            self.teacher_model = teacher_model.to(self.args.device)
            self.temperature = 3
            self.alpha = 0.5

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels").to(self.args.device)
            inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
            outputs = model(**inputs)
            # Get the teacher model's outputs
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
            
            teacher_logits = teacher_outputs.logits
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(self.args.device))

            loss_student = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            distillation_loss = torch.nn.KLDivLoss()(F.log_softmax(logits/ self.temperature , dim=-1),
                                                    F.softmax(teacher_logits/ self.temperature , dim=-1)) * self.temperature**2
            loss =loss_student*self.alpha+ distillation_loss*(1-self.alpha)

            return (loss, outputs) if return_outputs else loss
```

**Postprocessing**
This was the key towards improving our CV and LB. We used the following steps, mostly derived from patterns observed from mistakes in the OOF predictions:
- Using different thresholds for each PII label, rather than using the same threshold for all 13 PII labels
- Removed all NAME_STUDENT predictions that are not title-cased, or contain a digit/underscore
- If a token X in a document is predicted as NAME_STUDENT, all tokens X in the same document will be turned into NAME_STUDENT. (e.g. document 2722 'Saman')
- If there exists a PHONE_NUM prediction that is a number with 9 or more digits, turn it into ID_NUM
- Repairing the missed "\n" tokens in STREET_ADDRESS
- If there are 3 consecutive tokens being predicted as B-USERNAME and the 2nd one is a hyphen or a dot, turn the 2nd and 3rd token to I-USERNAME
- Removing ID_NUM predictions with <4 characters, or >25 characters
- Removing URL_PERSONAL predictions with <10 characters
- Removing EMAIL predictions which don't have @
- Finally, after doing all the steps above, we repaired the span for NAME_STUDENT and ID_NUM tokens, and added on regex predictions if the models missed them

**Details of the submission**
===
- We used a weighted voting ensemble method, using 7 groups and 10 models. The weights of different groups and the voting threshold are tuned via Optuna on the out-of-fold cross validation
- Each group contains 1-2 models, and both fold/full-fit models are used to improve the diversity
- All 3 submissions were ensembles, with different models and different weights. We selected 2 submissions with the full postprocessing and 1 submission with more conservative postprocessing, in the case if the postprocessing overfits

**What didn’t work**
- MLM pretraining
- Freezing layers
- Trying to repair I-URL_PERSONAL predictions
- CausalLM inference, too slow
- Training/Inference with Stride
- Longformer/LLM models
- Label specific model (i.e. train a model with only NAME_STUDENT labels while the other PII labels are turned to O)
- Augment the dataset with rare/multinational names
- Pseudo-labelling the test data
