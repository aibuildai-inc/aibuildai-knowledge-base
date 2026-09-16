# 6th place solution for The Learning Agency Lab - PII Data Detection

Competition: pii-detection-removal-from-educational-data
Rank: #6
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/498304

I would like to thank the host for organizing this competition and all the participants who generously shared valuable information.  I'm very happy to get the solo gold medal!


## Context
- Business context: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/overview
- Data context: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data

## Overview of the approach
- Weighted averaging of three deberta models
- Mapping from Deberta tokenizer tokens to submission format Spacy tokenizer tokens
- Created additional dataset by Mistral-v02 and used it with the publicly shared datasets
- Post-processing after error analysis


#### Data
Created and used the following two datasets in conjunction with competition data.  
- Dataset combined from publicly shared datasets. (Thank you very much for creating datasets!! @mpware , @pjmathematician , @nbroad ,  @valentinwerner , @tonyarobertson )
- Dataset generated using Mistral-7B-Instruct-v0.2. (This method is inspired by [this solution](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395))
    - Finetuning of LLM:  
		- Extracted design thinking tool names from the competition's training data and considered it as the representative tool for that essay.  
		- Used the above tool + the essay's PII + [the task instruction](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/478911) as the input Instruction, and the essay as the output for finetuning the LLM.  
	- Generated dataset with the finetuned LLM  
		- Randomly selected multiple tools from the design thinking tools (Visualization, Storytelling, Mind Mapping, Learning Launch, Brainstorming, Persona, Value chain analysis, Journey Mapping, Ethnography)  
		- Used Faker to create each PII and randomly selected multiple.  
		- Input the randomly selected tools and PII + the task instruction into the LLM to output the essay.  
        - I tried several prompts and generated 1722 data in total.

#### CV strategy
- Create one-hot labels for whether the essay contains any PII belonging to each label, and use multi stratified k fold.
- Evaluated with 4 fold cross validation.
- There wasn't much correlation between CV and LB...


#### Model
Trained Deberta-v3-large x2 (model1, model2) and deberta-v3-base (model3) each with the following Custom head in 4 folds. A total of 12 models were weighted average.

- Custom Head:
    - Since the token splitting by Deberta tokenizer and spacy tokenizer is different, it is necessary to map the output of deberta.
    - Specifically, each token's prediction probability, which is the output of the model, is converted to a prediction probability for each character by offset mapping, and this is further converted to a prediction probability for each token of the spacy tokenizer.
    - To achieve the above processing during training, a custom head as shown in the figure below was created.
    - When converting from char to spacy token, `torch.Tensor.scatter_reduce_(reduce="mean")` was used to take the average prediction probability of the characters belonging to one spacy token. (I reffered to [this solution](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313201).)
[custom head]

- Loss:  
    - Instead of using "B-" and "I-" for each label, the output of deberta was classified into 8 classes ("O" + each PII type).  
    - Focal Loss was used for the loss function.  
    - The weight of the O label was set to 0.1.  
    - The Loss was calculated using deberta's output tokens (deberta token loss), and the losses calculated from the character prob and spacy token prob of the above Custom Head were also added as auxiliary losses (char loss, spacy token loss).  
    - For Deberta token prob, char prob, and spacy token prob, output the positions of the first and last tokens of each label at the same time, calculate the BCE loss, and added it to the above folcal loss. (deberta token sp_ep loss, char sp_ep loss, spacy token sp_ep loss)

- Training parameters
    - Model1:
        - Backborn: Deberta-v3-large
        - Dataset: Trained with competition data + [mpware](https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays?select=mpware_mixtral8x7b_v1.1-no-i-username.json)
        - Loss:
            - This model does not use the custom head.
            - Deberta token loss + deberta token sp_ep loss
        - Freeze layer: 5 layers
        - Batch: 1

    - Model2:
        - Backborn: Deberta-v3-large
        - Dataset:
            - Trained with shared dataset ([mpware](https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays?select=mpware_mixtral8x7b_v1.1-no-i-username.json)+[pjmathematician](https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt)+[nbroad](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated)+[valentinwerner](https://www.kaggle.com/datasets/valentinwerner/pii-label-specific-data)+[tonyarobertson](https://www.kaggle.com/datasets/tonyarobertson/mixtral-original-prompt)) only for 1 epoch. (pretrain1)
            - Using pretrain1 as the initial weight, trained with the generated dataset for 1 epoch. (pretrain2)
            - Using pretrain2 as the initial weight, trained with competition data only.
        - Loss:
            - Deberta token loss + char loss + spacy token loss + deberta token sp_ep loss + char sp_ep loss + spacy token sp_ep loss
        - Freeze layer: 5 layers
        - Batch: 1

    - Model3:
        - Backborn: Deberta-v3-base
        - Dataset: Trained with competition data only
        - Loss:
            - Deberta token loss + char loss + spacy token loss
        - Freeze layer: No
        - Batch: 2

    - Common settings for Model1~3
        - Epoch: 3
        - Token max length:
            - Training: 4600
            - Inference: 6300
        - Learning rate:
            - Deberta layer: 2e-5
            - Custom head: 1e-4

#### Postprocessing
When I participated in a [previous NER competition](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes), I was surprised that most of the gold medal solutions were improving accuracy by post-processing the model's output. Therefore, this time I properly looked at the model's output and dealt with parts that were not inferring well through post-processing.  
        - Set the label of whitespace characters that are not recognized by Deberta tokenizer but appear in spacy tokens to "O".
        - Convert the "\n" between STREET_ADDRESS tokens to I-STREET_ADDRESS (since the deberta tokenizer deletes "\n", it needs to be restored in postprocessing)  
        - In the same essay, if a token is detected as NAME_STUDENT, set all other tokens with the same string to NAME_STUDENT as well.

#### ensemble  
| Model | CV | Private LB | Public LB |
| ---- | ---- | ---- | ---- |
| Model1 | 0.9702 | 0.9625 | 0.9757 |
| Model2 | 0.9706 | 0.9675  | 0.9694 |
| Model3 | 0.9533 | 0.9517 | 0.9416 |
| weighted average of model1-3 | 0.9765 | 0.9681 | 0.9770 |


### Souces
- https://www.kaggle.com/code/minhsienweng/create-ai-generated-essays-using-llm
    - I referred to the method of generating data with LLM.
- https://arxiv.org/abs/2402.14568
    - Reference material for data augmentation of NER task.
