# 14th place solution

Competition: pii-detection-removal-from-educational-data
Rank: #14
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497968

First, I would like to thank the host and the Kaggle staff for organizing such an interesting competition.
I am also grateful to everyone who shared many valuable resources, including powerful external datasets.

Lastly, I want to thank my teammates, @yyykrk and @irrohas. Thanks to their strong ideas, we were able to win the gold medal and I have finally achieved my long-desired title of Grandmaster!

# Summary
.png?generation=1714135424165353&alt=media)

- We implemented an ensemble of a 13-class modes(BIO format) and an 8-class models (models that does not differentiate between B and I labels).
- Before ensembling, we converted the predictions from the 13-class model into 8 classes. (For example, the probability of NAME_STUDENT = the probability of B_NAME_STUDENT + the probability of I_NAME_STUDENT.)
- For ensemble, the weighted averaging was conducted at the spacy token level.
- Postprocess including converting 8-class predictions back into 13-class predictions was done.
- We used deberta-v3-large, deberta-xlarge, and longformer-base. There are two types of models: before postprocess, one with high recall and another with high precision.

# Model
|backbone|weight|additional dataset|class num|type(before pp)|tokenizer|other details|cv/public/private※|
| --- | --- | --- | --- | --- | --- | --- | --- | 
| deberta-v3-large |  0.182|nbroad |13|high recall |default |WeightedCrossEntropy(O’s weight 0.1) |0.969/0.969/0.956 |
| deberta-xlarge |0.073|nbroad |13|high recall |default |WeightedCrossEntropy(O’s weight 0.1) |0.958/0.956/0.955 |
| deberta-xlarge |0.073|nbroad |13|high recall |default |WeightedCrossEntropy(O’s weight 0.01) |0.962/0.965/0.957|
| longformer-base |0.073|nbroad |13|high recall |default |WeightedCrossEntropy(O’s weight 0.1) |0.959/0.959/0.951|
| deberta-v3-large | 0.05|mpware|13|high precision|**Add "\n" and "\n\n" to the special**|SmoothFocalLoss|0.965/??/?? |
| deberta-v3-large | 0.1|nbroad |**8**|high precision|**Add "\n" and "\n\n" to the special**|WeightedCrossEntropy(O’s weight 0.05)|0.965/0.972/0.958|
| deberta-v3-large | 0.27|nbroad |**8**|high recall|default|exclude all negative data|0.966/0.974/0.957|
| deberta-v3-large | 0.18|nbroad |**8**|high precision|default|include all negative data|0.965/??/??|


※Please consider the scores as a reference only, as the post-processing used varies by model.

## About 8 class model（idea by @irrohas）
We built the model with a focus on an 8-class classification (or 13-class classification -> 8-class classification in ensemble) for the following reasons:
- First Name and Last Name do not necessarily correspond to B-NAME_STUDENT and I-STUDENT_NAME, respectively.  B- and I- were simply determined by the order in which they appeared.
- It was easier to convert from 8 classes back to 13 classes during post-processing.
- 8-class classification simplifies the problem for the model.
- we could not completely dismiss the possibility of having I-USERNAME and I-EMAIL in the private leaderboard.
- we can make postprocess easily.

At least in terms of  the public leaderboard, the 8-class classification performed better than the 13-class classification.

## Add "\n" and "\n\n" to the special tokens（idea by @yyykrk）
Deverta-v3-large is unable to tokenize "\n", yet STREET_ADDRESS includes "\n", so we add "\n" and "\n\n" as special tokens in certain models.

`tokenizer.add_tokens(["\n", "\n\n"], special_tokens=True)`

# Acceleration process（idea by @yyykrk）
To include many models in the ensemble, we implemented the following two measures and worked on speeding up the process. 

- Dynamic Padding(https://huggingface.co/learn/nlp-course/chapter3/2#dynamic-padding)
- Applying different batch sizes for each token length

As a result of this process, the inference code became more than twice as fast.(e.g., the inference time for DeBERTa v3 large has been reduced from about 1 hour to about 30 minutes.)

# Postprocess(pp)
①pp before converting 8-class predictions back into 13-class predictions.
- 【high impact】 Averaged over (document, token_str) for NAME_STUDENT.
- 【high impact】Applied a weight to the probability of 'O' (similar to [the discussed approach](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/470978)). The coefficient that resulted in the highest cross-validation score was adopted.
- 【high impact】Assigned STREET_ADDRESS to tokens not included in default DeBERTa-v3 tokens, typically \n, \n\n, enclosed by STREET_ADDRESS.
- Assigned ID_NUM to 12-digit numbers.
- Eliminated specific characters from probabilities.
- Assigned ID_NUM to strings like IV-0000.

②converting 8-class predictions back into 13-class predictions

③pp after converting 8-class predictions back into 13-class predictions

We drop the following predictions:
- "B-NAME_STUDENT" and "I-NAME_STUDENT" predictions that start with lowercase letters.
- "B-URL_PERSONAL" predictions that do not contain "tp".
- "B-EMAIL" predictions that do not contain "@".
- All predictions for whitespace strings other than "\n".
- "B-ID_NUM" predictions for ":" and "-".
