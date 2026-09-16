# [3rd place solution] Blend MLM pretrained DeBERTa & GBM

Competition: linking-writing-processes-to-writing-quality
Rank: #3
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466906

Thanks to kaggle and everyone involved for hosting this competition. It was interesting to see how good we can predict essay scores just from keystroke information. Special thanks to @kawaiicoderuwu for sharing the [essay constructor](https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor) we based much of our solution on.

### TLDR
Our solution is an ensemble of two parts. Firstly, we build a Gradient Boosted Trees (GBT) ensemble based on popular public features on essay level. Secondly, we build an ensemble of deberta models using the final essay text, which have been mlm-pretrained on persuade corpus where we obscured the essays similar to the competitions train set. We weight GBT-Ensemble and deberta-ensemble by a ratio of 40/60.

### Cross validation
We used 8fold cross validation and got a so-so correlation between CV and LB. One important insight is that GBM models have a way better LB/ CV ratio compared to deberta models. Or in other words, Deberta worked better on CV, while GBM were relatively bad on CV, but good on LB. That indicates a slight domain shift between LB and train set. Our guess is the essays might be split by topic or student year. 


### Data preprocessing
Similar to most teams we extracted the final text for each essay, and trained a model to analyse it. Deberta models were quite strong when using the final obscured essay text. We found significant improvement by replacing the obscurification character `q` with `i` or `X`. For both cases one can understand that the pretrained debertas have a better tokenization compared to using `q`. For example a lot of texts the debertas originally have been trained on, have explicit tokens for i, ii, iii, or X, XX, XXX. This also meant the tokenized sequence was shorter than using `q`. Following that rationale, we also trained a custom tokenizer. Models based on that tokenizer helped the ensemble a bit.

### Model
#### GBT ensemble 
For all models within the GBT ensemble we used the same [165 features](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features) as presented in the public kernel by @awqatak (who deserves another special thanks 🙏). We then trained a few models similar to public kernels on those features.

The final components were as following
- LGB (params tuned with optuna)
- XGB (params tuned with optuna)
- CatBoost (params from public kernel)
- Lightautoml (params from public kernel)
- Shallow NN

#### Deberta ensemble 
A key component of our solution is an ensemble of transformer models (Deberta) trained on the reconstructed essay text. 

##### Training procedure:

We found that pretraining the transformer on the persuade corpus with MLM-objective significantly improves CV and LB. For that, essays in the persuade corpus had to be obscured similar to the essays resulting from the train set. Then in a second step we fine-tuned those models on the training data, adding additional features like cursor position, etc and using a Squeezeformer layer to derive semantic features. Adding further keystroke features did not help, and we needed heavy dropout/augmentation on the three we added.


The final components were as following
- deberta-v3-base trained with q replaced by i
- deberta-v3-large trained with q replaced by i (with first 12 layers frozen in finetuning, to avoid overfit)
- deberta-v3-base trained with q replaced by X
- deberta-v3-base trained with custom spm tokenizer 

### Postprocessing
For some models we clipped predictions at [0.5,6.] but it did not really make a difference. 

### Ensembling
Within the Deberta ensemble as well as within the GBM ensemble we used positive Ridge Regression on Out-of-fold (OOF) predictions to determine blending weights. The final weighting of both parts was done manually, and represented if we trust CV or LB more. For final submissions we selected 50/50 kernel and one 40/60 kernel. 

### What did not help
- Using the deleted text
- Stacking
- Adding more keystroke features to the deberta based model. 
- More squeezeformer layers

### Used tools/ repos
- Pytorch
- Huggingface
- Optuna
- Neptune.ai was our MLOps stack to track, compare and share models. 

### Code
Training code repo [here](https://github.com/darraghdog/kaggle-linking-writing-3rd-place-solution).
Inference script [here](https://www.kaggle.com/code/darraghdog/3rd-place-lwq/notebook?scriptVersionId=160142140).
