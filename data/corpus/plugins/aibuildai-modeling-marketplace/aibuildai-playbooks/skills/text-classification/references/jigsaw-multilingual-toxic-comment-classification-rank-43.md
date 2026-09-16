# 43rd place. Sampling with Adversarial Validation

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #43
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/161103

Thanks to Kaggle and all kagglers who were generous enough to share their expertise. The learning was rich thanks to:  Dezso Ribli's @riblidezso -   [Finetune XLM-Roberta on Jigsaw test data with MLM](https://www.kaggle.com/riblidezso/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm), Alex Shonenkov @shonenkov with Pytorch [[TPU-Training] Super Fast XLMRoberta](https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta), DimitreOliveira @dimitreoliveira - [Jigsaw - TPU optimized training loops](https://www.kaggle.com/dimitreoliveira/jigsaw-tpu-optimized-training-loops), Abhishek @abhishek with his invaluable 'real-time coding' youtube videos and [I Like Clean TPU Training Kernels &amp; I Can Not Lie](https://www.kaggle.com/abhishek/i-like-clean-tpu-training-kernels-i-can-not-lie), Xhlulu @xhlulu - [Jigsaw TPU: XLM-Roberta](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta), Michael Kazachok’s @miklgr500 translations [dataset](https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api) and  [Jigsaw TPU: BERT with Huggingface and Keras](https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-with-huggingface-and-keras)

### Adversarial Validation.

The main difference of the solution is the way the training and validation sets were sampled. I used adversarial validation in this [kernel (previous version)](https://www.kaggle.com/isakev/jigsaw-adversarial-validation-folds-0-1) to sample [translations](https://www.kaggle.com/miklgr500/jigsaw-train-multilingual-coments-google-api) by Michael Kazachok to pick samples ‘most similar to test set’ (280k-480k samples for training and 4k samples for validation (+8k original validation.csv). Adversarial Validation, the idea successfully used often across Kaggle (e.g. [Quora Adversarial Validation
](https://www.kaggle.com/tunguz/quora-adversarial-validation) and [Adversarial validation](https://www.kaggle.com/konradb/adversarial-validation)

- selecting samples this way, at least, did not make the performance worse than lucky picks of random sampling

### The rest.

The best submission is the blend of predictions of 6 models, all Roberta-XLM-large MLM, 3 of them are based on the MLM finetuned to test set by  @riblidezso  Dezso Ribli (link)[https://www.kaggle.com/riblidezso/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm].
Diversity to models comes from different loss functions (apart from BCE, used focal loss and MSE (with soft labels)), 3 models with 4 top hidden layers’ outputs concatenated, 1 model sums those 4 layers. 3 models use differential learning rates for head and transformers as in [this kernel](https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large), different lengths 192 and 256, for those samples that exceed max_len=192 - concatenating the last 25%*max_len of text to the beginning of text,,  Alex Shonenkov’s [kernel](https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta) with addition of opensubtitles data,  training on english text and without, using training data from 200k to 480k filtered by similarity to test set and/or similarity by language ditribution and/or similarity by ‘toxic’ target distribution.

All were run in tensorflow and keras on TPU only. The GPU had been used for predictions only, or as in the case of Adversarial Validation when smaller Roberata-base model was employed.
