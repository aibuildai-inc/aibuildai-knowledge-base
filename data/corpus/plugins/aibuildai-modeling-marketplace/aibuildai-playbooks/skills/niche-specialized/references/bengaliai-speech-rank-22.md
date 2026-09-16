# 24th Place Solution for the Bengali.AI Speech Recognition Competition

Competition: bengaliai-speech
Rank: #22
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/448119

First of all, we want to thank the organizers who held this wonderful competition. This is my first kaggle competition, I fell satisfied with the final outcome. Thanks my teammate @nisshokuitsuki  , who worked with me during this three months journel. Thanks everyone who contributed to the discussion and the notebooks, your works gave us a lot insperiations. 
---
# Model
The model we used is
```
Wav2Vec2ForCTC
```
We finetuned it further on this model [bengali_wav2vec2_finetuned (kaggle.com)](https://www.kaggle.com/datasets/nischaydnk/bengali-wav2vec2-finetuned)
Since the data quality of the competition dataset is low, we used the train split of [Common Voice 13 | Bengali (Normalized) (kaggle.com)](https://www.kaggle.com/datasets/umongsain/common-voice-13-bengali-normalized?select=train.tsv)to train. We trained about 10 epochs with the 20000 data in the dataset, and splited about 700 data for validation. 
The data here is normalized and removed punctuations.
Here is the training arguments:
```python
training_args = TrainingArguments(
    group_by_length=False,
    weight_decay=0.01,
    num_train_epochs=10,
    fp16=True,
    learning_rate=4e-5,
    warmup_steps=600,
)
```
And we used cosine optimizer.

The best model got local wer 0.15, and Improved the Public Score from *0.445->0.434*

However, my teammate also trained a model with the same data for just 60 steps( different args), and got the same score. (even 0.001 better on the private LB). How interesting and confusing.

# Language Model
We trained a 6gram with [lm_no_punc (kaggle.com)](https://www.kaggle.com/datasets/mbmmurad/lm-no-punc). 
Mention that there is a error in the LM provided in [YellowKing_DLSprint_Model (kaggle.com)](https://www.kaggle.com/datasets/sameen53/yellowking-dlsprint-model). 
```
There is no terminator in the arpa file. So we need to add an </s> into the file.
```
Thanks to this notebook [Build an n-gram with KenLM | MaCro | Kaggle](https://www.kaggle.com/code/umongsain/build-an-n-gram-with-kenlm-macro), we are able to realize this. 
Adding the </s> improved the LB from *0.445->0.422*
And building an 6gram with [lm_no_punc (kaggle.com)](https://www.kaggle.com/datasets/mbmmurad/lm-no-punc).Improved about *0.001*
# Punctuation Restoration
Punctuation really matters. Thanks to this post [Bengali.AI Speech Recognition | Kaggle](https://www.kaggle.com/competitions/bengaliai-speech/discussion/432305), we are able to realize this. And an response under this post showed us a way to restore the punctuation:
[xashru/punctuation-restoration: Punctuation Restoration using Transformer Models for High-and Low-Resource Languages (github.com)](https://github.com/xashru/punctuation-restoration) First we trained a model with the dataset provided in this repositorie. It can restore 3 punctuations : 
```python
{1: ',', 2: '।', 3: '?'}
```

This improved the LB from *0.422->0.400*
Combining with the finetuned model, we have *0.400->0.397*
Afterwards, we thought that 3 punctuations might be not enouth. So we made a dataset with [oscar · Datasets at Hugging Face](https://huggingface.co/datasets/oscar), filterd datas that have only have bengali words. We chose 7
punctuations: 
```python
{1: ',', 2: '।', 3: '?', 4: '!', 5: '-', 6: '"', 7: ':'}
```
We trained 6 epochs with the default pharams.
And we have *0.393->0.387*

# Model ensemble
We simply ensembled our model like this: 
```python
            y = model_1(x).logits*0.7 + model_2(x).logits*0.2 + model_3(x).logits*0.1
```
Wait, This works???
Yes, thouth the predictions may not be aligned, But since the three models are trained on same datasets, the no-aligning problem is paritially solved. 
This improved our performance about *0.001*

# Decoder pharams selection
There are three main pharams for the decoder:
```
alpha: weight for language model during shallow fusion
beta: weight for length score adjustment of during scoring
beam_width: determines the number of candidate output sequences retained at each time step.
```
To find the best pharams, we used optuna [[0.444] Optimize Decoding Parameters with Optuna | Kaggle](https://www.kaggle.com/code/snnclsr/0-444-optimize-decoding-parameters-with-optuna) to search the best pharams. We searched the pharams with the example datas in the dataset, which is ood data, brought us better LB score.
The final decoder pharams are:
```
{'alpha': 0.46570704474381447, 'beta': 0.8635977171858652, 'beam_width': 768}
```


# What doesn't work for us
- Data augmentation. We added background noise downloaded from https://pixabay.com/sound-effects/search/noise/ and also pitch shift , time stretch etc. But The LB got worse (*0.397->0.415*). Every experiment of data augmentation takes too much time and I fells to tired to do more experiments. Maybe I could write some codes to do it automatically
- Denoise model. We tried three denoising models: 
		UVR: Notebook Run Out of time
		[facebookresearch/denoiser](https://github.com/facebookresearch/denoiser): decreased about 0.01
		[CleanUNet](https://github.com/NVIDIA/CleanUNet/blob/main/exp/DNS-large-high/checkpoint/pretrained.pkl): decreased about 0.005
	We thought that denoising harms the features and makes some short syllables unrecornizable.
- Train a bigger LM with more data. We used 15G normalized benglai data to build an kenlm, the score got worse. We still haven't found the cause of the problem.
