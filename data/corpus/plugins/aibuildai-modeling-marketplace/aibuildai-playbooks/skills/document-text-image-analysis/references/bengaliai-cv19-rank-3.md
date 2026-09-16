# 3rd place solution

Competition: bengaliai-cv19
Rank: #3
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/135982

Congrats everyone with excellent result.

Here is our solution summary.




# Solution summary
## Dataset
* <strong>pretrain</strong>: triple identities by hflip and vflip
* image resolution: 137 x236

## Model
<strong>for seen grapheme and unseen grapheme</strong>
* (phalanx) encoder -&gt; gempool -&gt; batch_norm -&gt; fc
* (earhian) encoder -&gt; avgpool -&gt; batch_norm -&gt; dropout -&gt; fc

<strong>arcface</strong>
* encoder -&gt; avgpool -&gt; conv1d -&gt; bn
* s 32(train), 1.0(test)
* m 0.5

<strong>stacking</strong>
* conv -&gt; relu -&gt; dropout -&gt; fc

## Augmentation
* cutmix
* shift, scale, rotate, shear

## Training
* <strong>seen grapheme</strong>: train model for seen grapheme with pretrain dataset, then finetune it with original dataset
* <strong>arcface and unseen grapheme</strong>: train pretrained model for seen grapheme with original dataset
* replace softmax with [pc-softmax](https://arxiv.org/abs/1911.10688)
* loss function: negative log likelihood
* SGD with CosineAnnealing
* Stochastic Weighted Average


## Inference
<strong>arcface</strong>
* use cosine similarity between train and test embedding feature
* threshold: smallest cosine similarity between train and validation embedding feature

<strong></strong>
