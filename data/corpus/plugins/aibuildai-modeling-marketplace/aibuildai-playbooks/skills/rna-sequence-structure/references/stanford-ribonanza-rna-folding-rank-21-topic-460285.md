# 21th solution: ESM2 + custom folding head

Competition: stanford-ribonanza-rna-folding
Rank: #21
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460285

**TLDR**

We entered the competition quite late, our first sub was from 10 days ago. Given the limited time, we decided to bet on a low development path and leverage the excellent ESM2 model family from META.

We pretrained ESM2 backbones on competition data and RNA Central long sequences, then added custom folding heads and fine tuned on DMS and 2A3 labels.

We honestly are a bit surprised by our end result, we expected worse when we started. 


**ESM2 Pretraining**

We have some experience pretraining [ESM2](https://github.com/facebookresearch/esm) protein language models, so we thought they would be a good fit for this competition. We pretrained three versions of ESM2 models (35M, 150M, 650M) on both train and test sequences using MLM (masked language modeling) loss. 

In particular, we used huggingface’s [`EsmForMaskedLM`](https://huggingface.co/docs/transformers/model_doc/esm#transformers.EsmForMaskedLM) model class and  [`DataCollatorForLanguageModeling(tokenizer, mlm=True)` ](https://huggingface.co/docs/transformers/main/main_classes/data_collator#transformers.DataCollatorForLanguageModeling) data collator which supports MLM loss.

Since there are no train sequences and only 8000 test sequences longer than 208, we downloaded [RNA Central ](https://rnacentral.org/) data and selected 12 million samples with length between 115 and 457 (the range in this competition data) to complement the competition data.

We pretrained 5-20 epochs depending on how many RNA central data used, for a total of 25M to 70M total samples seen.



**Folding Head**

We adapted the model CPMP used in the Covid Vaccine competition three years ago to devise a folding head on top of ESM2 token embeddings. That model was presented [here](https://www.kaggle.com/competitions/stanford-covid-vaccine/discussion/189723).

The folding head takes as input both the ESM2 token embeddings and the bpp files. Its main component is an attention/convolution layer, repeated twice, then a linear classification head. 

The Attention/Convolution layers are similar to the transformer encoder structure: an attention layer followed by a convolution bloc with skip connection.
The attention layer is a bpp attention layer followed by a bidirectional RNN (GRU or LSTM yields two model variants). Sure, GRU is not attention, but here it plays the same role as an attention layer where each node attends its two neighbors in the sequence.
The bpp attention is a standard attention layer where attention weights are the bpp values.
The convolution layer was designed after the Efficientnet convolution bloc. We then stack these as in this [notebook](https://www.kaggle.com/thebigd8ta/open-vaccine-pytorch-v).

We bet on this RNN/Convolution architecture rather than a transformer for two reasons:
- We literally started working on it 5 days before competition end, and we did not have time to tune a new architecture
- we hoped that ESM2 attention would have already learned what a transformer head attention would learn.

**Finetuning**

Our best models are the ones with the custom folding head on top of the ESM2 backbone. We also trained simple models with just a linear head. The final ensemble is a mix of both, most being with a folding head.

We used four  variants of training data:
Quick start
Quick start + SN >= 0.3
Quick start + SN >= 0.3, and SN as sample weight
Quick start + SN >= 0.1, and SN as sample weight
SN is signal_to_noise clipped to (0, 1). We used it as sample noise in some variants. Model quality increased from top variant to bottom one.

We used a 5 fold CV and submitted the average of each fold model predictions.

**Scores**

Our best single model is a 650M LSTM model with SN weight. CV = 0.12847, Public LB = 0.14308 (5 fold ensemble)
Our best ensemble is a 15 model ensemble (75 folds in total), CV = 0.12478, Public LB =  0.14174

**What didn’t work**

Pretrained models from [RNA-FM](https://github.com/ml4bio/RNA-FM) were not as good as our ESM2 pretrained models.

ESM2 can also output contact predictions. To do so it stacks all attention activations, then it applies a regression head.  We tried to use that to predict bpp values via an auxiliary loss. This improved the competition metric score a bit. But this comes at the expense of a 4x space and a 2x time increase, which prevented us from using it with 150M and 650M variants of ESM2.

Bo & CPMP
