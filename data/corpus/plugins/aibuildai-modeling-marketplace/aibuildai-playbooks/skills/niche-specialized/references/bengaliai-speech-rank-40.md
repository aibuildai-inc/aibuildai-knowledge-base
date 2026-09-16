# 40th Place Solution without External Dataset!

Competition: bengaliai-speech
Rank: #40
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/450531

We achieved 40th Place (Silver Medal). Congratulations to all of Team Members from **BengalX**: @iftekharamin , @mdfahimreshm , @fahimshahriarkhan 🎉

*I would like to acknowledge my Team Lead @iftekharamin Bhaiya for giving opportunity to do this competition and giving proper guideline for achieving silver medal.*

**Dataset:** We subset the dataset Based on @imtiazprio published train metadata. We used condition to filter clean dataset from train meta features (yellowking_preds & google_preds) wer = 90% similar. After this we further filter dataset which mos_pred > 2. And we found around 100k+ datapoints.

**Data Cleaning:** We filter audios which duration is less then 1 sec. as outlier and to not mislead model performance.

**Augmentations:** We used audio augmentations i.e, Noise , Background sound mixing, Speed up-down, SpecAug, Changing different Sampling Rates. 

**STT Modeling:** We used Indic wav2vec2 pretrained model. And we finetune with the Subset augmented dataset. 

**Post-processing- LM Decode:** We used arijit indic pretrained KenLM.

**Post-processing-Punctuation:** We used xashru/punctuation-restoration repo with xlm-roberta-base model and fine tune this competition dataset as punctuation restoration. We only consider 4 punctuation classes : {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3} 

**Post-processing-Erro Correction:** We used this repo solution as further error correction https://github.com/Tawkat/Bengali-Spell-Checker-and-Auto-Correction-Suggestion-for-MS-Word
