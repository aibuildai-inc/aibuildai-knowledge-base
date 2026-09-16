# 9th place solution

Competition: birdsong-recognition
Rank: #9
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183315

I would like to thank the hosts for this unique challenge. Congratulations to my teammate @canalici and to all competitors. It was indeed a very educative competition in the audio domain. Many thanks to @hidehisaarai1213 guidance through competition, and @doanquanvietnamca for GRU-SED and Dmytro Karabash for ideas.

##Data Augmentation
Models have trained in both original and extended datasets. Augmentations applied in both waveform and mel-spectrogram level.

- Gaussian Noise
- SpecAug

##Modeling
I have modified the SED model (PANN’s) and replace its inefficient backbone with a noisy-Efficientnet and further experimented with GRU’s, LSTM’s, and with Transformers for temporal modeling. We had a CNN backbone -> a GRU layer -> and attention layer in the final model. 

- EfficientNet-B4 (Noisy Student) 
- EfficientNet-B7 (Noisy Student)
- EfficientNet-B7 (Noisy Student)

To ensemble different solutions, I have removed the attention layer for each model, kept pre-trained weights of the extracted part, and re-trained an attention layer from features extracted from 3 different models listed above.


##Training
- Batch size of 32 for B4, and 8 for B7 models (single GPU)
- No mixup :( 
- BCELoss 
- AdamW with Cosine Anneal
- 5 seconds of audio clips (501, 64) (scaling mel_bins into 224, 300 would really have helped, but very expensive)
- Pre-training on primary labels, fine tuning with secondary labels

For validation, we have hand-labeled no-calls into gt_birdclef2020_validation_data and excluded irrelevant species, which provided a chance to test the algorithm in the wild. 
 
 
###Possible further work
Pre-trained models proven self to be leverage in many knowledge transfer tasks. It is tough not to over-fit our classifier, especially in this competition, where we had a few audio clips with very noisy labels. Hidehisa Arai pointed out PANN’s(one of the largest pre-trained models in the audio domain) for this issue, their CNN backbone was less potent than lighter alternatives. We have used a firm CNN backbone to overcome this issue, pre-trained on a large corpus of images (Noisy Student, Efficientnet). However, it is possible to extract mel-spectrogram encoder/decoder parts from very famous text-to-speech, speech conversation (Tacotron, Glow TTS...) algorithms that trained on a relatively larger corpus. It could be beneficial to adapt successfully pre-trained models from the Audio domain, fine-tuning it with all the bird data we have, then applying a noisy-student training scheme.
