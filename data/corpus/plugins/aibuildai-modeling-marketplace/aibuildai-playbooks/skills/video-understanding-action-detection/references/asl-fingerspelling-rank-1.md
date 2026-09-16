# [1st place solution] Improved Squeezeformer + TransformerDecoder + Clever augmentations

Competition: asl-fingerspelling
Rank: #1
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434485

Thanks to kaggle and everyone involved for hosting such an interesting competition. It was a great extension to the isolated sign language classification and it was very interesting to see how much of speech-to-text research could also be applied to sign language fingerspelling. As always it was a great teaming experience with @darraghdog 

## TLDR
Our solution is based on a single encoder-decoder architecture. The encoder is a significantly improved version of Squeezeformer, where the feature extraction was adapted to handle mediapipe landmarks instead of speech signals. The decoder is a simple 2-layer transformer. We additionally predicted a confidence score to identify corrupted examples which can be useful for post-processing. We also introduced efficient and creative augmentations to regularize the model, where the most important ones were CutMix, FingerDropout and TimeStretch, DecoderInput Masking. We used pytorch for developing and training our models and then manually translated model architecture and ported weights to tensorflow from which we exported to tf-lite.

## Cross validation
We split the training data into 4 folds by signer. In the beginning we had nearly perfect correlation between CV and public LB with this approach. With higher scores improvements on CV reflected a bit less on LB, mostly due to the fact that the LB score was always decently higher and hence saturated earlier. Most of the time we only trained and tracked the score of fold0 and not all folds. 

## Data preprocessing
In total 130 key points were used. These consisted of 21 key points from each hand, 6 pose key points from each arm, and the remaining 76 from the face (lips, nose, eyes). Locally the 130 key points were cached to .npy files for fast data loading. 
Prior to data augmentations, the data was normalized with std/mean and nans were zero filled. 

## Augmentations
Augmentations were essential to prevent overfitting, generalize to new signers and enable deep models. We used augmentations which were popular in the first ASL competition but also came up with a lot of new creative augmentations and some have proven to be very effective. 

- Resizing along time axis.
- Shift the sequence along the time axis. 
- Windowed resizing along time axis (similar to warping).
- Left-right flip of keypoints. 
- Cutmix of samples timewise - draw a random percentage between [0,1] and cut 2 sequences and related phrases at that percentage and mix. Mixing only within same signer was best
- Spatial affine - scale, shear, shift and rotate. 
- Drop/Zero-fill between 2 to 6 different fingers over 2 to 3 time windows. 
- Drop/Zero-fill either all face landmarks or all pose landmarks. 
- In rare cases (~5% of samples) drop/zero-fill all hand landmarks. 
- Temporal masking (zero-fill) in windows of different sizes or counts. 
- Spatial masking 

Most augmentations were applied to 50% of the samples, except for resizing and spatial affine which were applied to ~80% of samples.  

After augmentation, samples with more than 384 frames were resized along time axis, with channel-wise linear interpolation. Samples of less than 384 were padded to 384 for training only. tf-lite ran on variable length samples. 
No frames were dropped in preprocessing. 

## Model
In general, we observed that a deeper model gives significant gains (if we are able to prevent overfitting). As a consequence not only regularization techniques like augmentations are essential, but also every improvement in computational efficiency creates space to use deeper models and hence is equally important as the model architecture itself.

Our model consists of 3 parts, Feature Extraction, Encoder, Decoder which are shown in the image below



We interpret the data like a 3 channel image, where width is defined by the number of frames, height is given by the number of the selected 130 landmarks and channels are given by raw xyz coordinates. 
The feature extraction is based on a 2D convolution followed by batchnorm and a linear layer on the flattened features to extract features per frame. We have 5 of this feature extraction modules, one for all landmarks at once and one per landmark type (left_hand, right_hand, face, pose). The all-landmark module outputs 208 dim vector/ frame. The other 4 output 52 dim vectors each which are then concatenated to have also 208 dims. We had those two 208-dim vectors per frame and get a (batch_size x 384 x 208) input for our encoder where 384 is the maximum sequence length we chose.

The main component of our model is an encoder which was adapted from the Squeezeformer architecture. We did not use the actual “squeeze” idea, i.e. a temporal Unet, but used the general architecture of Squeezeformer Blocks which consist of a combination of MultiHeadSelfAttention (MHSA), Convolution and FeedForward modules. We made several improvements to this architecture:
ASR conformers (and Squeezeformer) use relative positional encoding which allow the self-attention module to generalize better on different input lengths. Relative positional encoding is performance intensive, as well as using many parameters, as they are stored separately in each layer. Replacing this with Llama attention which uses rotary embeddings sped up training ~2X and tf-lite inference approx ~3X allowing larger models to be used. In addition, we cached the rotary embeddings once and fed them into each layer with the input data so they are not duplicated in each layer. This resulted in 20% less parameters in the model. We saw no benefit in using time reduction which was introduced with Squeezeformer. So in our model all layers had the same sequence length as the original input. As suggested in the Squeezeformer paper, the pre-Layer Norm from the Macaron structure is redundant and was replaced with a learnable scaling layer which scales and shifts the activations. 

For decoding we used a simple 2 layer transformer decoder which is similar to hugging faces [Speech2TextDecoder](https://github.com/huggingface/transformers/blob/main/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L857), which outputs a sequence prediction. We then used cross entropy loss for training our model End2end. An extra cross entropy auxiliary loss of the reversed sequence was used. A causal decoder mainly uses encoder cross attentions for the sequence's beginning and previous characters and cross attentions for the end. To improve the model's accuracy, we use a separate causal decoder on the reversed sequence as an auxiliary loss, making the model rely more on the encoder cross attention for the label's end. For decoder inference early stopping and past key value caching was used which sped up inference significantly.  It should be noted that we found a transformer based Decoder superior to a CTC based decoding, even in a setting where computational efficiency matters a lot. 

Additionally to the decoder, we also added a single linear layer to take the features of the first token of the encoder output to predict a confidence score, which helps to identify garbage data and can be used in post-processing. As a target for this we used normalized levensthein distance clipped to [0,1] of OOF predictions of a decent previous model.

We explored and trained all our models in pytorch, but whenever we deemed it good enough for a submission we translated each component manually to tensorflow and ported weights from our pytorch models. 

## Training procedure
Models were trained with a cosine learning rate schedule for 400 epochs with peak LR of 0.0045, weight decay of 0.08, 10 epochs warmup, mixed precision and an effective batch size of 512 samples. Dropout of 0.1 was used in the transformer encoder/decoder layers. It was important to train with mixed precision in order to leverage fp16 inference without performance drop. Training with fp32 and using tf-lite fp16 inference causes a drop of ~0.01 in CV vs LB. 

tf-lite inference used a single sample without padding, while model training was performed with time padded mini-batches. To avoid the model learning with pads, and ensure optimal inference runtime, the feature extractor and Macaron structure encoder layers were masked time wise during training. This needed to be manually implemented in pytorch on each layer. This took some effort, but paid off by significantly speeding up inference. As [1st place team](https://www.kaggle.com/competitions/asl-signs/discussion/406684) in the previous ISL competition explained, this is much easier to do in tensorflow as keras has an off the shelf masking layer. 

Model training and tf-lite inference ran in fp16, so tf-lite files consumed almost half the disk size of of fp32. This was important, as the 40MB size was our limitation in the end. The two final model seeds measured 39988kb. 

## Postprocessing
The main idea of our postprocessing is to replace poor predictions with a dummy phrase which has a small levensthein distance to the train/ test data. @anokas showed in his [notebook](https://www.kaggle.com/code/anokas/static-greedy-baseline-0-157-lb) why '2 a-e -aroe' is a good candidate for that. Most of the poor prediction resulted from corrupted input data often only a few frames long. We used a confidence score predicted by our model as basis. Whenever the confidence score is below 0.15 or the sequence is shorter than 15 frames we replace the prediction with '2 a-e -aroe'.

## Supplemental Data
We only marginally profited from using the supplemental data. We think the main reason is that although there are 50k samples in this supplemental data there are only 500 unique phrases, and hence the model rather learns to classify then to actually decode character-by-character. We tried a lot of approaches but only the following one gave a small boost (0.838 -> 0.839): 
First we group the supplemental data by phrase, which only leaves us with 500 groups. In each epoch of training we add one sample per group to the training dataset for our model. That means in each epoch we use 50k samples of the training data and only 500 samples of the supplemental data. 

## Ensembling

Our final submission is a 2-seed ensemble of our model trained on the complete training data (fullfit). We average resulting logits in each decoding step for ensembling.

## What did not help

- Fully using supplemental data
- Using edit distance as loss (tried different approaches)
- CTC loss (even as an auxiliary loss it hurt score)
- Label smoothing
- AWP - kept getting nans with FP16
- TTA (flip/stretch)
- Mixup of hidden layers & Specaugment++
- Beam search decoding (too costly)

## Ablation study (roughly)

#### Augmentations
- Cutmix +0.005
- FingerDropout +0.005
- Face/PoseDropout +0.005
- masking decoder inputs +0.003

#### Model improvements
- CNN Feature extraction +0.005
- 2-branch Feature extraction with indiv norm +0.003
- Squeezeformer over 1stplace Net of 1 round +0.005
- Decoder over CTC +0.003
- Confidence over simple rules for post-processing +0.002


#### Efficiency Improvements:
- Deeper model due to fp16 +0.003
- Deeper model due to llama attention +0.003
- Deeper model due to masking/ variable sequence len +0.005
- Deeper model due to caching/ early stopping in decoder +0.005

#### Postprocessing
- Replace bad predictions with dummy phrase +0.006


## Used tools/ repos
- Pytorch/Tensorflow/Tf-lite (no onnx this time)
- Huggingface
- Albumentations (adapted their framework for using things like OneOf or Compose, but wrote our own augmentation implementations)
- Neptune.ai was our MLOps stack to track compare and share models. Below are example training runs using different model parameters and hardware (a100 card vs kaggle kernel). The data loading in the kaggle kernel below was slow and could probably be sped up with some work.  



## Code & model weights

https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution

## Paper
tbd.  Due to the novelty of our approach we are thinking about summarizing it in a paper.

**Thank you for reading, questions welcome**
