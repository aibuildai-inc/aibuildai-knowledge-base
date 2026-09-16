# [The 11th place] Shallow encoder-decoder model also works

Competition: asl-fingerspelling
Rank: #11
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434475

Acknowledgments for the great public notebooks:
[1] [MARK WIJKHUIZEN's data preprocessing notebook ](https://www.kaggle.com/code/markwijkhuizen/aslfr-eda-preprocessing-dataset)
[2] [HOYSO48's first place notebook from ISLR](https://www.kaggle.com/competitions/asl-signs/discussion/406684)

Reproducible Codes:
1. [Data preprocessing notebook](https://www.kaggle.com/code/baohaoliao/aslfr-data-preprocessing)
2. [Training codes](https://github.com/BaohaoLiao/aslfrv4): I need some time to clean it and will guide you in the following. In case someone can't wait to check it.


# Data preprocessing 
My data preprocessing and split notebook is based on [1]. The differences are:
1. I used the same landmarks (only x and y) from [2].
2. I split data based on unique phrases into 10 folds, and use the first fold as the validation set. There are no shared phrases in different folds. For data splitting, I found unique phrase split = random split > participant id split.
3. Use the same way to preprocess the supplementary data.

# Data augmentation ([file](https://github.com/BaohaoLiao/aslfrv4/blob/main/mask/datav2_distributed.py))
I almost used the same data augmentation and hyper-parameters as [2]. The main differences are:
1. I randomly concatenated two samples with a prob of 50%. I found it useful for overfitting.
2. For the input to the decoder, I randomly (20%) replaced some ground-truth characters with random characters from the vocab. I apply this trick to avoid exposure bias.

# Model architecture ([file](https://github.com/BaohaoLiao/aslfrv4/blob/main/auto_ctc/conformerencoder_transformerdecoder_mask_droppath_ctc.py))
I use Conformer as the encoder and the vanilla Transformer decoder. Conformer is better than the vanilla Transformer encoder, since it focuses on both local and global relations. The main highlights here are:
1. I use both CTC loss and cross-entropy loss. After the encoder, the CTC loss is applied to the encoder output. And cross-entropy loss is applied to the decoder output. The interpolation weights for these two losses are 0.2 for CTC loss and 0.8 for cross-entropy loss. The benefits of this setting are: (1) One trained model can be used in two ways, either non-autoregressive generation with the encoder or autoregressive generation with the decoder. In the end, both generations achieved the same 0.791 public LB, (2) One loss could be a regularization term to the other.  When you only want to do non-autoregressive generation, you can throw away the decoder parameters, which allows you to use more parameters during training.

2. The hyper-parameters are: 5-layer encoder with hidden_dim=384, mlp_dim=1024, conv_dim=768, num_heads=6, 3-layer decoder with hidden_dim=256, mlp_dim=512, num_heads=4. 

# Three-phase training ([file0](https://github.com/BaohaoLiao/aslfrv4/blob/main/train_cnnencoder_transformerdecoder_mask_ctc_distributed.py) and [file1](https://github.com/BaohaoLiao/aslfrv4/blob/main/train_cnnencoder_transformerdecoder_mask_ctc_awp_distributed.py))
All training uses AdamW, inverse square root schedule, weight decay=0.001 and max norm=5, lr=5e-4, batch size=512, warmup ratio=0.2, label smoothing=0.1, frame_length=368.
1. For the first phase, I only train on the training set and exclude the validation set with #epoch=100.
2. For the second phase, I include all training data and supplemental data for another 150 epochs.
3. For the third phase, I use AWP with awp_delta=0.2 and awp_eps=0 on all training data for 300 epochs. AWP is good for generalization, better than rdrop for my case.


# What doesn't work
1. BPE: I try to use subwords rather than characters, but it doesn't work.
2. Too wide but shallow model: For the abovementioned model, I quantize the model in FP16. The number model's parameters are about 18M, 37MB. We can use about 40M parameters if we use dynamic quantization. But dynamic quantization slows down the inference speed. For your reference, my FP16 CTC model runs 2h30m, while the dynamic quantized one runs about 4h. There is only 0.001 LB performance drop for the dynamic quantized one. Nothing drops from FP32 to FP16. In the last week, I tried to use an 8-layer encoder and a 4-layer decoder (same dimension as above). But I have to use frame_length=100 to reduce the inference time within 5 hours. The results are not good. I think there might be some overfitting with such a larger model.
3. Ensemble between encoder and decoder: My model could do both autoregressive and non-autoregressive generation at the same time. I first generate the output with the encoder, and do an average on the probabilities of the autoregressive and non-autoregressive output at each time frame. But the result stays the same.
4. Other data: I used [ChicagoFSWild and ChicagoFSWild+](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm#download), it doesn't help. Domain shift is the main reason.

# What could make my model better
1. Inspired by other top-rank methods, including pose landmarks and z coordination might be better.
2. Use a deeper but narrower model.
