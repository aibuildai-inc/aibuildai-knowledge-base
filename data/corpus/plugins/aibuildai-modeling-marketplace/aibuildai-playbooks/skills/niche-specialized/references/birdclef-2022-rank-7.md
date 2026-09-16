# 7th place solution

Competition: birdclef-2022
Rank: #7
Source: https://www.kaggle.com/c/birdclef-2022/discussion/326973

### Summary
- **Constant Q-transform**
- Multistep training with PL
- Pretraining on 2022+2021 data with noise from Rainforest
- Finetuning on 2022 data with weighted sampling (w~N^0.5)
- Sequence-based model
- Single model performance **0.7971/0.7859** private/public LB, while two models give **0.7964/0.7936**

### Introduction
Congratulations to all participants and thanks to the organizers for making this competition possible. It is my 3rd BirdCLEF competition, and the first gold medal I got in it. Though, this time I again followed the tradition of joining the competition close to the end... with the first sub 5-6 days before the deadline( Something should be changed in this life.

I'll take this opportunity and describe some of the ideas I came up with while working on the challenge. I hope they may be interesting for other participants and organizers as well. My work, to a large extent, is based on my [BirdCLEF 2021 paper](http://ceur-ws.org/Vol-2936/paper-141.pdf). Also, some details may be found in my [BirdCLEF 2021](https://www.kaggle.com/competitions/birdclef-2021/discussion/243343) and [2020 Cornell Birdcall](https://www.kaggle.com/c/birdsong-recognition/discussion/183258) writeups.

### Data
In this year's competition there were two main challenges to address:  **(1) considerable domain mismatch with very noisy test data** and (2) **long-tail class distribution with a few examples of rare classes**. 

To address the first challenge, I used a traditional **pseudo label (PL) 2-step training**: in the first pass the model learns where the signal from birds is located, and on the second pass a heavy noise is applied. With segmentation labels the model knows where to look for the true signal under a severe noise. As a noise source [rainforest data](https://www.kaggle.com/c/rfcx-species-audio-detection/data) is used in addition to pink noise and signal weakening.

To address the second challenge and make the model robust to different bird calls I pretrained it on a large **2021+2022 BirdCLEF dataset** (the size of 2022 data is ~ 5 times smaller). During fine-tuning I kept the backbone frozen and trained only the head on 2022 data. I assigned all unscored classes to "unknown" label and performed weighted sampling with probability conditioned on the number of samples of each class as w~N^0.5.

One of the novel things I used in my solution, in comparison to traditional audio models, is **Constant Q-transform** instead of mel-scale spectrograms and even pcen. The quality of the produced spectrograms is much better, as shown in the figure below: lower impact of noise, high level of details in the high-frequency domain, high contrast, and easy mapping for sequence models (without the need to care about padding added in FFT). However, I didn't have time for a quantitative comparison of the model performance. CQT1992v2 is a very effective Pytorch (GPU) implementation of CQT from nnAudio.Spectrogram library and can be easily integrated into a model (make sure to disable gradients for CQT).  During training I used log transformation of transform output: `x = (64*x + 1).log()`.


```
self.qtransform = CQT1992v2(sr=32000, fmin=256, n_bins=160, 
hop_length=250, output_format='Magnitude', norm=1,
window='tukey',bins_per_octave=27,)
```

### Model
I used [semisupervised pretrained](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models) ResNeXt50 and MiT-B2 transformer, [Segformer ](https://github.com/NVlabs/SegFormer) backbone. The last conv layer is followed by a convolution, collapsing the feature map into a sequence of vectors vector (nemb=1024), with a Transformer layer, and a head, producing sequence and clip outputs using the attention mechanism (somewhat similar to SED models):
```
class AttHead(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.attn = nn.Conv1d(n_in,n_out,1)
        self.cla = nn.Conv1d(n_in,n_out,1)
        
    def forward(self, x):
        if len(x.shape) == 4: x = x.flatten(1,2)
        attn = self.attn(x)
        cla = self.cla(x)
        x = (torch.softmax(attn,-1)*cla).sum(-1)
        return x, attn, cla
```
Initially, I believed that using MiT transformer backbone with a large receptive field is important: it is able to look into the entire frequency domain at one (instead of a small window for conv net) and compare bird calls in the clip across the time. But it appeared to be not really true, and my favorite ResNeXt50 performed nearly the same as MiT B2... both CV and LB. Though, in both cases using a transformer head quite helps based on my initial tests.  

### Training
I used multistep training: (1) pretrain the model on 2022+2021 data and generate PL; (2) train the model on 2022+2021 data with a high level of noise added and additional segmentation loss based on PL (global labels for selected chunks are also adjusted based on PL in case if there is no a bird call); (3) take the model from step 2 and freeze the backbone, finetune on 2022 data with weighted sampling considering 21+1 classes, and generate new PL; (4)  the same as 3 but noise and segmentation losses are added. 

At all stages MixUp augmentation (with label max and alpha of 2, i.e. the probabilities are close to 0.5) is applied to waves. Each stage is started with using 5s chunks (32 epochs) and is finished with a few epochs with 10-15s chunks. I use Focal loss with gamma=1 (with a bug fix to accept smooth labels).

### Inference
The inference is performed on entire audio files (while training is done on 5s clips followed with fine-tuning on 10-15s clips). I use sequence level output, illustrated in the image below for the test audio clip. It is split into 5s chinks, and the predictions are selected if the maximum output of the particular class reaches the selected threshold. If one compares the plot with similar plots from my previous reports, the model performance improvement is quite clear. Now the model is able to clearly distinguish bird calls even in very noisy audio. For postprocessing the model output I was using the following: `x = (torch.softmax(3*attn,-2)*torch.sigmoid(3*cla))`. Pay attention to the dimension, which is the class rather than the sequence dimension: if the model is paying more attention to a specific class at a particular moment, the signal should be enhanced accordingly. Also, I'm using temperature rescaling.


The final submission is composed of 2 models (MiT and ResNeXt50) and is scored as 0.7964/0.7936 at public/private LB. The best single model is scored as 0.7971/0.7859. In this competition, the main focus is bird classification, and no credit is given for call/nocall separation, in contrast to 2020 and 2021 competitions. So the best scores could be achieved at the selection of very low thresholds. If more reasonable thresholds, capable of nocall separation, are used, the performance drops by ~0.05.

### Things ~~didn't work~~ I didn't have time to make work
- Incorporation of ArcFace loss into attention pooling head (dealing with sequence and global labels).
- Performing metric learning and clustering to address the issue with a few training examples for rare classes

I think those two things might be the key to getting to the top of the LB.
