# 12th Place Solution - Overview with code files

Competition: prostate-cancer-grade-assessment
Rank: #12
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169637

First of all, I would like to thank all the people and organizations that have made this Competition possible. In capital letters, THANK YOU to all the TEAMS that with their dedication and effort I hope contribute to improve the diagnosis of prostate cancer and thereby improve people's lives. Indeed, my most sincere congratulations to the WINNERS.

I am very happy as you can imagine. In a few lines I share with you a quick overview of my time in this Challenge.

### Kaggle Learning
I want to comment here what is usually included in the acknowledgments part but I reserve this special section to highlight the work of those competitors who have made my final solution better, 1) because their ability was not present in my initial knowledge or 2) because their performance improves together with the experience of mine. I mean, in no order of priority,

- **(Salman)** @micheomaano:
1.  [Dataset tf-record-256-56-48](https://www.kaggle.com/micheomaano/tf-record-256-256-48)
2. [TPU Training Tensorflow Iafoos Method 42x256x256x3](https://www.kaggle.com/micheomaano/tpu-training-tensorflow-iafoos-method-42x256x256x3)
3. [Pandas 42x256x256x3 Inference](https://www.kaggle.com/micheomaano/pandas-42x256x256x3-inference)

- **(Qishen Ha)** @haqishen:
1.  [Train EfficientNet-B0 w/ 36 tiles_256 [LB0.87]](https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87)
2. [PANDA Inference w/ 36 tiles_256](https://www.kaggle.com/haqishen/panda-inference-w-36-tiles-256)

- **(RAHUL SINGH INDA)** @rsinda:
1.  [Panda Inference EfficientNet-b1](https://www.kaggle.com/rsinda/panda-inference-efficientnet-b1)

- **(Iafoss)** @iafoss: The Best Accelerator in the Competition, ahead of TPUs.

###Submission Notebook    
   
I have shared an original copy of my inference kernel without additional cleaning as well as a dataset that includes the necessary weights of each of the models that are used in obtaining the final submission,
    
- [Quick Save Inference](https://www.kaggle.com/coreacasa/12th-place-solution-quick-save-inference)
- [Dataset Model Weights for Inference](https://www.kaggle.com/coreacasa/pandaenetb042x256x256x3)
   
### [TPU] Kaggle/Google(Colaboratory)
For all my trainings I used the free TPU resources offered by Kaggle / Google (Colaboratory). Thank you very much.

### Training One: My only approach to validation

Very closed to Salman's training kernel I just re-ran its code to complete cross validation. I ran a fold up to 60 epochs to see the evolution of the loss and the rest down to 40 epochs.

Individually the behavior of the folds was more or less similar in final loss values (mse) and in the number of times in which it stopped improving. The issue is that its merge did not improve the individual performance of some of them over LB and their performance was also uneven when they were introduced into an external ensemble.

The noise of the labels is a probable cause as already discussed in the discussions or perhaps the sensitivity of the qwk metric to even small variations in mse when its jump to LB.

- **[code-base-training-one](https://www.kaggle.com/coreacasa/code-base-training-one)** file, training topics:

<code>Size Image</code> 256
<code>Size Tiles</code> 256
<code>Tiles</code> 42
<code>Augmentation</code> horizonal p=0.5 and vertical p=0.5 flips
<code>Validation</code> StratifiedKFold 5 on isup grade classes
<code>Arch</code> EfficientNetB0
<code>Convolutional Base's Weight</code> Imagenet trainable
<code>On Top</code> GlobalAveragePooling2D, Dropout(0.5), Dense(1024)
<code>Output</code> Dense(1) regression objective
<code>Loss</code> mean_squared_error
<code>Optimizer</code> Adam
<code>Leaning Rate</code> 5e-04 init
<code>Reduce LR</code> decreasing 0.5 with patience 3 epochs
<code>Save</code> weights only with best validation loss epochs
<code>Batch Size</code> 64

### Training Two: Art(Instinct) Validation

I never tried detecting noisy labels to remove them from training data. In general I am not in favor of losing any existing information, although in principle it could be harmful by elevating the non-regular part of a data generating process. I would rather transform data than remove it.

I didn't try either any transformation so I thought about training the models with full dataset in order to prevent the possible existence of more noise in some folds than in others, which probably would be increasing the variability in the inference results.

Art Validation appears here and it is when the art of the data scientist enters and it is his instinct that determines the goodness of fit and stability of performance in generalization against new observations. Yes, this is Alchemy.

- **[code-base-training-two-enets](https://www.kaggle.com/coreacasa/code-base-training-two-enets)** file, from which I trained 3 members of the EfficientNet family. Changes on training one training topics:

<code>Tiles</code> 48
<code>Validation</code> Art Validation on instinct
<code>Arch</code> EfficientNetB0, EfficientNetB1 and EfficientNetB2
<code>Convolutional Base's Weight</code> Noisy Student trainable
<code>Output</code> Dense(5,activation='sigmoid) ordinal regression objective
<code>Loss</code> sigmoid_cross_entropy_with_logits
<code>Leaning Rate</code> custom with 5up, 3sustain, 0.8decay
<code>Limits LR</code> 1e-05min, 4e-04max
<code>Save</code> weights only with best loss epochs
<code>Batch Size</code> 32
<code>Epochs</code> 60

- **[code-base-training-two-densenet](https://www.kaggle.com/coreacasa/code-base-training-two-densenet)** file, from which I trained 1 member of the DenseNet family. Changes on training topics of the previous net family:

<code>Arch</code> Densenet121
<code>Convolutional Base's Weight</code> Imagenet trainable
<code>Epochs</code> 40

### Inference: Diversity of Archs, nTiles and TTAs

Of the 2 training processes shown above, the following models were available,
1. EfficientNetB0 (5 skf), 42x256x256x3
2. EfficientNetB0 (1), 48x256x256x3 
3. EfficientNetB1 (1), 48x256x256x3 
4. EfficientNetB2 (1), 48x256x256x3 
5. DenseNet121 (1), 48x256x256x3 

Having re-run the Salman kernel, from the public notebooks referenced at the beginning I had,
1. EfficientNetB0 (1 skf), 36x256x256x3 (Qishen Ha) 
2. EfficientNetB1 (1 skf), 36x256x256x3 (RAHUL SINGH INDA)
    
- **Test Time Augmentation**
<code>Type A: 5xTTA deterministic</code> 
1xoriginal, 1xTranspose, 1xVerticalFlip, 1xHorizontalFlip, 1xTranspose-&gt;VerticalFlip-&gt;HorizontalFlip
<code>Type B: 4xTTA pseudo deterministic</code> 
1xoriginal, 1xVerticalFlip, 2xHorizontalFlip(p=0.5)-&gt;VerticalFlip(p=0.5)
<code>Type C: 2xTTA random </code> 
2xHorizontalFlip(p=0.5)-&gt;VerticalFlip(p=0)

- **White Padding Tile Extraction (Qishen modes)**
1x add zero pad and 1x add 256 pad, that is, 2 different extractions for ALL the images.

### Model Selection and Final Ensemble

<pre><code>(3/10)*Public-Quishen [TTA Type A]  
(3/10)*Public-RAHUL SINGH INDA [TTA Type A] 

(1/30)*EfficientNetB0-Fold0-Training One [TTA Type C] 
(1/30)*EfficientNetB0-Fold2-Training One [TTA Type C] 
(1/30)*EfficientNetB0-Fold4-Training One [TTA Type C] 

(1/15)*EfficientNetB0-Training Two [TTA Type C] 
(1/15)*EfficientNetB1-Training Two [TTA Type C] 
(1/15)*EfficientNetB2-Training Two [TTA Type C] 

(1/10)*DenseNet121-Training Two [TTA Type B] 
</code></pre>

The random component of the TTAs was not seed (I'll be lucky) and the reproducibility of the results may vary with it. I have just re-run my inference kernel and the results are Private Score 0.92983 (0.92960 original) and Public Score 0.89443 (089352 original).

With this models structure I was only able to test the last day of the competition. For example, this other ensemble got Private Score 0.93047 and Public Score 0.88889, not including random component in TTA.

<pre><code>(3.5/10)*Public-Quishen [TTA Type A]  
(3.5/10)*Public-RAHUL SINGH INDA [TTA Type A] 

(1/15)*EfficientNetB0-Training Two [TTA Type A] 
(1/15)*EfficientNetB1-Training Two [TTA Type A] 
(1/15)*EfficientNetB2-Training Two [TTA Type A] 

(1/10)*DenseNet121-Training Two [TTA Type A] 
</code></pre>

One more, my last submission and that finished tight after the deadline got Private Score 0.93052 and Public Score 0.89110,

<pre><code>(3.5/10)*Public-Quishen [TTA Type A]  
(3.5/10)*Public-RAHUL SINGH INDA [TTA Type A] 

(1/30)*EfficientNetB0-Fold0-Training One [TTA Type C] 
(1/30)*EfficientNetB0-Fold2-Training One [TTA Type C] 
(1/30)*EfficientNetB0-Fold4-Training One [TTA Type C] 

(1/15)*EfficientNetB0-Training Two [TTA Type C] 
(1/15)*EfficientNetB1-Training Two [TTA Type C] 
(1/15)*EfficientNetB2-Training Two [TTA Type C] 
</code></pre>




    
### That is all, Thanks a lot!
By the way, I still tremble with fear
Update: No longer!
