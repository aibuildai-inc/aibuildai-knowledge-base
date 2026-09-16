# Our solution and source code (0.22 Public 0.17 Private)

Competition: landmark-recognition-challenge
Rank: #24
Source: https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57913#latest-336756

Overall approach

**CNN Training**

One CNN was based on Xception implemented in Keras with an extra layer with 16384 features followed by a Hadamard classifier (to avoid huge FC weights and push all expression power to the CNN). 

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    image (InputLayer)           (None, 256, 256, 3)       0
    _________________________________________________________________
    axception (Model)            (None, 16384)             42938920
    _________________________________________________________________
    logits (HadamardClassifier)  [(None, 14951), (None, 14 14952
    _________________________________________________________________
    predictions (Activation)     (None, 14951)             0
    =================================================================
    Total params: 42,953,872
    Trainable params: 42,870,672
    Non-trainable params: 83,200

The training regime was as follows:

Trained with all 14591 classes duplicating images from landmarks with a single image. 
Augmentation kicks in after the net has seen an image at least once
Training started at small crops and gradually increased crop size up to 256x256.

This net was trained for 16 epochs until val_acc was `0.923113`. 

Once net was trained, we froze weights and added a distractor detection head, which took the the unscaled 14591 logits of the previous net as inputs as well as a VGG16 net trained on Places1365Hybrid followed by a few FC layers and a sigmoid output:

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    image (InputLayer)              (None, 256, 256, 3)  0
    __________________________________________________________________________________________________
    axception (Model)               (None, 16384)        42938920    image[0][0]
    __________________________________________________________________________________________________
    logits (HadamardClassifier)     [(None, 14951), (Non 14952       axception[1][0]
    __________________________________________________________________________________________________
    vgg16 (Model)                   (None, 512)          14714688    image[0][0]
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 15463)        0           logits[0][1]
                                                                     vgg16[1][0]
    __________________________________________________________________________________________________
    d_fc1024 (Dense)                (None, 1024)         15835136    concatenate_1[0][0]
    __________________________________________________________________________________________________
    bn_m1024 (BatchNormalization)   (None, 1024)         4096        d_fc1024[0][0]
    __________________________________________________________________________________________________
    act_mrelu1024 (Activation)      (None, 1024)         0           bn_m1024[0][0]
    __________________________________________________________________________________________________
    d_fc512 (Dense)                 (None, 512)          524800      act_mrelu1024[0][0]
    __________________________________________________________________________________________________
    bn_m512 (BatchNormalization)    (None, 512)          2048        d_fc512[0][0]
    __________________________________________________________________________________________________
    act_mrelu512 (Activation)       (None, 512)          0           bn_m512[0][0]
    __________________________________________________________________________________________________
    d_fc256 (Dense)                 (None, 256)          131328      act_mrelu512[0][0]
    __________________________________________________________________________________________________
    bn_m256 (BatchNormalization)    (None, 256)          1024        d_fc256[0][0]
    __________________________________________________________________________________________________
    act_mrelu256 (Activation)       (None, 256)          0           bn_m256[0][0]
    __________________________________________________________________________________________________
    d_fc128 (Dense)                 (None, 128)          32896       act_mrelu256[0][0]
    __________________________________________________________________________________________________
    bn_m128 (BatchNormalization)    (None, 128)          512         d_fc128[0][0]
    __________________________________________________________________________________________________
    act_mrelu128 (Activation)       (None, 128)          0           bn_m128[0][0]
    __________________________________________________________________________________________________
    predictions (Activation)        (None, 14951)        0           logits[0][0]
    __________________________________________________________________________________________________
    distractors (Dense)             (None, 1)            129         act_mrelu128[0][0]
    ==================================================================================================
    Total params: 74,200,529
    Trainable params: 16,528,129
    Non-trainable params: 57,672,400
    __________________________________________________________________________________________________

 This was trained on the recognition train data and a) yelp restaurant photo classification, b) open images and c) landmark retrieval index as distractors. It achieved 0.88 validation accuracy on distractor detection.

The code for the above part is all in `train.py` 

The other two CNNs predictions were based on fine-tuned SE-resnet 50, the out of distribution images are all set to a label 15000. The validation accuracy is about 94% while the public leaderboard is only about around 0.7 for probabilities from a single model. Simple additions of the model with different input resolution can achieve about 0.11-0.12(public leaderboard). 

**Ensembling**

We ensembled by voting the predictions for the 3 nets above. Each of them yielded 0.125 LB, 	0.7LB and 0.93LB and the ensemble yielded 0.156 LB.

3. Indoor images detection
We detected and re-scored indoor images in test dataset using the labels of the top5 predicted place categories from VGG16Places365 to vote if the given image is indoor or outdoor.

**NN Search** 

We extracted fc1 layer 4096-dimensional feature vectors from VGG16Places365 and computed cosine distances for all train-test image pairs. Brute force search on GPU using tensorflow took a couple of days. After that we increased the scores of images if CNN predicted label and top1 neighbour’s label agreed. Then we repeated this procedure for the same class average features. Finally we ended up using fast but less accurate faiss for NNs search on 512-dimensional feature vectors.

**Source code:**

Available @ https://github.com/antorsae/landmark-recognition-challenge
