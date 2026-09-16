# 23rd Place - Fast GPU Experimentation Pipeline!

Competition: rsna-str-pulmonary-embolism-detection
Rank: #23
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193402

Thank you Radiological Society of North America (RSNA®), Society of Thoracic Radiology (STR), and Kaggle for hosting this fun competition. Thank you Nvidia for providing compute resources.

This has been one of my favorite competitions. I enjoyed building an elaborate multi-stage pipeline of stacked models! Working with 3D images was fun and provided an additional challenge compared with 2D images. I particularly enjoyed tackling the challenge of building a fast experimentation pipeline when the training data is `1_000_000_000_000 bytes` of data! One trillion bytes! This is the largest dataset I have ever worked with.

# RSNA STR Pulmonary Embolism Detection
In the figure below, each row is an exam (i.e. study, i.e. single patient). The row of images are CT scan "slices" from the 3D image of a patient's chest. In this competition, we need to predict 9 targets for each patient (each row) (like is pe on left side? on right side? etc) and we need to classify every image (is pe present?). If below were all the data, then we would need to predict `3 rows * 9 targets = 27 exam targets` and `15 images * 1 target = 15 image targets`. In total we would need to predict 42 targets.



# Understanding the Metric
On the description page, the metric seems very confusing. However it is just a weighted average of 10 log losses (the 9 types of exam predictions and the 1 type of image prediction). (I explain the metric [here][2]). After computing the 10 weights, we find that 50% of our LB score is from the 9 exam predictions and 50% of our LB score is from the image predictions. Furthermore, the log loss for the image predictions is itself a weighted log loss where an image that is part of an exam without pulmonary embolism has weight zero (very important observation!) 

* Improving image level predictions is equally important as improving exam predictions.
* There is no penalty for false positives, so we can train our image prediction models with only the 30% of the data from positive exams!

# Stage 1 - Model One - Image Level Predictions
# (CNN EfficientNet B4)

    inp = tf.keras.Input(shape=(320, 320, 1)) # INPUT IS UINT8
    x = tf.keras.layers.Concatenate()([inp/255., inp/255., inp/255.])
    base_model = efn.EfficientNetB4(weights='imagenet', include_top=False) 

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)    
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(lr=0.000005)
    model.compile(loss='binary_crossentropy', optimizer = opt)

    model.fit(X, y, sample_weight = X.groupby('StudyInstanceUID') 
        .pe_present_on_image.transform('mean') * 5.6222 )

I built two models. Model one predicts image level predictions (i.e. `pe_present_on_image`) and model two predicts patient level predictions (i.e. exams i.e. studies, like `leftsided_pe` etc)
* EfficientNet B4 pretrained on `imagenet`
* **Only Mediastinal window, (ie. level=40, width=400)** i.e. 1 channel `uint8`
* Random crops of 320x320 from 512x512
* Rotation (+-8 deg) Scale (+-0.16) augmentation
* Coarse Dropout (16 holes sized 50x50)
* Mixup (swap slices of similar Z position with other exams)
* Adam optimizer with constant `LR = 5e-6`
* Training sample weight equal to pe proportion in exam
* **Only train on 30% of train data with pe present in exam**
* 40 minute epochs using 4x V100 GPU
* Train 15 epochs with batch size 128

Below illustrates my augmentations. For display purposes, we illustrate Mixup with a large yellow, green, or blue square so you can see it better. (During runtime, it was an actual second image).

  

# Stage 1 - Model Two - Patient Level Predictions
# (CNN EfficientNet B4)
Each patient has an average of 200 images. Among those 200, if PE is present, it is usually on the middle slices. Therefore I only train my patient level model with slices `0.35 < z < 0.65`. Then to predict the 9 targets for each patient, I only infer `0.35 < z < 0.65` and then take the 9 average predictions.

* Most details same as model one
* **Only Mediastinal window, (ie. level=40, width=400)** i.e. 1 channel `uint8`
* Output layer of 9 sigmoid units
* **Only train on 30% of train data with Z Position between `0.35 < z < 0.65`**
* Loss `weighted_log_loss`



# Experimentation Pipeline
How do we discover the details above? All the settings above were discovered by performing dozens of experiments on **smaller images and smaller backbones**. For example, use 128x128 (with 80x80 crops) EfficientNetB0 and/or 256x256 (with 160x160 crops) EfficientNetB2. Using these smaller models, we can test out ideas on a single GPU in minutes! Also note that we only use 1 channel images of `uint8`. This is 33% less data than converting images to 3 channels of 3 different CT window schemes.



Remember we are only training with 30% original data. Then using crops makes it 12% of data. Then using 256x256 reduces this to 3% of data. And using 128x128 reduces this to 0.75% of data! Even Kaggle notebooks P100 GPU can train quickly on 80x80 crops from 128x128 and 30% train data.

Once you find a configuration that works well, then run 512x512 with EfficientNetB4 overnight. Using only 2D predictions for image and 2D predictions for patient, **the above two models obtain LB 0.215 and CV 0.235.** 

We will now increase our CV LB by building stage 2 models that use stage 1 predictions as input
  
# Stage 2 - Model One - Image Level Predictions
# (Random Forest)

    FEATURES = ['oof']
    for k in NEIGHBORS:
        tmp = train.sort_values('PosZ').groupby('StudyInstanceUID')[['oof']]
        train['b%i'%k] = tmp.shift(k)
        train['a%i'%k] = tmp.shift(-k)
        FEATURES += ['a%i'%k, 'b%i'%k]
    train.fillna(-1,inplace=True)

    model = RandomForestClassifier(max_depth=9, n_estimators=100, 
                               n_jobs=20, min_samples_leaf=50)
    model.fit(train.loc[idxT,FEATURES],train.loc[idxT,'pe_present_on_image'],
                    sample_weight = 5.6222 * valid.loc[idxT,'weight'])

All images are slices from 3D images. So adjacent images (within the same exam) contain helpful information. Each plot below displays all 200 or so image level predictions from 1 study. The x axis is z position and the y axis is the prediction value (0 to 1). The blue line is the ground truth, the orange line is the prediction from the model described above. The black line is the random forest Stage 2 model.

For each image level prediction, a random forest model takes as input the prediction and neighbor predictions [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250] on either side. Then the random forest model predicts a new image level prediction display in black below. Notice when the original prediction is close to 1, then the random forest pushes it up to 1 and when the original prediction is close to 0, then the random forest pushes it down to 0.

**This stage 2 image level model increased LB to 0.204 from 0.215 and CV to 0.224 from 0.235 (gain = 0.011)**



# Stage 2 - Model Two - Patient Level Predictions
# (GRU + 1D-CNN)

    inp = L.Input(shape=(64, 1792))
    x = L.Bidirectional(L.GRU(48, return_sequences=True, 
                        kernel_initializer='orthogonal'))(inp)
    x = L.Bidirectional(L.GRU(48, return_sequences=False, 
                        kernel_initializer='orthogonal'))(x)
    x = L.Dense(9, activation='sigmoid')(x)
        
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(lr=0.00005)
    model.compile(loss=weighted_log_loss, optimizer = opt)

Similarly we can use adjacent slice information to improve our patient level predictions. Most patients have between 160 and 310 images per study.



Below are plots of the top view (Z position is vertical axis) of the 3D image (not the ordinary slice view of the 3D image). We notice that most of the crucial information is between 25% and 75% in top view. Therefore we extracted 64 images equally spaces between 25% and 75% Z position. Then we took those 64 images and extracted the GAP embeddings from both our stage 1 model one and stage 1 model two. We trained a stage 2 GRU model and a stage 2 1D-CNN model to predict exam level predictions from these 64 GAP embeddings.



**This stage 2 exam level model increased LB to 0.183 from 0.204 and CV to 0.203 from 0.224 (gain = 0.021)**

# Other Misc Ideas
When using global average pooling 2D in your CNN, location information is lost. Therefore I tried giving my models location information in various ways to help predict targets related to location (i.e. `leftsided_pe` etc) and related to size (i.e. `rv_lv_ratio_gte_1` etc). Unfortunately, none of my ideas increased CV or LB. My favorite is below.

## Locating PE with Class Activation Maps CAM

I extracted class activation maps from my stage 1 models and feed the location information into stage 2 exam models. (CAMs explained [here][1]). In the below figure, the ground truth is in the title. The green circles are the CAM of my EfficientNetB4 model. Note that CT scans are flipped so the left side of the image is "right" and the right side of the image is "left. You can see that the CAM does a good job of locating the PE.



# Thank you

[1]: https://www.kaggle.com/cdeotte/unsupervised-masks-cv-0-60
[2]: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193598
