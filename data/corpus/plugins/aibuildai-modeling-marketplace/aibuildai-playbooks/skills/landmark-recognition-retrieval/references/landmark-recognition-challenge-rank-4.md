# Our solution [4th place]

Competition: landmark-recognition-challenge
Rank: #4
Source: https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57896

**Step 1: Training a usual CNN**

We just trained a lot of different models: ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt101, InceptionResNetV2, DenseNet201.

Some of them were training with 15k classes. Random 224x224 crops, resizes, scales, shifts, rotates, flips were used for augmentation.

Pavel Pleskov firstly used 128x128 crops and then increased crop size up to 256x256. CNN's were training using fast.ai until accuracy on validation does not become 0.975. He did not use classes with less than 10 samples (the number of remaining classes was eight thousand). 

However, it is not enough to train a lot of CNN's and merge their predictions in this competition. There are a lot of challenges:

* Classes with one or two samples in the train dataset
* Images from Google Landmark Retrieval Challenge in the test dataset
* Non-landmark images in the test dataset

**Step 2: Recognizing landmarks from retrieval challenge**

We used a separate CNN (based on ResNet50) classifier to separate images with landmarks from this competition and from retrieval challenge. A simple heuristic improved our score: if is_from_another_competition_probability &gt; 0.95: confidence *= 0.1;

**Step 3: Recognizing images with no landmark**

Another separate binary classifier (based on ResNet50) trained on Open Images &amp; train data. The heuristic was the same as in step 2.

**Step 4: Few-shot learning**

We found a simple technique to improve the score of any of our neural networks:

 1. Extract features from hidden layer
 2. For each image from test set find K closest images from train set (K=5)
 3. For each class_id we computed: scores[class_id] = sum(cos(query_image, index) for index in K_closest_images)
 4. For each class_id we normalized its score: scores[class_id] /= min(K, number of samples in train dataset with class=class_id)
 5. label = argmax(scores), confidence = scores[label]

**Step 5: kNN with features from local crops**

We extracted 100 augmented local crops from each image. Crops with no landmarks were removed (using CNN from step 3). Then, we used the technique described in step 4 with K=20 independent for each local crop from query image (this is an image from test dataset). After that procedure, for each image from test set we got a set of local crops (up to 100) and a list of pairs (label, confidence) for each crop. For each class_id we computed its score as a sum of 5 - class rank for each crop. Class rank means the position of this class_id in the sort order of pairs (label, confidence). If at least one class rank was 5 or more, then we reset the entire sum. Thus, based on this technique we were able to choose approximately one thousand examples with high confidence in our prediction.

**Merging:**

One of the biggest challenges in this competition was how to merge predictions from different models and how to set confidence score correctly. We used several heuristics:

 1. Compute confidence score for each label using predictions from steps 1-4 as follow: score[label] = label_count / models_count + sum(label_confidence for each model) / models_count. Here label_count is a number of models where the prediction with max confidence is equal to the label.
 2. We also used each prediction from step 5 with confidence = 1 + confidence_from_step_5 / 100

**Slight leakage:**

The test set has a slight leak: many of the images have a name starting with "2016-" or "2017-". For example 2017-02-23.jpg. There are no such names in the train dataset, so our assumption was that these images contain no landmark. We tested our hypothesis but the score became worse. Then, we found that a simple heuristic could slightly improve our score: if confidence &lt; 1 and name is leaky: confidence *= 0.1
