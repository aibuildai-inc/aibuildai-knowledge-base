# 4th Place Solution

Competition: iwildcam2021-fgvc8
Rank: #4
Source: https://www.kaggle.com/c/iwildcam2021-fgvc8/discussion/243509

Thank you and congratulations to all the participants, Kaggle, and especially the organizers of this enjoyable and important competition!  Thank you for making a difference!

## Summary
My solution was based on a two-stage model, one image-based, the second, detection-based, where the output of first model was fed as metadata into the second model along with detection cropped images. The primary motivation for the design was the mixed species herds in eastern Africa. The two-stage pipeline was trained separately for each of the six regions.  A single model architecture, an ImageNet, pre-trained EfficientNet B3, was used in both stages with images cropped/resized to 400x400. Minor image augmentation was used in both stages. The system was implemented in PyTorch Lightning.

## Data Preparation

To identify the unknown locations and assign them to regions, I used a two-fold approach. In the first stage I clustered locations using associated species provided in the image annotations. In the second stage I used a supervised classification model (ResNet-18) where the labels were the known regions, run on the raw images, downsampled to 224x224. 

In both full image and detection-level processing, images were converted into numpy 3x400x400 uint8 arrays - a compromise between speed and storage. All other metadata was stored in a relational database largely mirroring the iNaturalist annotation structure derived from the json files. 

Google Colab was used to run MegaDetector V4 on all images and the V4 detections were used exclusively.

During initial EDA in the eastern African region I noticed that there were multiple species per sequence and that images could be labeled according to a minority member (e.g. a herd of impala (#96) with one zebra (#111) could be labeled zebra). There might have been a predictable labeling strategy employed at the sequence level but I did not spend the time to find it. Instead I emphasized those images where there were single detections per image, thus converting weak labels into strong. This worked well for many species but I think it had limitations for species that were rarely seen alone (e.g. domestic sheep (#72)). 

Three strategies were used to deal with imbalanced classes. The first was to randomly under-sample the most common classes. A weighted random sampler was used to oversample the rarer classes. The third method used in some cases was to incorporate species data from other regions. For example, in the southern-most South American region, there were minimal samples of mountain lions (#6), jaguars (#24) and collared pecarry (#8) in the training data, but these species were abundant in the regions further north.  This was only done at the detection level where the amount of background imagery presented to the model was minimal. 

## Model

My solution was based on a two-stage cascading model, one image-based, the second detection-based (see Figure). Six independent models were trained, one for each region; training and test data were partitioned into regions using locations.   I used a single model architecture based on an ImageNet, pre-trained EfficientNet B3 model for both stages.  The first stage model worked on resized camera trap images and associated image metadata (num_frames_per_sequence, frame_number, number_detections, time-of-day, day-of-year). The images were run through the feature extraction layers of EfficientNet, then combined with the metadata and fed into a small fully-connected classifier. The entire category prediction vector from stage 1, along with the original image metadata (plus zoom), was fed as metadata into the stage 2 classifier. Additionally, the imagery for each detection was cut from the full image, resized and fed into the feature-extraction portion of EfficientNet.  The final output of the 2-stage model was a category prediction vector for each detection. 

[Figure 1: Model]

I used a fine-tuning approach in model training. EfficientNet weights were kept frozen initially while the classifiers were trained, then the pre-trained network was fine-tuned. The final solution used a MultiLR scheduler to lower the learning rate for the fine-tuning phase of training. The loss function was CrossEntropyLoss in both models.  Early stopping was used based on plateauing of validation loss.

Data was stratified by location within each region and run with five-fold cross validation. The only ensembling performed were the 5 folds.  Minor augmentation was used in both stages (RandomHorizontalFlip, RandomErasing, and RandomGrayscale). A final step combined predictions from each region, applying a per-species threshold to the reported detections.

## Things that worked - but got lost along the way...

I performed a number of experiments early on that had very promising results but for one reason or another did not make it into the final solution. 

Training data was pseudo-labeled at the detection level. Single-detection images were used to train an initial model, then predict on the multi-detection images. High-confidence detections were added to the training data and the process was repeated until a fixed point was reached. The results of this were quite promising and predictable in many cases (e.g. many of the herd animals in Africa). In other cases (e.g. white-lipped peccary (2), domestic sheep (#72)), it did not work as well - possibly because there were two few singleton examples, or when they did congregate the detections were too densely packed. 

Another area that showed promise but was never taken beyond the experimental phase was to preprocess the detection data, eliminating static detections within the same location.  There were numerous cases of fixed tree limbs being classified as animal horns or tails but given their appearance in many sequences, they could have been safely removed, lowering false alarms.

I had partial success grouping the initially unknown locations into regions using a pertained ResNet-18 autoencoder followed by K-Means clustering.  I used Landsat 8 bands 4 and 5 to compute NDVI images that were then duplicated to provide the 3 channels required by the ResNet model. The main difficulty seemed to be the similarity of tropical locations and the presence of clouds. It is also possible that ImageNet-trained models are less suitable for satellite imagery. I stopped work on this approach once the GPS locations were released.

I performed some initial experiments with obtaining additional training data by moving up the taxonomic tree and this also looked promising, both as a way to get additional training data and as part of a broader organizational approach.
