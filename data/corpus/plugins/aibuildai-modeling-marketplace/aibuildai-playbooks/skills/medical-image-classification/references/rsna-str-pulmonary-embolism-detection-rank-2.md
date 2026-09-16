# [2nd place] Solution Overview & Code

Competition: rsna-str-pulmonary-embolism-detection
Rank: #2
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193401

Update: code available at https://github.com/i-pan/kaggle-rsna-pe

Congratulations to all the participants and the winners. Special congrats to prize winners @osciiart and @jpbremer who are on track to be physician GMs. This was a tough, compute-heavy challenge given the large amount of data and short timeframe. My setup was 4 24 GB Quadro RTX 6000 GPUs. I always feel guilty during competitions like these since I have the luxury of strong compute. Models were trained using DDP in PyTorch 1.6 with automatic mixed precision. 

Even though the results aren't final yet and my submission may be removed for violating the label consistency requirements (hopefully my heuristic for fixing those worked!), I still wanted to share my solution.



Here is a schematic outlining my solution. It has a lot of moving parts, so I apologize for the lengthy summary. 

# Step 1: Feature Extraction

Last year's RSNA Intracranial Hemorrhage Detection challenge shared a lot of similarities with this year's challenge. Most of the top solutions combined 2D CNN feature extraction with sequence modeling. The backbone of my solution also relied on a similar setup. 

I first trained ResNeSt50 on 2D images. Images were windowed using the PE-specific window (WL=100, WW=700) that I mentioned in one of my initial posts on DICOM processing. Each "image" was 3 channels, with each channel representing an individual slice. Thus the 2D image was a stack of 3 continuous slices. The targets were the 7 PE-related labels (i.e., excluding the RV/LV ratio labels). Note: most of the PE-related labels were exam-level labels. However, I just assigned  the exam labels to each slice positive for PE (negative slices had all zeros), with the understanding that there would be label noise. Also, I predicted the labels for the middle slice among the 3 slices in the image. Models were trained using vanilla binary cross-entropy loss (`BCEWithLogitsLoss` in PyTorch), 512x512 with 448x448 random crops (single center crop during inference), RandAugment data augmentation, batch size 128, 5 epochs, 2500 steps per epoch, RAdam optimizer with cosine annealing learning rate scheduler. I used generalized mean pooling and reduced the final feature vector to 512-D. Mean loss was around 0.06 (AUC 0.95-0.96 for slice-wise prediction of PE vs. no PE). Features were then extracted for all slices. 

# Step 2: Sequence Modeling

Many of last year's solutions used LSTMs/GRUs as the sequence model of choice. For this competition, I used `huggingface` transformers (specifically, the `Transformer` class from `transformers.modeling_distilbert`). I used one 4-layer transformer to produce slice-wise `pe_present_on_image` predictions and another transformer to predict exam-wise PE-labels. Sequence length was 512 during training, padded/truncated as necessary. During inference, I used the sequence without modifications. 

Important point: **Images from non-PE exams do not contribute to the loss.** At first, I was training the slice-wise transformer on all exams. Then, I decided to train these models on positive exams only. This lowered my CV by about 0.01-0.02. I used a custom weighted loss where I weighted the loss from each example by the proportion of positive PE slices (as described in the metric), though I'm still not sure I wrote it correctly. 

The exam-level transformer was trained using a weighted BCE loss based on the competition label weights. Exam-level validation losses ranged from 0.15-0.17.  

# Step 3: Time-Distributed CNN

To add some variety into my modeling, I then trained a time-distributed CNN, which is just another way of saying I stacked a transformer on top of a CNN feature extraction backbone and trained end-to-end. 

But before doing that, I performed inference using the slice-wise transformer model to get PE scores for every slice (5-fold OOF predictions). Then, when training the TD-CNN, I only trained on the top 30% of slices from each exam, sorted by PE score. These were trained on 3D volumes of size 32x416x416 cropped to 32x364x364 using the same windowing strategy (WL=100, WW=700) in batches of 16. 

I initialized the CNN backbone and the transformer head with trained models from steps 1 and 2 to help with convergence. I forced all the batch normalization layers in the backbone to `eval` mode as well- this prevents the running mean and variance in each layer from updating and only trains the coefficients. Exam-level validation losses ranged from 0.15-0.17, similar to step 2. 

# Step 4: Heart Slice Prediction

RV/LV ratio is a significant portion of the loss. I hand-labeled slices with heart in 1,000 CT scans and trained a model (EfficientNet-B1 pruned, AUC 0.998, 256x256->224x224 crops) to classify heart slices in each CT scan. I did this because I felt that by focusing a model on the heart, I could get better, more consistent results across scans. It actually wasn't hard to label 1,000 scans- probably a full day's worth of work. I just needed to find the top and bottom heart slices; everything in between thus must also contain the heart.

# Step 5: RV/LV 3D CNN

I trained a 3D CNN to classify RV/LV ratio. Specifically, I used a 101-layer channel separated network, pretrained on 65 million Instagram videos (https://arxiv.org/abs/1904.02811, https://github.com/facebookresearch/VMZ). 

This model was trained only on heart slices from each exam. Also, it was only trained on positive exams. This is because RV/LV ratio was not labeled for negative exams- both RV/LV labels are 0. Thus, it didn't make sense to me to try and train my model on the entire dataset's labels directly. Models were trained using a weighted BCE loss using the competition weights. I resized the input to 64x256x256->64x224x224 crops and used mediastinal window (WL=50, WW=350). AUC for RV/LV ratio > 1 was about 0.85. The validation loss ranged from 0.44-0.48. I then used these models to extract 2048-D features from each exam.

The challenge here was that the likelihood of an exam being PE-positive greatly influenced the RV/LV labels. From step 2, I calculated 5-fold OOF exam-level predictions for all exams. Then, I trained a linear model that took as input the concatenation of the 2048-D 3D CNN feature and the 7 PE exam labels. This linear model was trained across **all exams** using the labels directly. This way, the model could take into account the imaging features from the scan but also adjust the predictions based on the likelihood of PE. Validation losses ranged from 0.22-0.25. 

I didn't validate my entire pipeline that often due to time constraints, instead choosing to focus on the individual component losses and optimizing each one as well as I could. I did periodically do a sanity-check on 200 single-fold exams to make sure that my entire pipeline was working, and my final validation loss was 0.183 for my 0.150 public LB submission. I'm still not confident that I implemented the metric correctly, but I was seeing good correlation (0.159/0.195->0.156/0.188-> 0.150/0.183).

At the end, I applied a function to enforce label consistency requirements for each exam. There was about 0.001 change in CV after applying this function. I have some thoughts about this requirement which maybe I'll save for another post. 

# Final Models

Overall, I used these models:
-2x ResNeSt50 feature extractors
-6x exam transformers (3 for each extractor)
-6x slice transformers
-5x ResNeSt50 TD-CNN
-1x EfficientNet-B1 pruned heart slice classifier
-5x ip-CSN-101 3D CNN RV/LV feature extractor
-5x RV/LV linear model 

Inference took about 7-8 hours for the entire test set. 

Code to follow once I clean it up. 

Things that didn't work as well:
-3D CNN for PE exam-level prediction
-Pseudolabeling RV/LV ratio for negative exams
-Using logits/probabilities instead of features for second-stage model
-TD-CNN for RV/LV ratio prediction
-Training feature extractor on positive exams only to improve slice-wise PE prediction
