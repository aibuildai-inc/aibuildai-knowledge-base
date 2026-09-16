# 1st Place Solution

Competition: benetech-making-graphs-accessible
Rank: #1
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418786

First of all, I would like to pay tribute to all the participants who worked on this competition.   
I would also like to thank the hosts for organizing this interesting task competition.    
This task was very interesting and I enjoyed working on it because I could think of many different approaches.    
I am honored to have won first place in this very interesting competition.

# Overview
[Overview]

My solution consists of a two-step pipeline that first classifies chart types using a classification model and then performs data series inference.  
In the inference phase of the data series, Bar, Line, and Dot were end-to-end predictions by Deplot trained for each chart type, while Scatter was predicted by an object detection-based approach.  

The final scores are as follows.  
| | Overall | Scatter | Dot | Line | Vertical Bar | Horizontal Bar |
| ---     | ---  | ---  | ---  | ---  |  --- | ---  |
| public  | 0.86 | 0.10 | 0.00 | 0.32 | 0.39 | 0.05 |
| private | 0.72 | 0.30 | 0.01 | 0.13 | 0.26 | 0.01 | 

# Dataset
I used the following three data sets.  
1. Competition data set（comp_extracted_dataset/comp_generated_dataset）  
  * Using both extracted dataset (comp_extracted_dataset) and generated dataset (comp_generated_dataset).  
  * For the generated data, data with noise in the annotations were removed by a simple check (about 100 images).  

2. ICDAR data set（ICDAR_dataset/ICDAR_manualannot_dataset）  
  * I used 1406 pieces of data for which annotations were provided (ICDAR_dataset) and 1903 pieces of data for which no annotations were provided (ICDAR_manualannot_dataset).  
  * For data for which annotations were provided, I visually rechecked the annotation contents and manually corrected all data that did not follow the competition's annotation rules (e.g., % notation) or contained noise.  
  * For the data for which no annotations were provided, I first visually checked the appearance of all the data and selected the data that could be used in this competition. Next, I inferred and assigned pseudo-labels using the Deplot model, visually checked all the results again, and manually corrected all incorrect annotations.  

3. Generated synthetic data set（synthetic_dataset）  
  * After reviewing the images in the competition dataset and determining that the comp_generated_dataset alone did not have a sufficient variation to achieve robustness, so I generated about 65k synthetic data myself.  
  * I mainly generated synthetic data with features that comp_generated_dataset does not have.  
      * histogram  
      * Label contains line breaks  
      * Bar chart with error bars  
      * Line chart containing x labels not included in the data series
      * etc...  
  * To add diversity, I also included 10k images from the dataset published by @brendanartley.  

  (Example of synthetic data I generated)  
[generated_data]

# Solution pipeline  
## 1. Chart classification  
* There is not much to note, as I just performed a simple classification task.  
* I used two models, convnext_large_384 and swin_large_patch4_window12_384, and integrated their results in a weighted ensemble.  

**Training configs**  
* Dataset (training data and validation data were randomly divided into folds)  
    * Training（78k images）：comp_extracted_dataset, comp_generated_dataset, ICDAR_dataset, synthetic_dataset(horizontal_bar/dot)  
    * Validation（4k images）：comp_extracted_dataset, ICDAR_dataset, synthetic_dataset(horizontal_bar/dot)
* Hyperparameters
    * Epochs：15  
    * Batch size：16  
    * Adafactor (lr=3e-5)  

## 2. Data series prediction  
Based on the classification results of Step 1, different approaches are used for scatter and the rest.  
Except for scatter, the predictions are made in Deplot, and scatter is an object detection-based prediction.  

### 2-a. Bars & Lines & Dots  
* I experimented by switching the model to Deplot based on the Donut-based notebook that @nbroad had published. However, I could not train well just by changing the model, so I made some modifications.  
    * Ground Truth Format  
      Removed chart type from GroundTruth and adapted the format to that of the Deplot original.  
      `<0x0A> x_value1 | y_value1 <0x0A> x_value2 | y_value2 <0x0A> x_value3 | y_value3 </s>`
    * x-axis and y-axis swap of horizontal_bar  
      Since the concept of x-axis and y-axis was reversed in the annotation rules of this competition and the original format of Deplot, I trained according to the original concept of Deplot and swapped the values during inference.  

* Training was conducted in multiple stages. In the first stage, training was conducted using data from all chart types (All Chart-type Train), and using the results of that training as initial weights, one or two additional training sessions (Specific Chart-type Train) were conducted using only specific chart-type images to generate a model specialized for a chart type.
    * The scores for vertical_bar and line were slightly improved by this method. (Two Specific Chart-type Train runs were performed for vertical_bar and one for line.)  
    * Horizontal_bar had a worse cv after the second stage of training, probably due to the small amount of extracted data, so I decided to use the All Chart-type Train model to make predictions.    
    * Dot chart only has generated data and I decided that I could not validate it successfully, so I decided not to train after the second stage and use the All Chart-type Train model to make predictions.  
[Deplot_training]

**Training configs**  
1. All Chart-type Train  
    * Dataset  
        * Train（120k images）：comp_extracted_dataset, comp_generated_dataset, ICDAR_dataset, 
    synthetic_dataset  
        * Validation（2.5k images）：comp_extracted_dataset, ICDAR_dataset, ICDAR_manualannot_dataset
    * Hyperparameters
        * Epochs：8
        * Batch size：2  
        * Adafactor (lr=1e-5)
        * cosine scheduler with warmup (warmup_step=4000)
        * Augmentation: GaussianBlur, GaussNoise, some color augmentations 

2. Specific Chart-type Train (vertical_bar)   
    * Dataset  
      [1st time]  
        * Train（6k images）  ：comp_extracted_dataset, ICDAR_dataset, synthetic_dataset  
        * Validation（1.3k images）：comp_extracted_dataset, ICDAR_dataset, ICDAR_manualannot_dataset  

      [2nd time]  
        * Train（1500 images）：comp_extracted_dataset, ICDAR_dataset, ICDAR_manualannot_dataset   
        * Validation（500 images） ：comp_extracted_dataset, ICDAR_dataset, ICDAR_manualannot_dataset

    * Hyperparameters
        * cosine scheduler with warmup (warmup_step=0)
        * Otherwise, same as All Chart-type Train  

3. Specific Chart-type Train (line)  
    * Dataset  
      [1st time]  
        * Train（1150 images）：comp_extracted_dataset, ICDAR_dataset, ICDAR_manualannot_dataset  
        * Validation（400 images）：comp_extracted_dataset, ICDAR_dataset, ICDAR_manualannot_dataset  

    * Hyper paramete
        * cosine scheduler with warmup (warmup_step=0)
        * Otherwise, same as All Chart-type Train  

**Score**  
* All Chart-type Train
| | Overall | Scatter | Dot | Line | Vertical Bar | Horizontal Bar |
| ---     | ---  | ---  | ---  | ---  |  --- | ---  |
| public  | 0.78 | 0.06 | 0.00 | 0.29 | 0.38 | 0.05 |
| private | 0.53 | 0.13 | 0.01 | 0.12 | 0.26 | 0.01 | 


* Specific Chart-type Train
| | Overall | Scatter | Dot | Line | Vertical Bar | Horizontal Bar |
| ---     | ---  | ---  | ---  | ---  |  --- | ---  |
| public  | 0.81 | 0.06 | 0.00 | 0.32 | 0.39 | 0.05 |
| private | 0.55 | 0.13 | 0.01 | 0.13 | 0.26 | 0.01 | 


### 2-b. Scatter
* Only scatter was not accurate enough with Deplot no matter how I tried, so I adopted an object detection-based approach.  
* Since scatter is guaranteed to have label values of numeric type, it is easy to calculate values from ratios if the "label position," "label text," and "scatter point position" are known. I used CACHED to detect "label position," Deplot to read "label text," and YOLOX to detect "scatter point".  
* The mapping between label bboxes and label texts is a bit crude, but the smallest label text is mapped to the left-most (top in the case of y-labels) label bbox, and the largest label text is mapped to the right-most (bottom in the case of y-labels) label bbox. It is possible that the total number of detected label bboxes does not match the total number of read label texts, but we did not take any special measures because such a pattern was very rare when we checked the verification data.  
* The above object detection approach resulted in a scatter score of **Pulic:0.10/Private:0.30**, much better than the Deplot-based score (Pulic:0.06/Private:0.13).

**More Details**  
Scatter point detection  
* comp_extracted_dataset, ICDAR_dataset, synthetic_dataset were used for training data (about 12.5k images)
* Somewhat semi-automated using provided annotation data and image processing, but inevitably noisy, so corrected with manual annotation.  
* To increase detection accuracy, only the plot area was cropped for training and inference.
* The model was YOLOX-l, input image size was 1280 for both training and inference, and 50 epochs were trained.

Label text reading
* comp_extracted_dataset, ICDAR_dataset, comp_generated_dataset were used for training data (about 62k images, all chart types were used, not just scatter)  
* Use the same training configs as in (2-a) All_Chart-type_Train for the training parameters, and change the format of Ground Truth as follows  
  `<0x0A> x_label1 | x_label2 | x_label3 <0x0A> y_label1 | y_label2 </s>` 
* At first, I thought of using EasyOCR or similar to achieve this, but there were many issues to be addressed, such as support for character rotation, and it seemed that tuning would be very difficult to increase versatility, so I adopted a method of batch reading with Deplot.  

---
Finally, I would like to thank once again everyone involved in this competition.  
Thank you very much for taking the time to read this!

[edit] 
* I have published 1st place [inference notebook](https://www.kaggle.com/code/kashiwaba/benetech-1st-place-inference)(23/6/28) 
* I have published 1st place [training code](https://github.com/KASSII/benetech_1st_place_train)(23/7/8)
