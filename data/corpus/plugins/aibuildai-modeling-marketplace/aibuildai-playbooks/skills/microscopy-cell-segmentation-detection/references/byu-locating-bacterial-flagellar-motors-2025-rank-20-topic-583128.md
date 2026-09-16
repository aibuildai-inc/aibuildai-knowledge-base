# 20th place solution -- Keypoint-Based Dual-Graph Predictor

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #20
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583128

#Acknowledgement
Although I'm actually solo (with @daphne4sg ) in this competition, I want to thank all my teammates @goodcoder, @bigochampion and @cybersimar08 who help me in another LLM competition and share good things about personal life. I learn a lot from you guys. I also want to thank @andrewjdarley for hosting such amazing competition. This project gives a lot of helps for my job interview. I have high willingness to do further development because your goal is very interesting.  

#Short Story About This Solution
This solution originates from the best project of my career, which I completed during my master’s degree a year ago. I collaborated with a director at Micron Technology through an industrial partnership. The project was extremely challenging—we were required to build a high-performance anomaly detector using only 88 HBM scanned images due to data privacy constraints.

Additionally, the labels were a mix of AOI-judged results and human expert annotations. Our objective was to model the preferences of the human expert. Some parts of my BYU solution were adapted from this HBM project, such as keypoint generation and label inference from graph structures.

However, the most innovative aspect of my approach was leveraging few-shot prediction with a vision-language model (VLM) instead of traditional object detectors like YOLO. Essentially, I used prompts such as “this HBM is anomalous” or “HBM is OK” to represent the presence or absence of a target, allowing the model to make an initial classification guess.

My solution calibrates the text-image embedding space using graph-based methods to enable the VLM to perform effectively for our specific task. As a result, I was able to train the model with only about 20 images, and it successfully detected nearly all anomalies identified by human experts in the HBM dataset.

# Solution 

## Initial Idea
The initial idea can refer to the [notebook](https://www.kaggle.com/code/tom99763/first-idea-to-extract-keypoints-byu/notebook) where I made it public earlier. I want to produce a lot of points then refining them by building their relationship with labels. My strategy is sampling around the bacteria since motor existed in the head of it. If I aggregate all the keypoints  and summarize them, the correct semantic might can be captured and get away from noisy labels. However, the biggest issue is how to represent the feature of each point, so I only use patch feature extracted from a simple CNN then building GNN to predict the location of motor. It turns out a really bad model. 

##Further Improvement -- YOLO as Feature Extractor and Keypoint Generator 
It's funny that I almost leaked my solution in [this post](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/573491) two month ago. I had short discussion with @hengck23 about this development. After this discussion, I start to implement it and the idea becomes the pipeline showing below.  So basically I use the C3K2 feature map of YOLO11 because the feature in each spatial position can represent refined object. Setting small confidence threshold (conf=0.05) can filter enough object locations determined from YOLO11. This design boosts the score, improving basic approach a lot.

## Node Feature -- C3K2 + RandomWalkPE
Feature makes very big difference for motor prediction. I've tried almost every types of features like HoG, Daisy, EfficientNet, ResNet and different YOLO versions, all of them do not work well except YOLO11, but the score is still not good enough. After several days, I found `AddRandomWalkPE` on the [PYG document](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.transforms.AddRandomWalkPE.html). I suddenly realize this is the missing ingredient in 3d task. The score instantly boosts after I add it. So the node feature is C3K2 + `RanadomWalkPE` with `walk_length=8`. I think the main reason it improves the score is because it can represent the position of each point, implicitly modeling the position of motor.


## Create More 3D points 
YOLO successfully generates a lot of points by setting low confidence threshold, but it is still sparse in 3d space and has high possibility missing object existence. So my strategy for solving this issue is using the predicted point as center then uniformly sampling other points within a radius. I set `radius` and `num_samples` to control the range and density, respectively. I found using this strategy for training GNN also boosts score. 

## Label Development -- Similarity within Radius
Since annotating a single point as motor can not completely represent the actual semantic (that's why host gives so much radius tolerance in the metric), I label a point as positive by the following conditions:
* Features similarity between current location and ground truth location larger than `thr_sim`.
* The current location is within the 3d radius `thr` of ground truth location.

```python
def feat_labeling(points, feat, extract_feat, tomo_id, train_labels, thr = 10, thr_sim=0.5):
    label = train_labels[train_labels.tomo_id == tomo_id]
    print('location:', label[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values[0])
    n_motors = label['Number of motors'].item()
    if n_motors == 0:
        return np.zeros(points.shape[0], )
    d, h, w = label[['Array shape (axis 0)', 'Array shape (axis 1)', 'Array shape (axis 2)']].values[0]
    loc = label[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values[0]
    tz, ty, tx = loc.astype('int32')
    fd, fh, fw = feat.shape[0], feat.shape[2], feat.shape[3]
    z = tz.astype('int32')
    y = ((ty/h) * fh).astype('int32')
    x = ((tx/w) * fw).astype('int32')
    #sim
    extract_feat = extract_feat / np.linalg.norm(extract_feat, axis=-1, keepdims=True)
    target_feat = feat[z, :, y, x][None, :]
    target_feat = target_feat / np.linalg.norm(target_feat, axis=-1, keepdims=True)
    sim = extract_feat @ target_feat.T
    bool1 = sim[:, 0]>thr_sim

    #dist
    dist = np.linalg.norm(points - loc[None, :], axis=-1) #(N, 3)
    bool2 = dist<thr

    #label
    label = np.array(bool1 & bool2, dtype='float32')
    return label
```

## Choice of Graphs 
I experimented with various graph structures and ultimately found that combining a k-NN graph with a [Delaunay graph](https://en.wikipedia.org/wiki/Delaunay_triangulation) yields the best results. The k-NN graph provides high density, while the Delaunay graph establishes connections between distant clusters. By leveraging both properties, the combined approach results in a more generalized model. The experiment results below clearly explain why someone only take 16 submission attemps and go to sleep for a while.

|  | Public Score | Private Score | CV |
| --- | --- |
| Knn Graph | **0.835** | **0.835** | 0.968 |
| Radius Graph | 0.826 | 0.830 | 0.966 |
| Delaunay Graph | 0.801 | 0.806 | 0.962 |
| Knn + Radius | 0.841 | 0.835 | 0.966 |
| Knn + Delaunay | **0.856** | **0.843** | 0.966 |




## Pipeline
First training a YOLO on the competition dataset. I don't use external dataset because I think the annotator is not the same person so it may gives differenet judgement for the motor existence. Then training the two GraphSages for each fold with `Binary Cross Entropy` with `positive_weight=8`. The ensemble function is maximum in my selected submission, but HDBSCAN is better in some scenarios. The training data for GNN is collected from the points filtered by YOLO which contains both positive and negative data. No extra dataset being used here.





##YOLO Size 
Unfortunately I'm trapped from my experiments and cannot make the right choice in the final minutes of deadline. YOLO11L predicts a lot of false positives which is good in this competition. But I didin't select it. This means I still have to improve my mindset and experience as a data scientist.

|  | Public Score | Private Score | CV | Sub Time |
| --- | --- |
| YOLO11L | 0.856 |**0.843** | 0.968 | 7HR |
| YOLO11X | **0.865** | 0.831 | **0.983** | 11HR |


## What didn't work
* All the segmentation models 
* Pure keypoints generation 
* Larger model


# Code Release

[yolo11l inference notebook](https://www.kaggle.com/code/tom99763/21th-place-solution?scriptVersionId=240706129) (0.843 pb, best private score) 
[yolo11x interence notebook](https://www.kaggle.com/code/tom99763/21th-place-solution-yolo11x?scriptVersionId=243423147) (0.831 pb, final submission) 
[Github training code](https://github.com/tom99763/21th-place-solution-BYU)

# Experiment Record 
My pipeline is quite complicated, verifying those methods take me a lot of engineering effort. Here is all the experiment records in the past three months: [Google Sheet](https://docs.google.com/spreadsheets/d/1KCo8G_LF3T7FMXkVHWeKxo4vz24zoY7i697syACEaUY/edit?usp=sharing)



If you have any question please comment below or use [email](tom99763@gmail.com) & [linkin](www.linkedin.com/in/lin-chieh-huang-4b2231227) to contact me! 
You're welcome to share my solution on your blog—it's a great honor for me! I hope others will find it helpful and refer to it in future competitions.
