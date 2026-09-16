# 2nd Place Solution

Competition: landmark-retrieval-2021
Rank: #2
Source: https://www.kaggle.com/c/landmark-retrieval-2021/discussion/277273

Code: https://github.com/WesleyZhang1991/Google_Landmark_Retrieval_2021_2nd_Place_Solution
Paper:  
https://arxiv.org/abs/2110.04294       OR
https://github.com/WesleyZhang1991/Google_Landmark_Retrieval_2021_2nd_Place_Solution/blob/master/ILR2021_2nd_solution.pdf. 
Submission Notebook: https://www.kaggle.com/zhangwesley/ilr2021-retrieval-2nd-solution

# Introduction
Image retrieval is a very important computer vision task which aims at finding images similar to the query image. It is different from instance-level retrieval. Image retrieval aims to retrieve objects holding the same appearance with the query, even they are not the same instance. This makes the task more easier compared to instance-level retrieval. 
On the other hand, it is different from the fine-grained level image retrieval. The fine-grained level image retrieval pays more attention on the local attentions to discover more details due to its small intra-category variance, such as person re-identification.
Landmark retrieval is an instance-level retrieval task, which aims to search the same landmark from a large candidate set. In this paper, we will introduce our techniques used in the fourth landmark retrieval competition, Google Landmark Retrieval 2021 held on Kaggle. Some of them are inspired from the state of the art algorithms in person re-identification.

Besides, we also involve many techniques that are commonly used in previous competitions including model structures training strategies, loss functions. These techniques have been well explored and introduced in previous competitions, so we only introduce our methods, denoted as new contributions listed below.
- We involve bags-of-tricks from person re-identification and conduct careful experiments on these tricks.
- We propose a continent-aware sampler to balance the distribution of training images based on their continent tags.
- We design a Landmark-Country aware reranking algorithm and integrate it with the K-reciprocal reranking method.

# Method and experiments
## Train and validation set
The official GLDv2 dataset has provided a clean and a full version. As pointed out by previous works, many noisy images with the same IDs as clean set have been filtered out during the cleaning stage. Expanding these noisy data into clean set results in ‘c2x‘. Although dis-similar, these noisy images may contain very valuable information, e.g., the in- door or outdoor of a building. What’s more, we also include the index set from GLDv2, which shares many common ids with trainfull. We list the dataset below for a better understanding.

| Trainset  | \# Samples | \# Labels |
| :-------: | :--------: | :-------: |
|   Clean   | 1,580,470  |  81,313   |
|    C2x    | 3,223,078  |  81,313   |
| Trainfull | 4,132,914  |  203,094  |
|    All    | 4,825,830  |  203,094  |

We use the 1129 GLDv2 test set together with the 76,176 index set from the competition. We put all GT images of each query into the index set and expand it to 78,959 im- ages. In this case, any query could find all its GTs in the index set.


## Baseline network
We select several large CNN networks including SE-ResNet-101, ResNeXt-101, ResNeSt101 and ResNeSt269 as backbones. IBN extension is used for SE-ResNet and ResNeXt-101. The input size is selected as 384 for pretraining and 512 for the last fine-tuning. The last stride of the CNN network is set to 1.
We use generalized mean-pooling (GeM) for pooling method with p=3.0. Arcface loss with scale=30 and margin=0.3 is used. We use weight decay=0.0005. Training details with gradually enlarging input size and data scale can be found in implementation details.

Some tricks from person re-identification has been explored.  Since the dataset is very large, we use R50 backbone with an in- put size of 256 × 256. We list the validation accuracy as well as public/private scores. Random Erasing randomly erases out image patches and has shown great success in many fields. Label smoothing by using soft targets that are a weighted average of the hard targets can often be useful in many computer vision tasks. From the table below, ransom erasing proves to be effective while label smoothing fails.

|    Setting    | Validation | Public | Private |
| :-----------: | :--------: | :----: | :-----: |
|   baseline    |   32.60    | 28.44  |  30.59  |
|      +RE      |   32.78    | 29.29  |  30.60  |
| +label smooth |   32.55    | 28.21  |  29.79  |


##  Sampling Strategy
On one hand, in person re-identification or face recognition, id-uniform is widely used as a data sampling strategy. For a batch, we randomly select P Ids and then K images for each Id. Thus we have P*K images as a batch. Each id is treated fair for this setting. On the other hand, softmax sampling has been widely used in previous competitions. The softmax sampling just shuffle all dataset once at the beginning of the epoch and then samples iterative through the data. Head data which appears more will be put more attention with this setting.
We have tried these sampling strategies and find neither of them are good enough for our task.

As the provided landmark dataset pays more concentration on Asia landmarks, we manage to design a sampling strategy based on their continent labels. 
First, we use the country-and-continent-codes-list to find how many countries each continent has. Then we search and list all the landmarks in each country. Based on this processing, we can get the country tag and continent tag for every landmark.

We setup a continent sampling prob by {'Asia': 0.5, 'Europe': 0.2, 'Africa': 0.15, 'North America': 0.1, 'South America': 0.02, 'Antarctica': 0.01, 'Oceania': 0.01, 'OTHER': 0.01}. For an epoch of images, we sample continent images by the corresponding ratios. Also, we learn from previous paper which set 0.66 probability for clean data and 0.33 probability for noisy data.


We list results for different sampling strategy below. The widely used id-uniform strategy fails for landmark retrieval. We think the reason is due to the large amount of tail data. These tail data may be noisy and thus degrades the model training. The softmax strategy works better than id- uniform. For continent-aware strategy, it achieves the best performance.

|     Setting     | Validation | Public | Private |
| :-------------: | :--------: | :----: | :-----: |
|   Id-uniform    |   31.05    | 24.78  |  27.28  |
|     Softmax     |   32.60    | 28.44  |  30.59  |
| Continent-aware |   33.07    | 31.37  |  32.44  |



##  Reranking
Reranking is very essential to the final performance. Besides K-reciprocal reranking, a Landmark-Country aware reranking algorithm is specially designed for our task.

We observe that there are many images which contain the same landmark and can't be easily retrieved by visual features due to great variations caused by views and illumination. Considering this, a Landmark-Country aware reranking is proposed by taking fully use of the training set. Specifically, as each image in training set has its landmark tag and its country tag, we first assign query and index images in the testing set with a training tag, and then retrieval the query from index images by the assigned tags. 

For query image tagging, We give a list of potential landmark tags and country tags to every query image according to its top K similar images in training set, and each landmark tag and country tag is scored by the accumulation of similarity in top k . With the Landmark-Country aware reranking, images have the same landmark tag or country tag with the potential tag of query are advanced in the retrieval sorting. An illustration can be found below and more details and equations are listed in paper.

[rerank]

##  Implementation details
Motivated by previous solutions, we first train the models on ‘clean‘ subset with an input size of 384(For ResNeSt269, the size is 448). The initial learning rate is 0.01 and we train for 10 epochs with the first epoch as warmup. Then we keep the same category number but with more data as ‘c2x‘. The initial learning rate is 0.001 and we train for 6 epochs. Then we expand dataset to ‘trainfull‘ and train for another 2 epochs with the initial learning rate as 0.0001 and input size as 512. At last, we use ‘all‘ data from GLDv2 and train for another 2 epochs with the initial learning rate as 0.0001 and input size as 512.



# Conclusion
In this paper, we import techniques and bag-of-tricks from person re-identification. For the specific task of landmark retrieval, we propose continent-aware sampling strategy and Landmark-Country aware post processing, which has proven to be very effective on the private leaderboard. The relationship between landmark retrieval and landmark recognition would be studied in the future.
