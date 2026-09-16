# 2nd place solution summary

Competition: landmark-retrieval-2020
Rank: #2
Source: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/177078

**[update]**
add google driver download link: 
https://drive.google.com/file/d/1XnzxMOHhzua9tjrAjo-X55ieKVJzRJw_/view?usp=sharing
**[update]**
submission to arxiv is on hold. We provide detailed information in pdf file, download link: https://vis-bj.bj.bcebos.com/landmark/landmark2020_retrieval.pdf

**Method:**
Our retrieval method for this competition is depicted in Figure 1. We mainly train two models for final submission and each model includes a backbone model for feature extraction and a head model for classification. ResNeSt2692 and Res2Net200 vd are selected as the backbone model since their good performance on ImageNet. Head model includes a pooling layer and two fully connected(fc) layers. The first fc layer is often called embedding layer or whitening layer whose output size is 512. While the output size of second fc layer is corresponding to the class number of training dataset. Instead of using softmax loss for training, we train these models with arcmargin loss. Arcmargin loss is firstly employed in face recognition, we found it works well in retrieval tasks which can produce distinguishing and compact descriptor in landmark.


The training process mainly consists of three steps. Firstly, we train the two models with resolution 224x224 on GLDv1 dataset which has total 1215498 images of 14950 classes, and GLDv2-clean dataset which has total 1580470 images of 81313 classes. Secondly, these two models are further trained on GLDv2-clean dataset with resolution 448x448, the parameters of arcmargin loss may change during the process. We believe that using large input size is beneficial to extract feature of tiny landmark. However, we have to adopt the training strategy “from small to large” mainly due to the large cost and lack of GPUs. In the final step, some tricks are experimented to increase the performance. We have tried a lot of methods, such as triplet loss finetuning, circle loss finetuning and etc but only “GemPool” and “Fix” strategy are helpful.

**Training and test details**
At the data level, we ﬁrst used GLDv1 and GLDv2-clean data with small resolution to train at a large learning rate, and then used GLDv2-clean data with large resolution to train at a small learning rate. The speciﬁc details are listed in Table 1. Table 2 shows the results of training with the above strategies. Table 3 lists the mAP@100 score of model ensemble.




