# 10th Place solution (brief summary)

Competition: image-matching-challenge-2022
Rank: #10
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328903

First of all, I'd like to thank Kaggle staff and the organizers for  organizing this very practical and challenging competition. 
Our team is ranked 10th on the private leader board. 

## **Overview**

[[IMC2022-10th.png]](https://postimg.cc/64yWk0Kn)

The solution can be summarized as below.

## **Key Points**

- Ensemble with DKM-ROI
- Make Faster and Better Estimation
- Outlier Handling
- Trust CV

## Ensemble with DKM-ROI

DKM showed a robust tendency in our experiments. It can find many correspondence points almost all the time, but peak performance was a shortage as compared with another algorithm. 

From this tendency, we regarded DKM as an “Average Hitter”. And used it for making cropped images as ROI in the following steps.

1. Inference with DKM and calculate confidence threshold using Otsu's method
2. Filtering correspondence points using calculated threshold
3. Making a bounding box with filtered correspondence points  
4. Cropping image from bounding box region
5. Ensemble with LoFTR, SE2-LoFTR, and SuperGlue

A sample result of ROI extraction by DKM-ROI is shown below

[[Untitled.png]](https://postimg.cc/HjZz0yhy)

Figure.1 confidence map of points confirmed by DKM

[[Untitled.png]](https://postimg.cc/Whc3H9CL)

Figure.2 Binarized result of the confidence map by Otsu’s method

[[Untitled-1.png]](https://postimg.cc/62ds2h0n)

Figure.3 bbox of the corresponding point on the original image

Basically Ensemble with DKM-ROI worked well. It makes finding key points easier. But sometimes it missed important regions. As a simple countermeasure for this issue, our solution does not only ensemble with DKM-ROI ensemble and also inference with the original image using LoFTR and ensemble with it. 

## Make Faster and Better Estimation

As with usual competition, one of the important points of this competition is increasing inference time somehow or other. Typical ideas of our solutions are below:

- Pipelining:
    -  We optimized the overall process as a multithreading implementation. It can be described as three components. 1. Preprocess(e.g. load image), 2. Inference(e.g. matcher), 3 . Postprocess(e.g. RANSAC) . Our solution ran multithread to process those parallels. Especially for the tasks in this competition, pipelining is very effective, because GPU processing and CPU processing are highly loaded with different processes.
- Key points NMS: 
    - Detected key points are usually noisy and redundant. It makes time to calculate. To avoid this issue, we used NMS algorithms for key points.  The name SSC (Suppression via Square Covering). SSC could reduce processing time while maintaining accuracy.
- Weighted Random sampling:
    - For the same reason, we tried to reduce the number of key points fed to RANSAC in a way that would maintain estimation accuracy. As an approach, we applied the weighted random sampling algorithm used in DKM to the key point ensemble results.

## Outlier Handling

The input images in this competition are expected to be very diverse. 

In a difficult case, such as with no texture and far distance and very different viewpoints, the effect of a simple ensemble was limited because few correspondence points could be found. Therefore, the following measures were taken for these pairs that should be called outliers.

- Outlier case 1: When only a few dozen percent of the usual number of correspondence points are found.
    - Boosting key points by incorporating methods that differ in tendency from those used in the previous stage of the ensemble. We used SGMNet as an additional ensemble.
- Outlier case 2: When only a few percent of the usual number of correspondence points are found.
    - Judging the scene to be extremely difficult to find sufficient correspondence points with the usual approach, search for correspondence points with DKM alone, not with an ensemble.

These conditions were derived by analyzing the tendency in the output of the ensemble from image pairs of validation data.

## Trust CV

To my best knowledge, some candidates for this competition said “this competition is not Trust CV”. But actually validating with CV was a fundamental process for our team with original validation settings.
I posted about this one as a separated thread.

https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328999


## Plot results for test images

We also agree with this post same as the 11th place solution.
https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328715
https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328887

The results are here.
[[image-7.png]](https://postimg.cc/9RgQNbKM)
[[image-8.png]](https://postimg.cc/H8514tCF)
[[image-9.png]](https://postimg.cc/tY3HsDkx)

---

## Acknowledgment

We deeply acknowledge great OSS such as PyTorch, Kornia, OpenCV, etc.
Special thanks go to authors of these great papers and pretrained models that we used in our final submission:

・DKM

[https://github.com/Parskatt/DKM](https://github.com/Parskatt/DKM)

Edstedt, J., Wadenbäck, M. and Felsberg, M., 2022. Deep Kernelized Dense Geometric Matching. arXiv preprint arXiv:2202.00667.

・LoFTR

[https://github.com/zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)

Sun, Jiaming, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. "LoFTR: Detector-free local feature matching with transformers." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8922-8931. 2021.

・SE2-LoFTR

[https://github.com/georg-bn/se2-loftr](https://github.com/georg-bn/se2-loftr)

Bokman, Georg and Kahl, Fredrik “A case for using rotation invariant features in state of the art feature matchers” ,CVPRW, 2022

・SuperGlue

[https://github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)

Sarlin, Paul-Edouard, et al. "Superglue: Learning feature matching with graph neural networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

・SGMNet

[https://github.com/vdvchen/SGMNet](https://github.com/vdvchen/SGMNet)

Hongkai Chen, Zixin Luo, Jiahui Zhang, Lei Zhou, Xuyang, Bai, Zeyu Hu, Chiew-Lan Tai, and Long, Quan. Learning to match features with seeded graph matching network. In *CVPR*, pages 6301–6310, 2021. 1, 2, 6

・Otsu’s method

[https://cw.fel.cvut.cz/b201/_media/courses/a6m33bio/otsu.pdf](https://cw.fel.cvut.cz/b201/_media/courses/a6m33bio/otsu.pdf)

Nobuyuki Otsu. A threshold selection method from gray-level histograms. *IEEE
transactions on systems, man, and cybernetics*, 9(1):62–66, 1979.

・SSC

[https://github.com/BAILOOL/ANMS-Codes](https://github.com/BAILOOL/ANMS-Codes)

Oleksandr Bailo, Francois Rameau, Kyungdon Joo, Jin- sun Park, Oleksandr Bogdan, and In So Kweon. Efficient adaptive non-maximal suppression algorithms for homo- geneous spatial keypoint distribution. Pattern Recognit. Lett., 106:53–60, April 2018.
