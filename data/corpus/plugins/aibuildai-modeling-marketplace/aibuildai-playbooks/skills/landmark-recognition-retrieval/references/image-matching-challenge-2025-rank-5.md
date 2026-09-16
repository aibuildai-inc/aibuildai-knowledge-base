# 5th Place Solution : MASt3R is All You Need

Competition: image-matching-challenge-2025
Rank: #5
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583711

First and foremost, I'd like to thank the organizers and Kaggle team for hosting such an exciting and challenging competition. Congratulations to all the top participants! While I’m not new to the field of 3D vision, this was my first time properly participating in the Image Matching Challenge. One of the main challenges I faced was joining the competition quite late - around two weeks before the deadline. As a result, a significant portion of my time went into setting up Kaggle notebooks, leaving me with limited opportunities to iterate and experiment with different approaches and their variants. That said, let's dive into the simple yet effective solution that worked for me.

#### Overview

From recent personal experiments, I’ve observed that foundation models like DUSt3R, MASt3R, and VGGT offer superior matching performance compared to traditional detector-descriptor-matcher pipelines (e.g., ALIKED or SuperPoint + LightGlue). Based on this, I built a straightforward pipeline using MASt3R and evaluated it on the IMC-2025 dataset. As expected, it performed well in terms of matching accuracy. However, a major challenge was the high computational cost of the MASt3R matcher, which led to frequent notebook timeouts during inference. I addressed this by implementing an efficient image-pair shortlisting strategy, tuning its hyperparameters, and applying a few engineering optimizations to reduce runtime without significantly impacting performance.

The pipeline that gave the best result on Public LB :-
1. Image-Pair Similarity based Shortlisting using [MASt3R-ASMK](https://arxiv.org/abs/2409.19152)
2. [MASt3R](https://arxiv.org/abs/2406.09756) Semi-Dense Matching on the shortlisted image pairs
3. COLMAP based Verification, Reconstruction and Clustering

[PipelIne Overview Block Diagram]


#### Solution Details

##### 1. Image-Pair Similarity based Shortlisting 

<p> I tried 3 methods to compute a similarity matrix between all the images of a dataset in a fast way and shortlist the pairs using certain thresholds, for the more expensive MASt3R Matcher to run on them. </p>

(i) [**DINO-v2**](https://arxiv.org/abs/2304.07193) : Similarity matrix was computed by normalizing the (1 - distance_matrix), where L2-distance was used.
(ii) [**MASt3R-ASMK**](https://arxiv.org/abs/2409.19152) : Used the official implementation to compute the similarity matrix.
(iii) [**XFeat local-feature-aggregation**](https://arxiv.org/abs/2404.19174) : XFeat is a local keypoint feature extractor which is very fast but comparable to Superpoint, etc in terms of accuracy. Created a custom function to compute the "mean cosine-similarity" of top-k nearest-neighbor matching keypoint features between 2 images, as the image similarity.
<p>
The similarity matrix from each method was passed to a function with the hyper-parameters topk_min, topk_max, topk_percentile and sim_thres were used to shortlist the candidate set of pairs for image matching.
</p>
- **sim_thres** : Hard threshold on similarity score 
- **topk_percentile** : Having a fixed top-k threshold restricts adapting the no. of similar images when the dataset size varies. Defining it as a percentile makes it adaptive.
- **topk_min** and **topk_max** : The minimum and maximum no. of image pairs allowed for matching, these thresholds limit the pairs in case top-k percentile pairs are too low or high in number.

[table-1]

<p>As you can observe from the above table that MASt3R-ASMK performed better than the others, so it was chosen. The hyper-parameters topk_min, topk_max, topk_percentile and sim_thres were tuned separately for each method using the local validation dataset (IMC-2025-train). I submitted some of the hyper-parameters combinations to the public LB to verify the best choice. </p>

[table-2]

<p> Though MASt3R Matcher has high precision and is able to discard false positives by predicting very less matches, it still helps to pass only relevant image-pairs. Firstly, because the matching and reconstruction needs to be completed within the 9h time budget. Secondly, more pairs doesn't necessarily mean better accuracy (refer to Table-2 row-4). </p>

<p> Another thing that I wanted to try is the ensemble of the image-shortlisting methods but due to lack of time and attempts, couldn't test it. </p>

##### 2. Image Matching

I used the MASt3R model's feature extraction `(image_size = 512)` and semi-dense matching using Fast-Reciprocal-NN from the official implementation `(subsample = 8, pixel_tol = 5)` . I tried to tune the min_conf_thres (Minimum Confidence Threshold) parameter for the matches a little bit, keeping all other params fixed. `(MASt3R-ASMK: topk_min - 10, topk_max - 30, topk_percentile - 0.3, sim_thres - 0.001)` .

[table-3]

##### 3. COLMAP based Verification, Reconstruction, and Clustering

<p> Once the matches are estimated using MASt3R, they are exported to COLMAP format and final image pair indexes are dumped. Then pyCOLMAP API's geometric verification is used to verify the matches. </p>

<p> After that, pyCOLMAP's Incremental Mapping pipeline is used to generate the list of reconstructions or models. Due to MASt3R's superior matching precision, COLMAP is able to easily cluster images into separate sparse models. The following parameter values were used for the mapping. </p>

```
mapper_options = pycolmap.IncrementalPipelineOptions()
mapper_options.min_model_size = 3
mapper_options.max_num_models = 25
```

<p> I have also tried pre-clustering the images using either MASt3R-ASMK or DINO-v2, followed by MASt3R matching in each cluster and then generating single reconstruction model by either COLMAP (Incremental-SFM) or GLOMAP (Global-SFM). But this approach performed worser than the automatic clustering using COLMAP, on local validation set, so discarded. </p>

##### 4. Engineering Tips and Tricks

- Parallel Processing of Individual Datasets on the 2 x T4 GPUs (weighted distribution based on dataset size).
- Build cuRoPE module of CroCo (MASt3R) using CUDA.
- Isolate modules into subprocesses which can randomly crash (like for e.g. pyCOLMAP) and retry execution for max_retries
