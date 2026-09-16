# 4th place solution

Competition: landmark-recognition-2021
Rank: #4
Source: https://www.kaggle.com/c/landmark-recognition-2021/discussion/277386

Thanks to my longtime teammate @haqishen and new teammate @hongweizhang . It was again great teamwork. We got 3rd place in retrieval and 4th place in recognition competition. I'm posting the solutions in both forums for easy future reference, even though there is large overlap.

## Approach
Qishen and I were part of the 3rd place team (together with @garybios @alexanderliao ) in recognition 2020. Last year we developed a strong model, i.e., sub-center ArcFace with dynamic margins. 

This year two competitions run simultaneously with tighter deadlines, so we largely re-used last year's pipeline, with newer architectures and better post-processing. We trained models the same way for both retrieval and recognition competitions, but selected different model combinations and different post-processing for final submissions.

## Training recipe
We mostly follow our last year's solution (see [here](https://www.kaggle.com/c/landmark-recognition-2020/discussion/187757) for details). The recipes are
- sub-center ArcFace with dynamic margins
- progressive training with increasing image sizes
- the indispensable [timm](https://github.com/rwightman/pytorch-image-models) library
- cosine learning schedule with Adam/AdamW optimizer
- multi-GPU training with DistributedDataParallel

## Model choices
There are quite a few new SOTA image models published in the past year, notably EfficientNet v2, NFNet, ViT, Swin transformers. We trained all these new models using last year's recipe. For transformers, we used 384 image size. For CNN models, we used progressively larger image sizes (256 -> 512 -> 640/768). 

We found that the CNNs and transformers perform equally well on local GAP scores (recognition metric), but transformers have significantly higher local mAP (retrieval metric) even with smaller 384 image size. We think this is because transformers are patch-based and can capture more local features and their interactions, which are more important for retrieval tasks.

Besides the new models, we also reused some of our last year's models by finetuning, mostly EfficientNets.

## The ensemble
Our final ensemble consists of 11 models: 3 transformers, 3 new CNNs and 5 old CNNs. Their specifications and local scores are below. Old CNNs' CV scores are not shown because they are leaky (last year's fold splits were different).

Our best retrieval ensemble has 4 fewer CNNs and only 7 models in total (3 transformers + 4 CNNs), because the addition CNNs hurt the retrieval score. This is because transformers are more important for retrieval, and adding more CNNs would reduce transformers' weights.

|       Model      | Image size | Total epochs | Finetune 2020 | cv GAP (recognition) | cv mAP@100 (retrieval) |
|:----------------:|:----------:|:------------:|:-------------:|:--------------------:|:----------------------:|
|     Swin base    |     384    |      60      |               |        0.7049        |         0.4442         |
|    Swin large    |     384    |      60      |               |        0.6775        |         0.5161         |
|     ViT large    |     384    |      50      |               |        0.6589        |         0.5633         |
|   ECA NFNet L2   |     640    |      38      |               |        0.7053        |         0.3600         |
| EfficientNet v2m |     640    |      45      |               |        0.7136        |         0.3811         |
| EfficientNet v2l |     512    |      30      |               |        0.7146        |         0.4117         |
|  EfficientNet B4 |     768    |      10      |       ✓       |                      |                        |
|  EfficientNet B5 |     768    |      20      |       ✓       |                      |                        |
|  EfficientNet B6 |     512    |      20      |       ✓       |                      |                        |
|  EfficientNet B7 |     672    |      20      |       ✓       |                      |                        |
|    ReXNet 2.0    |     768    |      10      |       ✓       |                      |                        |

## Post-processing 
Post-processing was shown to be very effective in last year's top solutions. We incorporated ideas of top3 solutions
1. Combine cosine similarity and classification probabilities (our [3rd place solution](https://www.kaggle.com/c/landmark-recognition-2020/discussion/187757))
2. Penalize similarity with non-landmark images ([1st place solution](https://www.kaggle.com/c/landmark-recognition-2020/discussion/187821))
3. Set similarity=0 if top5 similarities with non-landmarks > 0.5 ([2nd place solution](https://www.kaggle.com/c/landmark-recognition-2020/discussion/188299))
4. Penalize CosSim(query, index) for index class’s count in train set (ours, new)
