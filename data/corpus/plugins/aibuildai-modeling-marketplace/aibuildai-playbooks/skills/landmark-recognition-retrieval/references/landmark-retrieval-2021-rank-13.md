# 13th Place - Soft Landmark ID Assignment

Competition: landmark-retrieval-2021
Rank: #13
Source: https://www.kaggle.com/c/landmark-retrieval-2021/discussion/275924

Thanks to google and kaggle for hosting this competition and clairvoyant.ai for supporting this work. It was a lot of fun and a full of learning experience.

Certainly, due to limited hardware, extremely large dataset, and time constraints, my models are not that powerful like top teams, but I would love to discuss post-processing, which uses soft landmark id assignment for index and test images, and gave **+0.075** boost on private LB.

But before, to give a glimpse of models.

## Models
I was using CurricularFace loss along with Adam Optimizer and Gradient Accumulation to train my models. I was using an ensemble of 5 models. All models were trained on the full GLDV2 dataset.

Their scores without any post-processing and using simple KNN on model embeddings are - 
| Model | Image Size | Public LB | Private LB |
| --- | --- |
| Effnet B4 | 512 | 0.322 | 0.342 |
| Effnet B6 | 384 | 0.331 | 0.349 |
| Effnet v2 - s | 512 | 0.325 | 0.351 |
| Effnet v2 - m | 384 | 0.357 | 0.373 |
| Effnet v2 - xl | 256 | 0.318 | 0.319 |
| Concat all embbedings |  | 0.379 | 0.389 |
| Concat all + Post Processing |  | 0.448 | 0.464 |

## Post-Processing
###Soft Landmark ID assignment
**Step 1:** Landmark assignment - Test Images

For the landmark id assignment, I calculated the cosine similarity between the test image embeddings and the embeddings of images in the GLDV2 dataset. Next, I picked the top 'n' most similar GLDV2 images for every image in test data and assigned labels and label similarity scores to that image. An example label and label score calculation for an image A in test data is mentioned below.

For a test image A, pick the top  'n' (in my case, it was 5) most similar images from GLDV2-
|  | GLDV2 Img 1 | GLDV2 Img 2 | GLDV2 Img 3 | GLDV2 Img 4 | GLDV2 Img 5 |
| --- | --- |
| Landmark ID | 121 | 10 | 121 | 10 | 199 |
| Similarity Score | 0.9 | 0.7 | 0.5 | 0.47 | 0.45 |

Then Image Landmark ID score can be calculated as
| Landmark ID | Images | Landmark ID score |
| --- | --- |
| 121 | GLDV2 Img 1, GLDV2 Img 3 | 0.9+0.5 = 1.4 |
| 10 | GLDV2 Img 2, GLDV2 Img 3 | 0.7+0.47 = 1.17 |
| 199 | GLDV2 Img 5 | 0.45 |

Consider landmark id score as zero for landmark ids that are not there in the top 5.

**Step 2:** Landmark assignment - Index Images

A similar procedure was followed for index images, and landmark ids and landmark id scores were assigned to them as well.

**Step 3:** Test-Index similarity score calculation

Finally, calculated similarity between Image A from test and Image B from Index as - 

Similarity (A, B) = Landmark_ID_1_score(A) x Landmark_ID_1_score(B) +
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Landmark_ID_2_score(A) x Landmark_ID_2_score(B) +
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Landmark_ID_3_score(A) x Landmark_ID_3_score(B) + 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...........

Then for every image in test data, arrange all similar images from index data in descending order based on their similarity score.

As the next step, At the end of this list, I concatenated the knn-neighbours not found in the above search.

This post-processing gave about +0.069 boost on public LB and +0.075 on private LB, with 0.448 and 0.464 as final Public and Private scores.

## Additional
Additionally, I am [attaching](https://www.kaggle.com/prateekagnihotri/curricularface-and-gradient-accumulation) my gradient accumulation code for TPUs and a TensorFlow implementation of CurricularFace. I hope you will find them useful as well.
