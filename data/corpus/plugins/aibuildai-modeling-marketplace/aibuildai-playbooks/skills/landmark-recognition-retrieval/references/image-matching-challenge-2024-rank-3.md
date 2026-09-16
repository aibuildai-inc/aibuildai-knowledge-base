# 3rd Place Solution: VGGSfM

Competition: image-matching-challenge-2024
Rank: #3
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510338

I am excited to share my participation in IMC 2024 and extend my gratitude to the organizers @oldufo @eduardtrulls , sponsors, and the Kaggle team! This was my first experience at both IMC and Kaggle, and it was truly enriching.

My implementation is based on the public code of [IMC2024 Starter](https://www.kaggle.com/code/nartaa/imc2024-starter/notebook) by @nartaa. It helped a lot! This baseline uses ALIKED+LightGlue and pycolmap for reconstruction. Like others, I used the training set as the evaluation set, with 50 samples.



Below is my solution for IMC 2024.

# 1. Introduction
The core of my solution is [VGGSfM: Visual Geometry Grounded Deep Structure From Motion](https://vggsfm.github.io/) (https://vggsfm.github.io/). This work has been accepted to CVPR 2024 as a Highlight.

Basically, given N input frames, VGGSfM first uses a camera predictor to estimate preliminary camera extrinsic and intrinsic parameters. After selecting a query frame and some 2D query points, it estimates 2D tracks using a track predictor. These tracks, alongside the preliminary camera data, allow us to perform triangulation to obtain 3D points, followed by a bundle adjustment to refine the camera and point combination. More details can be found in the paper. 





# 2. Solution

## 2.1 VGGSfM across All Frames

I began by applying VGGSfM across all input frames, which proved effective on the IMC2024 evaluation set. With the input frame number of 50, its mAA is higher than the baseline by 4% (0.04). However, due to the 16 GB GPU limitation on the Kaggle server, accommodating this approach in Kaggle submission was challenging. OOM is bitter. Therefore, to accommodate the server constraints, I integrated VGGSfM into the existing pycolmap pipeline, as detailed below.

## 2.2 Additional Tracks

The first idea is using VGGSfM track predictor to estimate tracks, and feed them into pycolmap as additional 2D matches. For each image, I identified its N nearest frames using NetVLAD or DINO V2. Using the image as the query frame, I selected query points via Superpoint and found corresponding tracks on the N nearest frames. With N=5, this helped improving mAA by 3% on the evaluation set. However, in the public leaderboard, this only improves the score from ~0.17 to ~0.18. I suspect a higher N could yield better results, but don’t have enough submissions to try. I joined the competition quite late, after half of the competition and with ~120 submissions in total. 

This strategy runs track prediction once for each input frame, avoiding the quadratic complexity issue of exhaustive matching. On the Kaggle cluster, it requires 1.8 seconds per frame, in total 90 seconds for a 50-image scene. On my local A100 cluster, it takes 0.7 seconds per frame.






## 2.3 SfM Track Refinement 

The VGGSfM track predictor adopts a two-stage design, which first predicts coarse tracks and then refine them by local appearance. Inspired by the winner solution by ZJU3DV @xingyihezju3dv last year, I found we can refine the SfM tracks of pycolmap reconstruction. The idea is, first running the baseline with ALIKED+LightGlue (or any other methods), you can find a best reconstruction, the one with most registered images. Each valid 3D point within this reconstruction corresponds to several 2D points, and these 2D points construct a SfM track. Therefore, I extracted the image patches (31x31) for those SfM tracks, refined them by VGGSfM fine track predictor, and updated the pycolmap reconstruction with the updated 2D positions. Then, we run one more global bundle adjustment to optimize the cameras and 3D points based on the refined tracks. For example, in the Church scene, it improves the mean reprojection error from 0.64 to 0.55. 

Importantly, this process does not require recompiling colmap. I have provided a code segment here (https://www.kaggle.com/code/jianyuanv/vggsfm-to-refine-pycolmap-tracks) about how to do this by pycolmap and VGGSfM model. 

During submission, we only refine the top 4096 SfM tracks with highest reprojection errors. This will take around 150 seconds for each scene, i.e., around 30 tracks per second. In the public leaderboard, this improves our solution from ~0.18 to ~0.20.

## 2.4 VGGSfM to relocate missing images 

We also observe that colmap usually cannot register all the images in a scene. Therefore, we try to relocate missing images by VGGSfM. For every missing image, we identified the top 5 registered images it has matches with. If an image cannot find 5 registered images, we use the nearest frames by DINO v2. 

Now we have a subset of 6 (1 unregistered + 5 registered) images in total. We use VGGSfM to reconstruct their camera poses. While, the poses predicted by VGGSfM stay in a different camera coordinate to the original camera coordinate of pycolmap reconstruction. We use the algorithm by umeyama to find an alignment transform for the centers of 5 registered images, and apply this alignment transform on the camera pose of the unregistered image. 

For each missing image, this process will take around 5 seconds. It improves the public leaderboard score from ~0.20 to ~0.21. It is worth noting that sometimes the missing image may be too hard, which may not have enough overlaps (matches) with all other frames. In this situation, bundle adjustment may fail, and we use the camera poses predicted by VGGSfM camera predictor only. To be honest I think I did something wrong for this part because the code was written in a rush. I may mess up with some camera coordinate transform, but it works not bad.

## 2.5 Final Solution

In addition to the methods described above, our final solution also:

- **a.** Handles the rotation using https://github.com/ternaus/check_orientation. If the rotation prediction model predicts a score higher than 0.5 for a frame, we rotate the image by 90, 180, and 270 degrees, and pick the one with a highest number of matches.  
- **b.** Uses the matches from both ALIKED+LightGlue and SP+LightGlue
- **c.** Extract an area of interest for transparent images (by DBSCAN on keypoints), and run keypoint detection again on the area of interest. 

## 2.6 Ideas tried but not worked obviously 
- **a.** Histograms equalization or CLAHE.
- **b.** Disambiguation methods such as the ones included in https://github.com/cvg/sfm-disambiguation-colmap or [Doppelgangers](https://github.com/RuojinCai/doppelgangers) (I probably did something wrong with Doppelgangers, but not enough time to debug it).
- **c.** Homography estimation to filter out outlier matches. 
- **d.** Adjusting pycolmap hyper-parameters.
- **e.** Test Time Augmentation, such as using different image resolution.

# 3. Learned from IMC

The three-week journey was great and highly rewarding, despite some sleep deprivation. One of the good stuffs was discovering several ways to reduce the GPU consumption of VGGSfM, which I will soon update on the VGGSfM GitHub. Additionally, it was gratifying to find that VGGSfM can effectively complement colmap when computational resources are limited. And, being familiar with Kaggle took me a lot of time. I hope to do better in the next year.


(This article was written very quickly and I may further update it next week. Time to have a short break now. But feel free to drop me an email if you have any question about VGGSfM :)
