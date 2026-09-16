# 12nd Place Solution: SP+SG

Competition: image-matching-challenge-2023
Rank: #12
Source: https://www.kaggle.com/c/image-matching-challenge-2023/discussion/420471

Thanks to the competition organizers and Kaggle staff for hosting this amazing competition and solid support. Thanks to each my team member for their creative perspective and effort which gain us this place in competition.

1. Overview
 Our final model is rather simple. The solution is based on a modular toolbox named Hierarchical-Localization (implementation of typical SP+SG structure), and the key modification is to set the input resolution to 2000.
 As the resolution increased from 1024 to 2000, the public LB scores increased from 0.24 to 0.46. As the resolution further increased to 3000, there is still improvement in the trainset (especially "wall").
2. Configuration in detail 

    ```
    confs = {
        'superpoint_aachen': {
            'output': 'feats-superpoint-n4096-r1024',
            'model': {
                'name': 'superpoint',
                'nms_radius': 3,
                'max_keypoints': 4096,
            },
            'preprocessing': {
                'grayscale': True,
                'resize_max': 2000, # Scaling longside to 2000
            },
        }
    }
    ```

    ```
    'superglue': {
        'name': 'superglue',
        'weights': 'outdoor',
        'sinkhorn_iterations': 50,
    }
    ```


3. Tricks didn't work
   Our code suffers from randomness; the difference in public LB score for the same code can be up to 0.04! Unfortunately, during the whole competition, we failed to find a way to eliminate randomness. We decide a modification useless if no obvious improvement observed during repetitive submissions, but still, there are possibly mistakes.
   - Image retrieval
     After incremental mapping , try to add the images mapped failed to the successfully reconstructed model.
   - TTA
     Reverse and concatenate and apply NMS to features extracted from [original left-right-flip 10-deg-rotation], however, computation time hugely increased while no obvious improvement observed.
   - Ensemble SIFT to SP
     We learned that SP fails when the image is extremely in-plane rotated. So we tried to perform SIFT extraction and matching using pycolmap and extract the match from the database and concatenate with SPSG. This trick can improve the score for "cyprus" but is useless in this year's test set. Worth noting is that it doesn't harm the LB score either.
   - Multi-models and multi-resolution
     We tried to ensemble SPSG with DKM, Loftr, Quadtree, silk, but these models perform poorly despite being in the same resolution with SPSG. Some of them are one-stage models, which made them inefficient for the task. We tried to concatenate [800, 1500, 2000] resolutions calculated by SPSG; the result is nearly the same with 2000 alone.
   - Suppress randomness by more iterations
     Hloc uses geometry verification (`pycolmap.verify_matches`), set `max_num_trials=40000` didn't work.
4. Some experience from the competition
   Reading images using cv2 can lose EXIF information, which is not conducive to reconstruction. But there's a counter-example, "theater" can score even higher after the information removed.
5. Referrence
 - Hloc
  https://github.com/cvg/Hierarchical-Localization/
 - Superpoint introduction
  https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/assets/DL4VSLAM_talk.pdf
