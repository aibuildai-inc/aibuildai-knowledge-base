# 4th place solution (brief summary)

Competition: image-matching-challenge-2022
Rank: #4
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328798

Thanks for organizing the interesting competition! All members of our team are in civil engineering and have almost no relevant knowledge in image matching before the competition, so it is a challenging and funny experience for us to start from scratch. Competition host provided valuable learning material (problem definition, correspondences filtering and learned matching paper), which allowed us beginners to get started quickly. It is pleasantly surprised for us to get the fourth place. 😄

# Summary

Our method is entirely based on pretrained models, and no fine-tuning of these models is made in our method.  Without enough computing resources, therefore, we focused on the ensemble methodology of existing methods at the very beginning.

We refer to the following notebooks in our work: (Sincere thanks to the authors of the papers and notebooks!)

•	LoFTR: https://www.kaggle.com/code/cbeaud/imc-2022-kornia-score-0-725/notebook
•	QuadTree: https://www.kaggle.com/code/dschettler8845/quadtree-image-matching-challenge-2022
•	DKM: https://www.kaggle.com/code/radac98/public-baseline-dkm-0-667
•	SuperGlue: https://www.kaggle.com/code/losveria/superglue-baseline/notebook

We combined all the matched key points obtained in different methods and used MAGSAC to calculate the fundamental matrix. The process is roughly as follows:

1. **Extracte matched key points of multiscale images**
Resize image pairs to different scales and apply matching method.
2. **Rescale the key points**
Rescale the coordinate of key points to its position in original image
3. **Combine all key points**
4. **Calculate the fundamental matrix**

| | resize strategy | 
| ----- | ----- |
| loftr | resize the length of long side to 1000, 1200, 1400 + adjust the height-width ratio |
| superglue | resize the length of long side to 1200, 1600, 2000, 2800 + adjust the height-width ratio |
| dkm | resize image pixel to 346800 + maintain the height-width ratio |

Since superglue has strict license and is not allowed for prize-winning, but permitted to be used in the competition to get better rank, we took two routes after breaking into the top five rank:

1. ensemble of loftr, superglue and dkm
best private score: 0.853 (not selected as final submission)
https://www.kaggle.com/gufanmingmie/imc-2022-final-ensemble

2. ensemble of loftr, quadtree and dkm (exclude superglue)
best private score: 0.843

best private score selected as final submission: 0.852, ensemble of loftr, quadtree, superglue and dkm
