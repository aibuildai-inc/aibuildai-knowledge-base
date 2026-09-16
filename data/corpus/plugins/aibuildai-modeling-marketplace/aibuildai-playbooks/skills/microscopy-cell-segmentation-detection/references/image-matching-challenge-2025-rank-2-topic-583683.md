# 2nd Place Solution

Competition: image-matching-challenge-2025
Rank: #2
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583683

I would like to thank the Armed Forces of Ukraine, Security Service of Ukraine, Defence Intelligence of Ukraine, and the State Emergency Service of Ukraine for providing safety and security to participate in this great competition
## Solution Overview:
### Feature Extraction (all images resized to longest side is 1280 pixels):
1: Superpoint (using the GIM checkpoint) to extract features and filter pairs with a minimum of 30 matches.
2: Alike + LightGlue applied on pairs obtained from 1.
3: SIFT + Nearest Neighbor matching on pairs from 1.
### Crop Generation and Matching:
4: Generate crops for target object based on the ensemble of matches from 1., 2., and 3.
5: Apply Alike + LightGlue on the crops generated in step 4.
6: Apply Superpoint + LightGlue on the crops from step 4.
### Final Matching and Reconstruction:
7: Concatenate matches from steps 1, 2, 3, 5, and 6; then run RANSAC and filter pairs with a minimum of 150 matches.
8: Run COLMAP with default parameters and generate clusters similarly to the public notebook.
