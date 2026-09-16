# 4-th Place Solution

Competition: vesuvius-challenge-surface-detection
Rank: #4
Source: https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/4-th-place-solution

**Thanks for host for this meaningful and interesting competetion as well as detail answer for metric questions. And congradulations for winners!**

**This competetion just show the powerful generalization of nnUNet again. It's the best segmentation framework in the world (I think) !**

# TLDR:

**Huge nnUNet and heavy postprocessing**

# 1  . **Segmentation Model:**

Model training : 
I did nothing just train a huge nnUNet (Resenc UNet with 7 stages covolution) with sufficient epoch (2000). 

Inference: Uses nnUNetPredictor with a tile_step_size of 0.5 and test-time augmentation (mirroring) to ensure smooth and robust predictions.

Preliminary postprocessing: threshould=0.2 and remove components with size < 1000.

## 2  . **Postprocessing: Two Stage hole Filling and Metric Hacking**

- **1st Stage: PCA Hole Filling (Projection-based)**

This stage targets large, obvious topological loops (Betti-1 errors) by analyzing 2D projections.

Detection: The projection_betti1_finding function projects 3D connected components onto XY, YZ, and ZX planes. It identifies "isolated regions" (background holes) that do not touch the image boundaries.

Refinement:

It extracts the 3D coordinates of these problematic surfaces and applies Principal Component Analysis (PCA) to find the optimal 2D fitting plane.
The points are mapped to 2D, where a binary_fill_holes operation is performed.
The filled results are re-mapped back to 3D space using the inverse_transform of the PCA.
This step intend to fill some large holes.

In fact, there are another more elegant solution for this part: 
component detection -> PCA transformation -> RBF function fitting (torchrbf acceleration) -> interpolation -> PCA inverse transformation -> 3D dialation

However, there are a critical problem in this part:

**How to separate adhering sheets ?**

For those adhering sheets, this method would fail. And till now, I still can not find a good method to solve this problem. Looking forward other teams solution.

- **2nd Stage: PCA Hole Filling (Betti-Matching-based)**

A more granular refinement stage using specialized topological tools.

Detection: Utilizing the betti_matching (C++) library, the bmbarcode_betti1_finding function performs a local persistent homology analysis.

Local Processing: The volume is divided into 20, 20, 20 chunks. If a local block exhibits a Betti-1 error (determined via barcode calculation), the PCA filling logic is applied locally to seal small breaks or gaps in the structure.

If one check the competetion metric, it can be found that the topo score is very sensitive, even if a small hole would contribute to a large drop for topo score.

- **Hacking the Metric**

This specific logic is designed to optimize the final topological score by manipulating the betti-2 count. Although host claim that [practical impact at the top end should be limited](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/678904#3414194), it still exist some bugs (I think) and can be utilized in the follow way:

Automatic Betti-2 Hole Filling: The fill_betti2_holes function identifies background components completely enclosed by the foreground and fills them to eliminate unwanted internal cavities. Set betti-2 = 0

Manual create betti2 number: In cases where specific error thresholds are met (e.g., high betti-0 or betti-1 error, for example: betti-0 > 50 or betti-1 > 5 , can be calculated using the bm.compute_barcode function), then create a small betti-2 error mannually. This step is dangerous to some degree, as it can just calculate  betti-0 or betti-1 numbers of the whole predictions, but in evaluation, there are igore masks which could contribute to totally different betti-0 / 1 numbers.

It locates a voxel that is "fully enclosed" (all 26 neighbors are foreground) within the largest connected component.
It sets that single voxel to 0, effectively creating a manual Betti-2 hole.

Strategy: This "hack" serves to balance the topological metrics if the evaluation criteria favor a specific distribution of Betti errors across different dimensions.

# Some other thoughs but have not try:

1. end-2-end training a rbf function to get 0 holes sheets
2. traing a point cloud seperation model to seperate the adhering sheets

**It must be acknowledged that my solution is not so elegant, but it do works. Looking forword to learn other teams solutions.**

**This writeup maybe lose some details. I will refine it in following days. Feel free to ask any questions.**

Some ablations (Based on my best models)
|  | Public LB | Private LB |
| --- | --- | --- |
| Segmentation Model + Remove dust | 0.590 | 0.609 |
| + Stage1 + Stage2 + Hacking | 0.602 | 0.622 |

submission notebook: https://www.kaggle.com/code/shtljw/vcsd-4th-place-solution
