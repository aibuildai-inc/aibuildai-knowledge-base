# 8th-place-solution

Competition: vesuvius-challenge-surface-detection
Rank: #8
Source: https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/8th-place-solution

Thanks to the organizers for hosting this technically challenging and topology-sensitive competition, and congratulations to all prize-winning teams.

## One-line summary:
Multi-scale nnUNetResEncUNetL ensemble + mean fusion + probability diffusion + direction-aware adaptive hysteresis.

## Models.
The backbone is nnUNetResEncUNetL, without architectural modification. Three models were trained with different patch sizes, focusing on MedialSurfaceRecall: patch 224 (500 epochs), patch 256 (500 epochs), and patch 288 (500 epochs). Different patch sizes provide complementary receptive fields, balancing local surface precision and global continuity.

## Inference & fusion.
Inference followed the standard nnUNet sliding-window pipeline (step_size = 0.9). Patch 224 and 256 used mirroring TTA, while patch 288 did not due to inference speed constraints. The final prediction was obtained by direct mean fusion of the three probability maps, without logit weighting, which showed more stable leaderboard behavior.

## Post-processing (key part).
Most performance gains came from post-processing. First, gradient anisotropic diffusion was applied twice to the fused probability map (conductance = 2.0, time_step = 0.04) to smooth fragmented responses while preserving sharp surface transitions in probability space.
Second, adaptive hysteresis thresholding with directional morphology was applied. Parameters were T_low = 0.50, T_high = 0.90, z_radius = 1, xy_radius = 0, and min_size = 100. Voxels above T_high define high-confidence surface cores, while voxels between T_low and T_high are included only if connected to the core. Morphological connectivity is restricted to the z direction to reflect the thin layered structure of the scroll surface and prevent lateral over-expansion, followed by removal of small isolated components.

A lighter diffusion variant (single iteration, conductance = 1.5, time_step = 0.03) slightly improved private performance (up to 0.618) but reduced the public score (0.595 → 0.593). Compared to using adaptive hysteresis alone, diffusion improved local validation by ~0.005, although its public score was slightly worse. We therefore selected the stronger diffusion setting, trusting the local evaluation and prioritizing overall stability.

Overall, the solution avoids complex architectural tricks and relies on scale diversity, stable mean ensembling, probability-space smoothing, and direction-aware adaptive thresholding.

Due to the Chinese New Year holiday, I was not able to devote much time to the competition during the final two weeks; given that, I am very satisfied with the final ranking. Thanks again to the organizers for this unique and inspiring challenge.

The inference notebook is [here](https://www.kaggle.com/code/lingyundev/notebook686f6dd4fe).
Code is [here](https://github.com/LingYunGit/SurfaceDetection_8th_Solution).
