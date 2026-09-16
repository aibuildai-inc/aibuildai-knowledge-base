# 11th place solution

Competition: image-matching-challenge-2025
Rank: #11
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583097

# Overview


# Notebook
- https://www.kaggle.com/code/motono0223/imc2025-11th-place-solution
    - private lb = 43.34
    - public lb = 43.53

# Descriptions

## Create image pairs
- global feature extractor: DinoV2/Base
- Topk = 150

## Rerank image pairs
* Due to the large number of image pairs, it is necessary to reduce the processing time for image matching.
* To address this, image pairs are re-ranked and filtered using the ALIKED model (local features) in this process.
* Specifically, the procedure is as follows:
    * For each image, local descriptors are extracted using the ALIKED model.
        * The image size is set to 2048 pixels, with 2048 keypoints and a detection threshold of 0.01.
        * All descriptors are normalized with norm=1.
    * A Euclidean distance matrix is computed between the local descriptors of each image pair.
        * If image1 has N descriptors and image2 has M descriptors, an NxM distance matrix is calculated using the GPU.
    * For each row in the distance matrix, the minimum value is extracted (resulting in N minimum values).
    * If these minimum values are below the threshold of 1.0, they are counted as valid matches.
       Image pairs with 15 or fewer valid matches are discarded.

* The key point is that since LightGlue (GNN) is computationally expensive, a fast simulation of image matching is performed using the distance matrix as a lightweight approximation.

## Image Matching
* Local feature extractor: Aliked (n16)
   * resize_to: 1536 pixel
   * max_num_keypoints: 8192 points
   * detection_threshold: 0.01
* Matcher: LighGlue
* post process
   * the number of matches > 15 matches

## Filter image pairs
* This step filters image pairs whose matches are not critical for SfM. The pruning accelerates the reconstruction and gives COLMAP’s incremental mapping more opportunities to reconstruct small clusters.
* Image-matching results can be expressed as a network in which images are nodes and match counts are edges; the number of edges connected to each node can then be counted.
* Non-essential pairs are removed by capping the number of edges per image node (threshold = 6). If a node has more than six edges, those with the fewest matches are discarded until only six remain.

  * In COLMAP’s incremental mapping, an initial pair is chosen and additional images are registered one by one; the initial pair is selected statistically from the matches stored in the database.[1]
  * Match results sometimes concentrate on a handful of images, causing the incremental mapping output to favor certain clusters.
  * Pruning pairs connected to images with many matches increases the likelihood that initial pairs will be formed inside smaller clusters.

## Rough clustering
* A lightweight clustering routine is implemented with NetworkX.
* For every image pair, the number of matches is read from `matches.h5`; edges with 150 or more matches are added to `G = nx.Graph()`.
* Clusters are then obtained via `nx.weakly_connected_components(G)`.

## colmap (exhaustive matching)
* Computation is performed with **pycolmap**’s `exhaustive_matching`.
* All settings remain at their defaults (pycolmap 0.6.1).

## colmap (incremental mapping)
* For each clustered image set, `incremental_mapping` in **pycolmap** is executed four times.
* Parameters:

```python
mapper_options = pycolmap.IncrementalPipelineOptions()
mapper_options.multiple_models   = True
mapper_options.min_model_size    = 2
mapper_options.max_num_models = 15
mapper_options.init_num_trials   = 1000
```

## Merge maps
* For every cluster, maps are merged.
* Maps are selected for merging when:
  * Six or more images are shared between the maps, and
  * Each map’s `mean_reproj_error` is ≤ 2.0.
* Merging is carried out with **COLMAP**’s native APIs (not pycolmap):
  * `model_merger` combines the models.
  * `bundle_adjuster` optimizes camera parameters.
       - --BundleAdjustment.refine_focal_length = 1
       - --BundleAdjustment.refine_principal_point = 1
       - --BundleAdjustment.refine_extra_params = 1


[1] pycolmap.incremental mapping()

