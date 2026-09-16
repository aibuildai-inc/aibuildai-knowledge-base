# 44th Place Solution: [BYU - Locating Bacterial Flagellar Motors 2025]

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #44
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583294

First of all, I would like to thank @playwithme for sharing an excellent notebook. This is the notebook I referred to during the competition.
LB0.81-MHAFYOLO -tta-wbf-Submission Notebook(https://www.kaggle.com/code/playwithme/lb0-81-mhafyolo-tta-wbf-submission-notebook)
The model used in the notebook (https://www.kaggle.com/models/yyyy0201/mhafyolo/PyTorch/default) was pre-trained and publicly available.

My main improvement was quite simple — I only modified one of the functions

```python
def perform_3d_nms(detections, iou_threshold):
    """
    3D NMS-like clustering that considers spatial distance and Z-slice thickness.

    Parameters:
        detections (list of dict): Detection info list [{'z': int, 'y': int, 'x': int, 'confidence': float}, ...]
        iou_threshold (float): Threshold to scale the XY spatial distance limit (e.g., 0.2)

    Returns:
        list of dict: Representative detections after NMS (can include multiple clusters)
    """

    if not detections:
        return []

    # Clustering conditions
    xy_distance_threshold = 24 * iou_threshold  # 24 is the assumed side length of a box
    z_distance_threshold = 10                   # Acceptable range in Z-slice direction
    min_z_span = 3                              # Minimum number of Z-slices to consider as a valid cluster

    # Sort detections by confidence in descending order
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    final_detections = []

    while detections:
        base = detections.pop(0)
        group = [base]
        z_set = {base['z']}
        rest = []

        for d in detections:
            dz = abs(d['z'] - base['z'])
            dy = abs(d['y'] - base['y'])
            dx = abs(d['x'] - base['x'])
            if dz <= z_distance_threshold and dy <= xy_distance_threshold and dx <= xy_distance_threshold:
                group.append(d)
                z_set.add(d['z'])
            else:
                rest.append(d)

        # If the group spans enough Z-slices, take the highest-confidence detection
        if len(z_set) >= min_z_span:
            best = max(group, key=lambda g: g['confidence'])
            final_detections.append(best)

        detections = rest

    return final_detections

```

Function Overview
The perform_3d_nms function performs 3D clustering of detection points, mimicking a Non-Maximum Suppression (NMS) mechanism across 3D space. It filters out redundant detections by identifying spatially close points across Z-slices and keeps only the most confident detection within each valid cluster.
________________________________________
Parameters
•	detections: A list of detection dictionaries, each with keys 'z', 'y', 'x', and 'confidence'.
•	iou_threshold: A scaling factor for the XY distance threshold (e.g., 0.2).
________________________________________
How It Works
1.	Threshold Definition:
o	XY distance threshold = 24 * iou_threshold
o	Z-slice range threshold = 10
o	Minimum Z-slice span to be a valid cluster = 3
2.	Sort detections by confidence (descending)
3.	Clustering Process:
o	Use the most confident detection as the base.
o	Group together detections that are close in Z, Y, and X directions.
o	If the group spans at least 3 Z-slices, keep the one with the highest confidence.
4.	Remove processed detections and repeat
________________________________________
Returns
•	A list of representative detections (with the highest confidence) from valid 3D clusters.

The following hyperparameters were used.
CONFIDENCE_THRESHOLD = 0.7  # Lower threshold to catch more potential motors
MAX_DETECTIONS_PER_TOMO = 1  # Keep track of top N detections per tomogram
NMS_IOU_THRESHOLD = 0.7  # Non-maximum suppression threshold for 3D clustering
CONCENTRATION = 1 # ONLY PROCESS 1/20 slices for fast submission
SIZE = 1024
