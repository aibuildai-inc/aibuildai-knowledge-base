# [9 Place] Recall, Rotate and Zoom in

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #9
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583242

# Overview
We are thrilled to win a gold medal in this shake-up competition. Firstly, we'd like to express our gratitude to the organizers for hosting such an great game and for open-sourcing the entire workflow—from EDA to data processing, training, and submission. This significantly lowered the barrier to entry for us. We also want to thank all of the open-sourced authors for their contributions. Finally, a special thanks to my teammate, @forcewithme . We poured much effort into this competition, and we're so happy about the rewarding outcome

Overall, our inference pipeline consists of two stages. The first stage uses a very low threshold to recall as many candidates as possible, while the second stage applies an appropriate threshold to make the final decision.

Regarding training, we coudn't find a stable way throughout the entire process. In the end, all models we used were either YOLOv8 or YOLO11 models trained simultaneously on both external data @brendanartley and competition data, sharing the same training configuration.

## Inference Pipeline

1.  **Two-Stage Detection Pipeline:** 
    *   **Stage 1 (Candidate Generation):** Identifies initial potential motor locations. This stage uses standard model ensembles or SAHI-based ensembles.
    *   **Stage 2 (Candidate filter):** Further processes and validates candidates from Stage 1.
2.  **Ensemble Models:** Multiple models or different configurations of the same model are used in combination to enhance detection accuracy and robustness, which applied in both Stage 1 and Stage 2.
3.  **Test-Time Augmentation (TTA):** Since the test dataset has large variance in image size, we use multiple resolutions for inference to capture features at different scales.
4.  **Bypass Logic & Midpoint Reasoning:**
    *   **Bypass Logic:** Skip Stage 2 if Stage 1 generates highly confident detections, which improving efficiency with no cost. The bypass threshold is set to 0.6. 
    *   **Midpoint Reasoning:** If top2 detections are very confident and close(either in Stage 1 or after Stage 2), a new detection point at their geometric midpoint with a slightly boosted confidence will be created returned as prediction point.
5.  **SAHI:** In some slides in Stage 1(if Stage1 doesn't bypass), we use SAHI method to devide large tomogram slices into non-overlapping patches for inference, then merges results to enhance multi-scale detection.
6.  **Rotation-based Refinement with zoom in in stage 2:** Instead of rotation around the candidate, we crop a rotated zoomed-in square around the target, and pad the original image slice with the mean pixel value. 

---

## Model training

1. We trained yolov8l or yolov11l with ultralytics.
2. All of the models are trained with a mixture of external data('trust' 3) and competition data('trust' 4).
3. The external dataset is fixed by @tatamikenn .
4. We trained some 'local' model by random cropping around target while training. These models are trained for sahi or stage2.

## Submission Comparison

| Submission           | A  | B   | C   (Selected)         | D(Selected)            |E|
|:-------------------|:------------|:-------------|:-------------|:-------------|:-------------|
| **Overall Scheme** | Std+S2 | Parallel[Std+SAHI]+S2 | STD+S2 | Parallel[Std+SAHI]+S2 |STD+S2
| **Stage 1A Config**| yolo8l <br/> res:960/1280/832 | yolov8l no sam<br/>yolo11l sam:1/4 res:960 | yolov8 no sam res:960 <br/>yolov11 sam:1/2 res 960<br/>  | yolo8l no sam res:960 | yolov8l res:960 |
| **Stage 1B(SAHI)** | N/A | yolo8l<br/>sam:1/3<br/>patch:768 | N/A | yolo11-cz model<br/>sam:1/4<br/>patch:640 | N/A |
| **Stage 2** |  yolo8l z1.5 res512 <br/> yolo11-cz z1.5 res640 <br/> yolo8l-cz z2 res640 | yolo8l z1.5 res512 <br/> yolo11-cz z3 res640 <br/> yolo8l-cz z2 res640 | yolo8l z1.5 res512 <br/> yolo11-cz z1.5 res512 <br/> yolo8l-cz z2 res640| yolo8l z1.5 res512 | yolo8l z1.5 res512 |
| **Public Score**   | 0.837 | 0.853 | 0.856 | 0.856 | 0.860 |
| **Private Score**  | 0.853 | 0.855 | **0.852** | 0.832 | 0.826 |


### Configuration Notes
  - `Standard`: [Public inferece pipeline](https://www.kaggle.com/code/andrewjdarley/submission-notebook).
  - `sahi`: [Sahi inference pipeline](https://www.kaggle.com/code/fautei/byu-yolo-sahi-submission-notebook).
  - `S2`: Stage2 inference. Infernce on zoomin+rotate+crop images, with almost same code as `Standard`.
  - `sam{}`: The sample rate applied to the SAHI processing
  - `no sam`: No sampling, meaning taking all of the slices.
  - `max_image_side`: Maximum dimension allowed for input images before slicing
  - `patches_per_side`: Number of grid divisions (n x n) for each slice
  - `patch_size`: Target dimension for each patch (e.g., 768x768 pixels)
  - `res`: image resolution.
  - `-cz z{1.5/2/3}`: Means this is a 'local' model, 'cz' means crop and zoomin. And `z{}` indicates the zoom-in scale while inference.  

## End
Any questions or suggestions are welcome. Happy kaggling!
