# My solution for the RSNA 2024 Lumbar Spine Degenerative Classification, using only the competition data

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #31
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539555

I would like to extend my sincere thanks to RSNA and Kaggle for organizing such an amazing competition. This was my first time participating solo, and the journey has been an incredible learning experience. I’m excited to share my solution with the community and am looking forward to receiving valuable feedback to improve my skills for future competitions. Thank you again for this opportunity!


**Step 1: YOLO Object Detection**

I started by utilizing the YOLO object detection algorithm, trained on the competition dataset, to detect bounding boxes for each lumbar spine level across different modalities. This allowed me to pinpoint the exact regions of interest for further processing.

**Step 2: Identifying Axial Slices and Directions**

Using the coordinates of the spinal canal bounding boxes, I identified the three closest axial slices for each lumbar spine level. Additionally, these coordinates helped me determine the left and right directions in sagittal slices, which were crucial for ensuring accurate cropping of the relevant images.

**Step 3: Image Cropping and Preprocessing**

With the spinal canal bounding boxes and directional information, I used YOLO to extract cropped images for each condition, based on the direction and spinal level. I created three cropped slices of the spinal canal, six images for the neural foraminal area, and six images for the subarticular space. These images were essential for training the model.



**Step 4: Pretraining the Model**

I pretrained a model that processed the 3 cropped spinal canal slices, 6 neural foraminal images, and 6 subarticular space images. Each image group was passed through a dedicated backbone—in this case, EfficientNetB0. The extracted features from each group were then fed into a Dense Neural Network to classify them into five categories, each with three classes: Normal/Mild, Moderate, and Severe.



**Step 5: Fine-Tuning for Each Lumbar Spine Level**

After pretraining the model, I fine-tuned it for each specific lumbar spine level. This allowed me to generate a trained model for each level of the lumbar spine, ensuring precise classification for all levels.
