# 2nd place solution

Competition: sartorius-cell-instance-segmentation
Rank: #2
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/297988

Thanks to Sartorius and Kaggle for hosting this interesting competition. I also would like to thank @steamedsheep for the great collaborative teamwork during the competition, we both work very hard to achieve this result.

Our solution is an ensemble of 2 object detection model, 1 unet and 2 maskrcnn model as shown in figure below. I am in charge of object detection and Unet while @sheep focuses on maskrcnn and ensemble. 


We use yolov5x6 and effdetD3 for the object detection task. The training procedure is the same for both model as shown in the figure below. 
The models are trained several rounds on Livecell and train-semi-supervised dataset before finetuning with the competition data. In the inference phase the output boxes of 2 model are ensemble with maskrcnn boxes using WBF.



The output boxes after WBF are feeded into a Unet and Maskrcnn (mask head) to get the segmentation mask for each box. We use weighted average to ensemble the raw mask of Unet and Maskrcnn. 

We use an unet with effificientb5 encoder to do segmentation on the cropped cell. Since the cropped cell sometimes include the neighbor cell, we predict the mask of the center cell and neighbor cell as 2-class segmentation. 





From left to right: cropped cell; ground-truth of center mask; predicted mask of center mask; ground-truth of neighbor cell; predicted mask of neighbor cell.
