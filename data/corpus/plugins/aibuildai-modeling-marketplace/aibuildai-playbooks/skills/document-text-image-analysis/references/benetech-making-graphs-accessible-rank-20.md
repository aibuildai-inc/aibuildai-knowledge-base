# 20th place solution (YOLOv7+EasyOCR)

Competition: benetech-making-graphs-accessible
Rank: #20
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418389

Congratulations to all the prize/medal winners of the competition🎉🎉. It was a great opportunity to learn and grow.I would like to share my approach here.
 
My approach to the problem involves two main steps: object detection and Optical Character Recognition(OCR).


**Object Detection(Yolov7-X):**
I trained a yolov7 model to detect the x-axis labels, y-axis labels, the chart bounding box, and the data points on the chart. The coordinates of the data-points were not provided in the dataset. I was able to accurately compute the position of the datapoints on the chart images by linear interpolation of the x-axis and y-axis tick coordinates with respect to the x-axis/y-axis labels (values). [Here](https://www.kaggle.com/samratthapa/data-prep-for-yolov7-data-point-x-y-label-chart) is a notebook to create a dataset using this approach. 

During inference, I inverted this process by calulating the data-series from the linear interpolation of the values of x-axis and y-axis labels with respect to the co-ordinates of the data points.

This approach also works relatively well for scatter plots, compared to other approaches like Donut.

Notes:

- This object detection model was also used as the chart-type classification model.
- There are overlapping bounxing boxes for some x-axis labels like the image below. However, my OCR model was able to extract the correct text despite the input image including text from neighbour bboxes.




**OCR model:**
Using the EasyOCR libary, I trained a ResNet(feature extractor)+BidirectionalLSTM model with Connectionist Temporal Classification(CTC) loss. The additional dataset(ICPR 2022 CHART) improved the accuracy of the OCR model by about 5% from 84% to 89%. I used [this notebook ](https://www.kaggle.com/code/nadhirhasan/let-s-finetune-easyocr-boom)as a starting point to train the OCR model (Thank you @nadhirhasan ).

**Post processing:**
After receiving the bounding boxes from the model, I performed some post-processing based on some simple heuristics like: removing the data points that lie outside the chart bbox, restricting x-labels(y-labels for horizontal-bar) to lie under the chart bbox, and restricting y-labels(x-labels for horizontal-bar) to the left side of the chart bbox. Also, the x/y axis tick coordinates are calculated using the x/y-axis bbox, and the chart bbox. I use the nearest point that lies on the chart bbox from the center of the x/y label bbox as the respective x/y tick coordinate. I chose this approach because the precision and recall of the x/y labels was higher than the x/y axis ticks in an older version of the model.


I participated in this competition only for the last 4 weeks. So, due to lack of time, I wasnt able to try out other approaches like Donut . I think there is a lot of room for improvement for this model. For example,about 25% of the predictions made by the model automatically get scored 0, because of mismatching number of predictions. This mismatch is due to only 1 or 2 points for charts besides scatter plot.

Submission Notebook:
https://www.kaggle.com/code/samratthapa/20th-place-submission-yolov7-easyocr/notebook?scriptVersionId=134338733

For model checkpoints, and training scripts: 
https://github.com/SamratThapa120/benetech-making-graphs-accessible
