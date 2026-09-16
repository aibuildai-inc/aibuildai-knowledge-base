# 7th place solution - no external data

Competition: benetech-making-graphs-accessible
Rank: #7
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418510

For this competition, I chose to use a pipeline of multiple models instead of an end-to-end model (such as Donut, Deplot, or Matcha). I made this choice because the baseline end-to-end model model I tested with Kaggle data only did not perform well, and I thought it may require a powerful machine to train and cost time to generate data.

# Models

- Image classification: used to classify chart types.
- Text detection: used to detect x/y labels.
- Text recognition: used to read the crops (pretrained model only, no fine-tuning).
- Object Detection: used to detect x/y ticks, scatter points, and vertical and horizontal bars.
- Object Segmentation: used to segment lines on line charts and segment vertical bars (to classify if the vertical bar is a histogram or not).
- Pretrained Deplot Model: used for cases where the pipeline cannot predict anything (applied only in one case out of 559 CV files).

# Data

- Use only Kaggle data.
- Split the `extracted` data into a 50% training set and a 50% validation set.
- Use all `generated` data for training set
- Finally, use all the data to train the final weights for each type of model.

# Result:

Best CV - LB (with train data only): 0.871 - 0.86

Private LB: 0.67

Best public - private LB: 0.86 - 0.69

# Pipeline Details

**Step 0: Chart type classification**

I used a v2s backbone model to classify the charts, with a CV accuracy of 99.2%.

## General components/steps applied for all chart types:


**Step 1**: Detect and read x/y labels:

- First, use a text detection model to detect the polygons of x/y labels.
- Then, use a text recognition model to read the text inside the polygons.
- Post-processing:
    - For x labels, draw a horizontal line from top to bottom of the image and select the line that intersects the largest number of x labels polygons. This is because the text detection model is not always accurate.
    - Repeat the process for y labels.

**Step 2**: Use an object detection model to detect x/y ticks.

- Post-processing: Similar to x/y labels, I draw a horizontal line to filter x ticks and a vertical line to filter y ticks.

**Step 3**: Map x/y labels and x/y ticks.

- Based on the relative position between labels and ticks, I create a 1-1 mapping for these label-tick pairs based on IOU in Ox direction and ignore all other labels/ticks that can't be paired.
- Because there are cases where the x labels are not straight, to make the mapping more accurate:
    - Get the rectangles of x_label from polygons.
    - Draw a rhombus with vertices at the center of the rectangle edges.
    - Draw a new rectangle with the center at the highest vertex of the rhombus.
    - Then create a 1-1 mapping with x_boxes similar to mapping x labels and x ticks.

## For a vertical bar chart


**Step 4:** Detect the point on top of each bar using an object detection model. I will call these points "value boxes" because I predicted boxes instead of points.

**Step 5:** Map x ticks and value boxes.

- This step is similar to mapping x labels and x ticks.

**Step 6:** Get the final value of each bar.

- Project the center of each value box to Oy, then get the final value by comparing it to the two nearest y ticks' values.
    - `value_box_value = y1_value - abs((y2_value - y1_value) / (y2_pixel - y1_pixel) * (value_y_pixel - y1_pixel))`

**How to check if a vertical bar is a histogram:**

- Because all columns of a histogram always sit next to each other without space, I use a bar segmentation model to detect the bar region.
- Calculate the percentage in the Ox direction. If the segmented region (in the Ox direction) is greater than 95% of the distance between the last bar and the first bar, then it is a histogram.

## Horizontal Bar Chart


I process a horizontal bar chart follow these steps:

- Rotate the chart 90 degrees and then flip it horizontally to make it similar to a vertical bar chart.
- Use the same steps as for a vertical bar chart to process the horizontal bar chart.

## Line Chart


**Step 4**: Use a segmentation model to predict the lines in the chart. For dashed lines, connect them to make them continuous.

**Step 5**: Project the line onto the Ox axis and keep only the x labels and ticks that can be projected onto the line in the Ox dimension. Then, get the corresponding value boxes for the remaining x labels.

**Step 6**: Obtain the final value for each value box:

- Project the center of each value box onto the Oy axis.
- Compare the projected value with the values of the two nearest y ticks to get the final value.

## For scatter chart


**Step 4**: Detect scatter points using object detection models. I find the box size of scatter points by getting the largest rectangle connected component that contains the scatter point, then average all the box size in the chart to get the final box size for all scatter points. These predicted points will be referred to as "value boxes" in the following steps, in order to maintain consistency with other chart types.

**Step 5**: Map the x/y ticks and value boxes.

- Similar to mapping the x ticks and value boxes in a vertical bar chart, we do the same for the y ticks in the Oy dimension.

**Step 6**: Obtain the final value of each value box.

- Project the center of the value boxes to Ox/Oy, then obtain the final value by comparing them to the values of the two closest x/y ticks.

## For dot chart

Similar to vertical bar chart, except if the x labels are numerical, we use scatter postprocessing methods instead.

### And another post-processing functions using computer vision based algorithms, some highlights:

- To avoid errors in text recognition and detection, keep the y-label values as the longest increasing sequence.
- Splitting words inside polygons since the pretrained text recognition model doesn't include spaces,, I use another word detection model. Then sort the words based on their geometric location can help with accuracy.

More detail in the notebook. The inference notebook is public here: [https://www.kaggle.com/code/thanhhau097a/bmga-submission-0-86](https://www.kaggle.com/code/thanhhau097a/bmga-submission-0-86)

Cheer
