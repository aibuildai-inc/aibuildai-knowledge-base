# Overview of approaches used

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #73
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35432

Congratulation to all the winners and medal winners and thanks to threeplusone for the great [script][1] to calculate the coordinates.

Overview of the approaches used:

 1. **Segmentation**: Segmentation is applied to the image to predict small squares centered on the coordinates of the dots as shown in the image below. [Tiramisu][2] network is used with 1:1 upsampling to predict 6 classes(5 types of sea lions and background), Log loss is used with Nadam optimizer for training. Results with this approach were not satisfactory, the model was biased towards predicting the background.
[Imgur]
 2. **Masking with segmentation**: A mask is created to focus the predictor on the areas where sea lions might be present using the same segmentation approach as above by increasing the size of the squares for sea lions and using the dice coefficient loss, it was able to provide satisfactory results in detecting the regions of interest but after running a regression it seems to overfit a lot, due to lack of time more experiments were not performed with this approach. An example is shown for predicting ROI
[Imgur]1. Actual Image   2.Mask   3. Predicted Mask
 3. **FCN Regression**: Thanks to [@mrgloom][3] we were able to get into top 100 using the fully connected net to perform regression on the whole image, an average of our two best-performing models give us our current position the leaderboard.

It was a great competition with a lot of data which helped us to learn a lot of new things, some of the other approaches that just remained on the to-do list were the new object detection API from tensorflow and using U-net for segmentation. We will share our github soon after performing some cleaning.

Cheers !


  [1]: https://www.kaggle.com/threeplusone/sea-lion-coordinates
  [2]: https://arxiv.org/abs/1611.09326
  [3]: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/33900
