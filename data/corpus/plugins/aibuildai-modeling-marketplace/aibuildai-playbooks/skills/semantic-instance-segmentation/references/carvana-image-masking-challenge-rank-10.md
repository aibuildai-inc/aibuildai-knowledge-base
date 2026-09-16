# LB 10th place 0.9971 solution

Competition: carvana-image-masking-challenge
Rank: #10
Source: https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40133

We created three models, all variants of Unet, and then used some post processing techniques to more robustly detect antenna's and car tops by colors.  The three models were ensembled with an equal vote.

1st model- Background detection.  Inverted the mask image and created a Unet to detect the background (instead of the car).  5 up/downsamples. Then inverted the final background prediction mask to convert back to the car prediction.  This model on it's own scored 0.9970 and was the highest performing of the three.    It seemed to do best at the areas underneath the car, wheel/spokes, and detecting true car edges on difficult test images.

2nd model - Classic Unet architecture to detect car.  6 up/down samples.

3rd model - Split images in half for top bottom, trained Unet for each, and then concatenated final results.  Same architecture as 2nd model, just on half images.

Post processing:
It appeared that many of the antenna's on the cars were non-continuous in the predictions.  I id'd these by counting the external contours of small contours that were within 50 pixels of each other and above the center of mass of the largest contour of size &gt; 100k pixels.  I then cropped an image containing the antenna  range and ran a simple canny that ran through the already identified points.  This seemed to clean up the antenna's substantially, but my estimate on final score impact was only around 0.00004.

Used Keras with TF backend, and primarily ran on a 4x 1080 Ti setup.  Found early on that running the Ti's for many hours straight caused them to get too hot and throttle their speed, so I had to install watercoolers on each of them to keep them at stable temps.

Another interesting finding that I didn't take full advantage of was the difference in the stage backgrounds.  It was stated by Carvana in one of the posts that there were three stages used, and if you look at enough images carefully you can identify the three distinct backgrounds.  One of the three backgrounds performed substantially better (0.0001) across all cars when blocked by color, size, and rotation.  I think there was opportunity to increase the score by creating custom predictors off of each of the backgrounds separately.
