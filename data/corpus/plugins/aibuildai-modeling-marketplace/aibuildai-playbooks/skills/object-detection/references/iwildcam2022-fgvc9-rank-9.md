# 9th place solution

Competition: iwildcam2022-fgvc9
Rank: #9
Source: https://www.kaggle.com/c/iwildcam2022-fgvc9/discussion/328526

Congratulations to the winners and thank you to the competition hosts, current and past, for continuing to sponsor this important, enjoyable competition.  I confess that it still challenges me even in its reduced animal-only, counting form.  Goats! How can anyone count goats, wandering here and there, staring into the camera?  I would love to hear how the top teams - anybody for that matter - approached this counting problem.

**Outline of my approach**
* Use *MegaDetector* v4 detections to train a 2nd animal detector based on [*Yolov5*](https://github.com/ultralytics/yolov5).
* Merge MegaDetector and Yolov5 detections using *weighted boxes fusion* [WBF](https://github.com/ZFTurbo/Weighted-Boxes-Fusion).
* Apply a custom frame-to-frame object tracking algorithm focused on individuals within herds/packs. 

**The herd/pack species that I focused on:**
| Category Id |Common Name |
| --- | --- |
|2|white-lipped peccary|
|8|collared peccary|
|70|wild goat|
|71|domesticated cattle|
|72|domestic sheep|
|90|african bush elephant|
|96|impala|
|256|dromedary camel|

**Observations**

- I first learned about WBF in the [Kaggle COTS starfish competition](https://www.kaggle.com/c/tensorflow-great-barrier-reef) where it was used by many teams to post process detections. I used it successfully to merge the sponsor-provided MedaDetector detections with my Yolov5 detections and it gave me a public/private LB score of 0.275/0.265 (above Benchmark: iWildCam 2021 winner). I spent some time tuning the algorithm's hyperparameters but I'm not sure it provided any significant benefit beyond my initial success.

- I found plenty of anecdotal evidence that an inter-frame counting approach could improve on the standard **max** approach of the benchmarks, but I was never able to get my solution to provide any benefit once I scaled up to the whole dataset.  Since the approach works well in video, I hoped it would work with some of our sequences. The first step in my process was to try to estimate the overall direction a herd was moving so that I could eliminate many candidates from the inter-frame matching process. Again, this was easy to do in many cases, but unreliable overall, leading to weaker matching in most cases. I'm still looking into the details of exactly what happened.

- Additional detail for the inter-frame matching approach:  Herd sequences were identified by a 9-class Yolov5 detector. For any sequence with a preponderance of herd detections, detection-level matching was performed using the following method.  An autoencoder was trained on small chips from the center of mass of the DeepMac mask for each detection. The autoencoder produced a 512-element latent feature vector for each detection. These latent features (along with X,Y, area, delta-T) where used in detection-to-detection matching within the sequence.  Using the DeepMac segmentation masks to select which chips are fed to the autoencoder showed improvement over using the entire detection (scaled) or the center of the detection. Presumably this is due to the elimination of non-individual pixels due to occlusion and/or background between the individual's legs.

- My largest counting errors in the validation set usually involved domestic cattle. Consequently, I spent a lot of time looking at sequences of domestic cattle and getting a little discouraged about not being able to spend more time with the more *exotic animals*. Then I heard a story on public radio about how damaging cows can be in many different situations.  This got me thinking about camera traps for habitat destruction monitoring and suddenly counting cattle seemed much more significant.
