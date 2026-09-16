# Public 34th/Private 47th place insights - Part1 (Inference)

Competition: deepfake-detection-challenge
Rank: #47
Source: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/140303

Hi all,

There is a lot to say about this competition. First I would like to share about our inference pipeline here and then I will make another post about the training part. Our solution uses only videos/frames (no audio at all) and total frames used in inference was quite important for us, the more the better.  We tried to move frames into GPU early in the pipeline to benefit from GPU fast computation in the next stages. We've fighted with NVidia DALI and Decord as they can load/decode videos through GPU, unfortunately it never worked, Decord had a memory leak and DALI finished with timeout during submission (impossible to troubleshoot). So we used OpenCV and moved decoded frames in GPU just after. 

Then, for **faces extraction**, we've used MTCNN with facenet-pytorch with small changes to make it support GPU input tensors. Prior to MTCNN, we added a GAMMA correction (still in GPU) on frames to help MTCNN (some dark videos or bad contrast). Outputs are faces resized to 256x256, aspect ratio preserved and large margin (30 pixels on each border).

Next steps is **face tracking** to identify how many faces in a videos, cleanup artifacts (face detected but not face) with different rules (based on tracked confidences and maximum faces). It is based on centroid boxes tracked across even spaced frames.



Next is **models inference**. We've 3 to 4 EfficientNet CV5 models with different input shapes (256x256, 240x240, 224x224) cropped by the normalizers. Each model was trained with a different validation strategy. Some with full data, some with partial data to make hold-out validation.

Final step is **post-processing**. One may notice that some videos have blinking fake faces (a few frames real and a few frames fake within the same video). Probabilities of our models was moving up and down which means it worked but using simple average for final probablity would not work:



So we decided to evaluate what could be the thresholds to detect such behavior and use maximum probability instead of average in this case only. The question was how many frames with high probability should we have to consider it's a fake:



And the answer was around 20%. So if 20% of frames have probability higher than around 0.85 then we prefer selecting maximum probability instead of average.

There is an additional/optional step with ridge regression (not classification) built on hold-out data and applied to models' output.

This pipeline works only if we have enough frames and we were able to run it up to 100 frames per video. Each single EFB model got around LB=0.32. Ridge regression on ensemble got LB=0.29 and thresholds provided boost to LB=0.27.
