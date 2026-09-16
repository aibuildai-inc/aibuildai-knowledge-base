# 19th place solution overview [and trained weights]

Competition: google-ai-open-images-object-detection-track
Rank: #19
Source: https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64734

I trained [yolov3][1] with SPP using [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74) weights pretrained on Imagenet.

I followed the training regime and used the same architecture that the creators of yolo used for training on COCO. Most likely better results could be achieved on the Open Images dataset should the capacity of the model be increased.

I emulated the conditions of training on COCO and created two custom datasets. I initially (up to ~ 250k batches IIRC) trained on ~320k files, increasing the dataset to ~430k files onwards. I picked the files to train on to ensure as balanced a representation of each class as possible (picked all images that contained rare labels - limited number of images to include for more popular labels to arrive at the target file count).

I trained the model thx to the GCP coupon using a preemptible instance / premptible GPUs. I went with 16 vCPUs and 4 preemptible P100s. I didn't measure this precisely but I think in 24 hrs the model trained on ~50k batches, maybe a bit more. The full training schedule is just over 500k batches.

I believe that appropriate makeup of the trainset is key to achieving a good result. This is an observation that I have not verified though.

I have not modified the train set labels in any way. I also don't believe I have expanded the labels before training to include parent categories but might be wrong on that one. Doing something reasonable with the label hierarchy should grant further improvement in performance.

My private LB score is slightly different as I tried training on a subset of labels / files for trainingtrain set selected in a specific way to help me in the VRD track but I didn't get very far (partially due to this likely being the wrong approach and partially due to running out of time).

I am sharing the final weights and the config files in this github [repository](https://github.com/radekosmulski/yolo_open_images).

Here is the citation of the paper that introduces yolov3:

&gt; @article{yolov3,   title={YOLOv3: An Incremental Improvement},  
&gt; author={Redmon, Joseph and Farhadi, Ali},   journal = {arXiv},  
&gt; year={2018} }

And [here][2] is where you can find the paper on arxiv.

  [1]: https://pjreddie.com/darknet/yolo/
  [2]: https://arxiv.org/abs/1804.02767
