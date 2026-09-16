# [30th place solution] Weird TTA trick? Detect by quadrant provided a great boost!

Competition: global-wheat-detection
Rank: #30
Source: https://www.kaggle.com/c/global-wheat-detection/discussion/172827

Hi all! It's been a real pleasure working alongside you during this competition. Before YOLOv5 came into the mix, I was sitting in 2nd place on the public leaderboard thanks to a neat little trick that gave me a bigger boost than expected.

I was working off an EffecientDetB5 model which gave around 0.72 on the public scoreboard without any TTA. With my TTA trick I got a massive boost to 0.7535. I'll explain it below, and would love to get your opinion on why it might have been so useful. Also, here is a [notebook with a working example](https://www.kaggle.com/alexandersoare/gwd-detect-by-quadrant-tta-trick).

The motivation was that I didn't have time to be training 1024x1024, but I still wanted to work with full resolution while training 640x640. So the idea was:

1) Break the test images up into 640x640 quadrants (so there would be some overlap between them).
2) Do inference on all 4 quadrants and stitch the results together.
3) Do inference on the full image resized to 640x640 (to help fill gaps - more on this later).
4) Ensemble with WBF.

Here's a diagram explaining the quadrants + full image resized.



**About the stitching**

I didn't want to keep any boxes that got too close to the edge of the quadrants because they could cut through the middle of a wheat head. So I discarded them.



And now you can see why I also did inference on the rescaled full image. I wanted to capture any boxes that may have been discarded with the method above.

**About the training**

I found that with this method, mosaic augmentation during training didn't help my score. I stuck with doing random resized crops of the training images. The size of the crops were uniformly distributed in [640, 1024]. That way I had the benefit of training near full resolution a lot of the time.

**Question to Kagglers**

I'm actually not familiar with the concept of multi-scale inference. I understand it's a TTA technique. What exactly is happening in it? Is it the similar to what I did?
