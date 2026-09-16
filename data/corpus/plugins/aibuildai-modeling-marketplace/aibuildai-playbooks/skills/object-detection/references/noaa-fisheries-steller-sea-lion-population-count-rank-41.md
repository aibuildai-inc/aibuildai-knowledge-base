# My approach using coordinates as regression target

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #41
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35456

Firstly, thank you very much @threeplusone for the sea lion coordinates. I learned a lot of thing from your code.

I'm not so high in the leader board but I guess my approach may be worth mentioning.
I tried to feed the given information as precise as possible. So I used each sea lions' coordinates as regression target.

I scaled down the input to 1/2 size, and applied UNET. But at the end of network, I replaced sigmoid with two separate NN path:
       - 1x1 Conv2D with depth 1024 to predict whether sea lion is nearby in this pixel
       - 1x1 Conv2D with depth 1024 to predict two numbers - relative offset of the nearest sea from this pixel.
Loss function for offset is like sum(true_hit * (RMSE between true coordinate and predicted coordiante))

This works quite well for isolated sea lions but didn't work well when multiple sea lions are nearby. The attached images shows predicted location of sea lions and brighter red is more confident prediction.

This approach is quite slow. After seeing other solutions, I realized that I used too high depth for conv layers - higher depth yielded a little bit better solution, so I kept increasing depth for UNET. But now it looks like that tuning the simpler model with scale/augmentation was better use of time.

Anyway this approach got me find 70% of sea lions on validation set, and I noticed that the orientations of predicted point clusters are usually aligned with sea lion orientation(used skimage.measure.label and regionprop). So I rotated the sea lion patch and cut 1:2 area only - this reduced the amount of data for second pipeline that classifies sea lion type and also increased the classification accuracy by 10%.

I got RMSE 16 from validation set, but public leader board score was just 25. I couldn't get it any higher even after a month of tuning.

And after seeing mrgroom's solution, I decided to try similar approach - split image to 512x512 patch and predict the counts. This method is much simpler but yielded LB score around 25 also :(

I wasn't thinking of merging two results until two days before deadline at all. 
But after merging two results, I got public LB score around 21.13.
