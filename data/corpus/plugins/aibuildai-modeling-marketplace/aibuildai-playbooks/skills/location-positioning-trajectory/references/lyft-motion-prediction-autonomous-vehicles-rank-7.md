# 7th place solution - Peter & Beluga

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #7
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199649

#### Acknowledgements 
Thanks for the organizers for this tough but certainly interesting challenge.
Special thanks for Vladimir Iglovikov and Luca Bergamini for their active forum contribution during the competition.

Hats off to my team mate @pestipeti by the time I joined him he already had optimized the hell out of l5kit and had a solid training framework.
Then he managed to boost the training speed even further by [rasterizing the images on GPU](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199583).
With all those improvements we were able to run dozens of experiments with different config parameters and slightly modified encodings during the last months.

#### Back to the future
We noticed that the training dataset and chopped validation set had slightly different feature distributions. After some digging we found that the chopped datasets (valid, test) always had availability for at least 10 future frames. It was quite the opposite than the default `AgentDataset` settings so we used that for training too. 

```python
AgentDataset(
    cfg, dataset_zarr, gpu_rasterizer, agents_mask=dataset_mask,
    min_frame_history=1, min_frame_future=10
)
```

It helped both in terms of validation consistency and final score.

#### Poor Man's Ensembling
We did not hope that blending or any simple heuristic would help to combine different models. (I read clever tricks though and I hear that stacking works too...)

The speed of the agent matters a lot and we saw that in our experiments. Intuitively for slower objects we used smaller raster size but more history.
Our final and best submission used three models based on speed (Total distance in the last 1 sec) 
* [0-2] Slow model 320x220 with 3 sec history (compressed by 1.5 s) trained for 7+ days on slower examples
* [2-5] Medium model 320x220 with 3 sec history (compressed by 1.5 s) trained for 5+ days [1.5-12]
* [5+] Fast model 480x320 with 1 sec history on separate channels trained for 9+ days on [2+]

#### Things that did not work
* We tried to use additional meta data (speed, acceleration, position, hour of the day, day of the week etc.) but it did not really help.
* We did not use satellite images at all. We noticed that they could have additional info (especially for pedestrians or cyclists) but it would be too slow.
* Different backbone. We tried a few other networks but mostly used Effnet-B2.
