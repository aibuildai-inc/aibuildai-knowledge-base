# 11th place

Competition: vesuvius-challenge-surface-detection
Rank: #11
Source: https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/11th-place

Thanks to organisers for a fun ride and congrats to the winners!

The very complicated metric implied that postprocessing would be playing a huge role. My solution was quite simple - a geo-mean blend of two 3D LowRes models (different patch size, normalization, etc) followed by 3D Cascade Fullres. The latter takes care of the postprocessing quite well. Even simple thresholding would brings to a top20-top30 region. A 5-step public post-processing on top covers the last mile. In fact, changing radiuses from 3-2 to 2-1 would have brought me to the 3rd place - but it was not giving any visual benefits on validation.  

Overall **very happy** with the 🥇11th place given my recent painfull experience of missing the Gold train by 1 position. Twice. In a row 😁

I did not quite go far enough to use AI to write the complete solution but it was indeed indispensible to explain how nnUnetV2 works, where to change the optimiser from sgd to adam (which saves a lot in training compute btw), where to change tta, step size, things like that. 

Cheers!
