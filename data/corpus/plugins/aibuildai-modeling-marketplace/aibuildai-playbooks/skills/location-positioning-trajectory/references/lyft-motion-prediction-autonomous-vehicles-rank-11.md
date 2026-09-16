# Things I have tried and my final solution

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #11
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199541

Congratulations to all the winners and thank the host for this very interesting competition. I have to admit that I did not expect a medal at all at the beginning. At some point I almost gave up, as the training was painfully slow, and I just could not get a reasonable score. This result gives me great motivation to keep trying  in the future.

Forgive me if my terminology does not make sense. Any feedback would be greatly appreciated. 

To my understanding, this is not a problem of finding three most likely future routes, which is equivalent to finding the one most likely route, as the second most likely route will always be the most likely route shifted by one nanometre. Instead, this is a problem of finding three routes that could best represent the probability distribution. Ideally, we want to include the less likely, but nonetheless typical routes. The most damage to our score probably will be caused by those less likely but very different routes. Therefore, we want diversity in our predictions, and simple ensemble might not work. 

My work is based on the model shared by @corochann [here](https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence). While this simple approach to achieve multi-mode prediction is surprisingly effective, there is one thing that does not make sense to me, and I have been fighting with this problem most of the time: 

At every training step, the coordinates of all three predictions are all pulled towards the ground truth. The confidence of the prediction that is most close to the ground truth is increased, while the confidences for the other two predictions are decreased. Therefore, for the other two predictions that are relatively further from the ground truth, we are decreasing their confidence values (meaning now we think they are less likely) but pushing their coordinates to the ground truth (meaning making them more likely). Although at the early training stage this should not be such a big problem and the model is able to converge, I just cannot believe it can converge to an optimal point. 
Other models such as classification models or NLP models would not have this problem as the target possibilities are fixed and finite. Here we are basically assigning confidence values to moving targets. 

I thought about several solutions: 
- I constructed a “diversity” factor in the loss function. It is basically the average distance between three predictions. By adding this to the loss I was hoping I could gently push three predictions away from each other, reduce the effect of three predictions being pulled together. However, my experiments were of no success. The model either totally ignored this factor or used this factor as the only way to gain lower loss. I did not try many times because every experiment took too long. 
- Make the coordinate space discrete and finite and assign a confidence value to every possible point in this space at each step. Then generate randomly many possible future routes. Finally do a k-means clustering to cluster the routes into three groups and take the centre of each group as the final prediction. I did not even finish the implementation of this idea as the computer power needed would be out of my reach. 
- Very large batch size. It was until very late into the competition I suddenly realized that maybe increasing the batch size could mitigate (certainly not resolve) this problem. By letting the model see as many future possibilities as possible at each step, the model might learn to maintain the diversity of its three predictio[](url)ns. It indeed worked, although to be honest I am not sure it was only because of the problem I mentioned above. 

So, my final solution might seem surprisingly simple to most people. I just used the good old Resnet18, a very small image setting of 150x150 with only 5 history frames, which enabled me to fit in a batch of 512 samples into my 8G VRAM. The optimizer is again good old Adam, with a learning rate starting from 0.0001 and reduce by half every 50000 steps. I trained on the full dataset, not because I think I need so much data, but to mitigate the problem of overlapping samples as discussed [here](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/185762).  

I trained 400k steps, which intermittently took me more than 15 days!!! After this, my computer and I were so exhausted, so we did not try other models or optimizers. 

(I tried accumulating gradients to increase the effect batch size but again the training was too slow, and the early result was not fantastic, possibly because of the incompatibility with Batch Normalization.) 

I guess, if we could make the batch size even bigger, and image size a little larger, train longer, or maybe use a more sophisticated model, there is potential to further improve the score significantly. 

By the way, I have never really solved the problem of deviation between my training loss and validation loss. After removing some problematic parts of my model and setting the min future and history frames in alignment with the validation dataset, the problem was only partly solved. My training loss has reached below 10 but validation loss was never below 12.10. This is not too bad, but I know some of you get much better alignment. How did you guys get the scores aligned?
