# 26th Solution - 3D UNet + CCL with Morphological Post Processing + Custom Loss Function

Competition: czii-cryo-et-object-identification
Rank: #26
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561483

Hello,

First of all, I'd like to thank the competition hosts for the very interesting problem presented. I am in awe at the homogeneousity of the data, where our training 7 tomograms represent the data sufficiently well. The private-public split was also great, with minimal leaderboard changes.

Second, I'd like to thank the community, as always, for all the insightful discussions on the forum and the Kaggle spirit of sharing ideas. None of this would have been possible for me without the collective effort of the other participants. Special thanks to @hengck23, @fnands, @davidlist and @sacuscreed. I learned the basics of solving a computer vision problem from Hengck, and I learned a lot from dissecting his code.

# Solution Summary

My solution used MONAi's full 3D UNet implementation, with a window size of 184x184x184. I generated multi-class hard labels, using spheres of 0.5 the estimated particle radius. 

I ended up using a 50% overlap and generated gaussian weights for volume reconstruction. For the data augmentation pipeline, I used random rotations in the X,Y plane, random flips on all spatial dimensions, random gaussian noise and random zoom (0.95-1.05). 
I did also spend some time looking up ways of replicating the missing wedge artifact using affine transformations but ended up ditching the idea.

I attempted training using multiple loss functions, settling on Tversky loss with multiple alpha/beta combinations, obtaining decent results, with best individual models scoring 0.70x-0.73x on the public LB. 

For test-time augmentations, I used 90-180-270 rotations in the X,Y plane and full spatial dimensions flips. I used two particular ways of ensembling the TTA and models together:

1.  Raw Logit averaging, where I take the mean of the TTAs, add them with the other model predicted mean of TTAs and perform another average, reconstruct the volume and then apply a final softmax.
2. Softmax per TTA stack, then add max of probabilities together with the other model predictions for the subvolume, perform an average between models and reconstruct the volume.

Both methods ended up scoring very similarly, although obviously required different thresholds for CCL, with the max of TTA being overconfident in predictions. Looking back, I imagine I could get more ensemble variety if I combined the two methods, which I have not done.

Following thresholding to transform the output probabilities into hard binary masks, I played around with the following morphological operations: dilation, erosion, opening and closing. Fully dilating all my binary masks before performing CCL caused a jump on the LB score of around 0.010-0.015.

I then played around with Optuna HPO studies to search for best threshold/morphological operation per particle in order to maximize the pixel FBeta (beta of 1 or 1.25 performed best, going above would sacrifice too much precision) score for said particle (not to be confused with competition FBeta).

I have only validated on TS_99_9 and trained on the other 6 samples throughout the whole competition, and I tried finding a correlation between any local metric and the LB. I ended up using weighted pixel FBeta as I found the LB FBeta to be very unstable and not correlate too well on the leaderboard.

# Additional thoughts

A good part of January I spent on analyzing this problem, how to find this correlation. On the one hand, we are not working on a segmentation problem per se, so comparing the dice metric between two models does not say the whole story, on the other hand, there is some correlation between dice/IoU or whatever other metric you want to use and the way it will translate over to LB.

Say you already have a cluster of pixel predictions in the right place, and whatever peak detection method you use already 'scores' you that particle. Further improving those probabilities (getting them closer to 1, or covering more of the ground truth) improves dice without improving competition FBeta.

This has led me to think about a loss function that could prioritize identifying as many objects as possible (to encourage object level recall, directly translating to competition FBeta). 

```

def CustomLoss(self, predictions, labels, metadata):
    
    pred_mask = torch.softmax(predictions, dim=0)
    threshold_sigmoid = 0.05
    sharpness_sigmoid = 10
    eps = 1e-6
    n_objects = 0
    total_object_loss = 0
    objects_found = 0
    object_counter = 0
    for class_label, obj_id, obj_mask in metadata:
            
        pred_probs = pred_mask[class_label]
            if class_label == 2:
                continue
            
        tp = torch.sum(pred_probs * obj_mask)
        fn = torch.sum((1 - pred_probs) * obj_mask)  
        weight = class_weights[class_label]
            
        recall = tp/(tp+fn+eps)
        sigmoid_recall = torch.sigmoid(sharpness_sigmoid * (recall - threshold_sigmoid))
            
        if recall>threshold_sigmoid:
            objects_found += 1
            
        total_object_loss += (-torch.log(sigmoid_recall + eps)) * weight
                
        object_counter += 1
        n_objects += weight
            
    loss = total_object_loss / (n_objects + eps)
    print(f'Custom loss is equal to {loss}!')
    print(f'Objects found with over {threshold_sigmoid} in recall: {objects_found} out of {object_counter}!')
        
    return loss

```

The above function runs through the metadata for a particular batch (which is generated outside for each batch, to maintain differentiability) and iterates through classes and individual objects and calculates recall (assume you have 3 ribosomes and 5 apo-ferritins in this batch, metadata would create 8 ground truth masks, one for each particle sphere).

I then used sigmoid for soft thresholding and calculate loss as negative log of sigmoid of recall. The idea behind was to encourage the network to identify as many objects as possible (with the way metadata splits objects, only recall can be calculated, as the precision of the predictions would be way off considering it would take into account all predictions for that said class, dramatically decreasing precision).

This loss was combined with Tversky loss with a higher alpha to prioritize precision, in the attempt of antagonizing all the positive predictions that the custom loss function would generate to reach 5% recall per object.

This combination required a lot of tuning for the right recall threshold, the right class weights as well as the weights of the custom loss and Tversky loss blend (-log of epsilon would dominate Tversky which is bounded to [0,1]).

Unfortunately, I only got decent results with it, not really improving my LB score and I had to give up on the idea as I lacked the GPU required to further delve into this, plus the competition deadline was approaching.

In the end, I used a model trained with this combination, which at least helped the diversity of the ensemble blend.

# Lessons learned

This was my first time dedicating 3 months to a competition and it was interesting to see how the leaderboard evolved, the discussions on the forum and the shared code snippets. I now have better expectations of how to manage my time, my GPU, and I think the greatest take away would be to always revisit earlier parts of my code pipeline and be more mindful of how and why they work together.

I tend to tunnel vision on the task at hand and take the big picture less into consideration.

Sometimes, less is more, and taking a break from a problem you can't solve makes you come back with a fresher perspective.
