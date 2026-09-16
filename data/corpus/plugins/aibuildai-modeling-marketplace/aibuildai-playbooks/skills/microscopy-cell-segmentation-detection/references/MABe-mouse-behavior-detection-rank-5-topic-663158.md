# 5th place - GNN + Egocentric Squeezeformer

Competition: MABe-mouse-behavior-detection
Rank: #5
Source: https://www.kaggle.com/c/MABe-mouse-behavior-detection/writeups/5th-place-gnn-egocentric-squeezeformer

This was an interesting competition, big thanks to the organizers and congratulations to everyone who participated. From the beginning the idea was building a unified neural network model that can support different labs and bodyparts. My focus was on robustness and diversity with the hopes that it will achieve transfer learning across labs and unseen videos.

#Overview/TL;DR
1. **Architecture**:  GNN (TransformerConv) + Temporal Transformer (Squeezeformer) + Pairwise;RelationNetwork classification head
2. **Key Insight**: Social aspect and egocentric processing were very important for my solution. High dropout and model diversity for robustness and transfer learning of different labs/unseen videos.
3. **Features**: 
    - Standard features: per mouse velocities, accelerations, jerks, angular velocities, movement heading.....
    - Mouse specific features: body curvature, body length, body elongation ratio, ear spread, body length change, head angle, tail angle.....
    - Inter-mouse features from the perspective of each mouse:  centroid distances, distance changes, various body parts distances (e.g nose-to-tail for  sniff etc), relative speeds, approach angle etc.
    - 58 features per mouse resulting in 232 dim input. (with some variance between different training runs/model families)
4. **Training**: Biased action-rich window sampling, focal loss with per-class weights, AdamW + OneCycle, FP16 training+inference


# Data Pipeline & Feature Engineering
All of the Tracking and Annotation parquet files files were preprocessed and saved as .npy for faster training. 
One problem with this competition was that different labs track different bodyparts. To solve this I went for a master skeleton approach, where we take a subset of the bodyparts that are the most represented in the dataset. This resulted in 6 master skeleton bodyparts: 

[nose, ear_left, ear_right,head_center,body_center, tail_base]. 

When some of the keypoints were missing there was a fallback system. In the head scenario we first went for the head, then neck, then average ear_left/right. All keypoints were normalized by pix_per_cm scale.

Feature engineering consisted of extracting per-mouse and inter-mouse features to help the model with the self and social actions. Model was provided with standard features such as velocities, accelerations, jerks etc. along with mouse specific features such as body configuration features, headings, extremity features etc. Along with these features one important part was inter-mouse invariant features. In the model **NO** raw coordinates were used. All of the features were invariant since they are relative to ones body or between two mice, my intuition was that attack is the same whether it is in the middle of the cage or on the side - but the relative angles/distances between the mice is what describes an attack as attack. Final feature input vector to the model was [Batch(32), Time(512), Mice(4), Features(58)]. On some models very light augmentations were used such as augmenting the scale of features, adding noise and keypoint dropout. 

# Architecture

The models architecture is GNN + Squeezeformer + Classificaton Head. For the GNN i used TransformerConv to make the social interraction between the mice. Before this I tried modeling the interaction between the mice with convolution with the num_mice as the channel dimension but that gave only a small improvement and it had bigger gap over the CV/LB so I dropped this approach since the GNN was more effective and better modeled the cage interactions. After passing the features through the GNN the model now is socially aware and each mice features are enriched by the other mice's features in the cage. This feature vector is then passed into the Squeezeformer with the caveat that instead of collapsing the [B, T, M * F] dimension we collapse the B * M dimension into [B * M, T, F]. We therefore process each mouse trajectory "separately" in the temporal dimension. To fit into my GPU I had to reduce the Batch size by a factor of 4 because now we have x4 more data points in the Batch dimension. After passing the data through 4 squeezeformer blocks we reshape back to [B, T, M, encoder_dim] and we pass the data through the classifier. The data in the classifier with broadcasting and concatenation over the last dimension it becomes pairwise: [B, T, M, M, 2D] which is passed through MLP and gives us the final logits for each pairwise interaction. The logits from the classification head are further masked with the behavior_mask which only lets the labeled behaviors in the current video to contribute to the loss. The final shape is [B, T, M*M, Actions] where each frame is classified into one of the possible actions+no_action. 



#### Model Diversity*
- On top of the main architecture, there were other adjustments to get the diversity needed for the final ensemble. Instead of first passing the data through the GNN at the beginning and THEN into the Squeezeformer, some models interleaved the layers GNN->Squeezeformer->GNN->Squeezeformer. This didnt provide any substantial gains over the already effective approach, but it provided diversity into the final ensemble. Other models omitted the edge_features in the GNN and learned them on the fly, and some were given explicit inter-mouse features as edge attributes in the graph.
- Augmented RelationNetwork classification head. Instead of the Pairwise classification head, one model family used a relation network inspired classification head. I got the idea/inspiration from a [paper](https://arxiv.org/abs/1706.01427) I've read in the past where the authors use a simple plug-in network for relational reasoning in visual models. The addition over the pairwise classification head is the sum of the interraction vectors that give off a "cage vibe". In the pairwise classification head we only look at Mouse A vs Mouse B without knowing whats happening in the cage overall. However in the RelationNetwork head we also look at the global context of the cage - if we see that there are mice sleeping or in low energy state, its less likely that there is attack going on between Mouse A and Mouse B. Again since the model was pretty decent already, it provided just subtle improvement but more importantly it increased the diversity.

#Training Strategy
The model was fed window chunks from the lab videos. Each window was 512 frames long. During training sampling of action-rich frames was important since the dataset was imbalanced and the labeled frames were << total frames. We detect all of the frames where there is an action and we store the indices as "active frames". During sampling we bias the dataloader to sample a window from these active frames and further offset the start of the window by a random amount so that we can introduce variety in the start/stop of the actions. The prob of sampling action vs non-action windows was tuned via bias hyperparameter which ranged from 0.5 to 0.8. However this is not enough to combat the imbalance. Even if we sample the action rich windows with 0.5 probability, there is still another imbalance which is the duration of action vs non-action frames in each window. The median duration of most actions is below 100 frames, whereas the window is 512 frames which leaves more than~400 non-action frames, that means on average we sample much more non-action frames than action frames even after we biased the dataloader to sample action rich windows 50% of the time. To combat this I used focal loss with per-action class weights which were calculated by the ratio of the length of each action versus the length of the sampling window. When an action spans only 12 frames within a 512-frame window, the model is penalized equally for misclassifying those 12 action frames as it is for the 500 non-action frames.

Traning was done with lr of 3e-4 and 1e-3, AdamW optimizer + OneCycle scheduler with 0.15pct warmup. Batch size of 32. Focal loss with gamma 2.0. FP16 was used during training and inference. Dropout ranging from 0.1 to 0.35 in different runs/model families.

#Post-Processing & Validation
There is very little post processing after the models predictions. After getting the probabilities for each frame, we just find continuous segments and classify that as an action segment. There is a minimum duration filter which removes flickers and also optional segment-merging logic which finds segments of the same action that are N frames apart and merges them (this was not used in the final submission as I didn't have time to tune it and didnt want to risk overfitting). The validation is done by sweeping over the full video in overlapping windows. The stride was half the window size (Window = 512 -> We shift 256 frames and run the model) and finally the final probabilities are averaged where the frames overlapped. Most of the benefit comes from ensembling different model families into the final solution. One model finishes the prediction in about 15minutes on one of the T4 GPUs. 23 models were used in the final submission. Out of those 23 models most are just 4-Fold CV average of the same model family.

##What worked
1. Processing the mice with the GNN and looking at each mice separately in an egocentric trajectory in the transformer was one of the most important improvements. Previously I tried CNN over the mice dimension and also combining all of the mice features before passing them into the squeezeformer but even though the model learned and scored decent (ensemble of 6 single fold models without GNN and mixed mice features into a single feature vector is around 16~place) it wasn't optimal. 
2. High dropout and diversity. Some models were trained with dropout of 0.35. The model still converged very well and maintained robustness which I think was very important in the transfer learning across labs and unseen videos. Another major point is diversity. One model family was trained with GNN at the beginning and then Squeezeformer. Another had the blocks of GNN and Squeezeformer interleaved so they update each other. The next family wasn't given the edge_features explicity and had to learn it on its own, and another version had the calculated inter-mouse features provided as explicit edge_features in the TransformerConv layers. One model family had Egocentric view of all the features, in which we calculate rotation matrices and rotate the world/other mice so that always the mice sees the world from its POV. This diversity and robustness from high-dropout enabled the 5th place private.

##What didnt work
- Deeper or wider models. This had very little impact on the score. I've tried different depths up to 12-16 layers and also width up to 512dim but it provided no substantial improvement. I think the "path" of the data and architecture was more important than the scale.
- Reverse Time ensemble. Well known idea to process the video forward + backward and average the probs. However for some reason I couldn't make it to work. Probably some bug in my implementation, reverse videos always underperformed and dragged down the ensemble.
- Using plain 1D CNN. The CV/LB gap was much larger than the squeezeformer so I dropped this approach.
- Adding rolling window / fourier features. The idea was to detect cyclic actions such as groom. However the overhead of calculating the fourier and rolling window features in my first implementations was too great so I dropped them from the feature generation.
- Imputing keypoints in missing frames - this lowered local CV
- Larger/Smaller window size. I've tried 128, 256, and 768 frames long window but made little difference.

##What I wrote down but didn't try
Since I joined a bit late I wrote down several ideas I want to try but didnt have time. Some of them are:
1. Action Class + Start/Stop boundary regression. Use approach similar to the [ActionFormer](https://arxiv.org/abs/2202.07925) paper, where besides predicting the action class of the current frame we also try to predict the distance of current frame to the start/stop of the ground truth segment. As far as I understood when talking about it with an LLM, I think its much better than just predicting the current frame. Since in the ActionFormer approach (as far as I understand) many frames inside the action segment can "vote" on where they think the start/stop frames are relative to them. So even if some of the frames at the start of the action segment are corrupt, a frame in the middle can vote where it think the action started, and we can use that to get a probability mass from all of the frames voting where they think the action started. I still don't fully understand it and need to spend more time, but seems like a reasonable idea.
2. Different temporal strides/frame rates (each 2nd frame etc.)
3. Keypoint imputer pre-trained on 90% of unlabeled dataset used to impute missing bodyparts. Use it to upgrade from 6-point to more detailed 14-point master skeleton.
4. Since labs don't change, I wanted to make per-lab adapter finetuned on specific labs.


Thanks for the opportunity and the learning experience. 

Training Code: https://github.com/GosUxD/MABe-5th

Kaggle Raw Inference Kernel: https://www.kaggle.com/code/dankrstev/ratgi

##Acknowledgements
1.[Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](https://arxiv.org/abs/2206.00888)

2.[A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427)

3.[ActionFormer: Localizing Moments of Actions with Transformers](https://arxiv.org/abs/2202.07925)
