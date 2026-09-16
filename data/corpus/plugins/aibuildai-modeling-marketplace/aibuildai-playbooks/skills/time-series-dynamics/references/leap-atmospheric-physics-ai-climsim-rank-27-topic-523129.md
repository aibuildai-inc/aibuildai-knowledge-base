# 27th Place Solution

Competition: leap-atmospheric-physics-ai-climsim
Rank: #27
Source: https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523129

Thank you for hosting such an interesting competition. We are deeply grateful to the hosts and the Kaggle staff. I would also like to thank , @sae20gorilla , @hidebu, @takatoyoshikawa , and @khiroki for teaming up with me. Even though we formed the team just a week before the close, we had many discussions that were very educational, and it was enjoyable to see our scores and public lb rank improve each time.

# 1. Summary

Our best private solution is an ensemble of 8 models.

After forming the team, we started the ensemble using external data for validation that none of us had used for training. Since the relationship between CV and the public LB was favorable in this competition, we calculated the coefficients using the Nelder-Mead method for each model to significantly improve the CV (sub1). Additionally, anticipating the presence of outliers, we used the Nelder-Mead method to calculate coefficients for each column (sub2), but as expected, the private LB for this was not as good. Below, we will describe each part for sub1 in more detail.

| model | part     | model                  | loss      | use train volume | Comment              | cv       | publicScore | privateScore | weights   |
|-------|----------|------------------------|-----------|------------------|----------------------|----------|-------------|--------------|-----------|
| 1     | tenten   | 1dConv-Transformer-GRU | Huberloss | 73.5M            |                      | 0.77776  | 0.77697     | 0.77157      | 0.49187   |
| 2     | tenten   | 1dConv-Transformer-GRU | Huberloss | 35M              |                      | 0.77257  | 0.77292     | 0.76602      | 0.09789   |
| 3     | chumajin | squeeze former         | Huberloss | 16M              | dim 192 + retraining | 0.76961  | 0.77050     | 0.76239      | 0.32106   |
| 4     | sae      | 1dConv - GRU           | Huberloss | 16M              |                      | 0.76787  | 0.76890     | 0.76192      | 0.01156   |
| 5     | tenten   | 1dUnet                 | Huberloss | 73.5M            |                      | 0.76723  | 0.76611     | 0.76139      | 0.07264   |
| 6     | sae      | 1dConv - LSTM          | Huberloss | 16M              |                      | 0.76618  | 0.76739     | 0.76001      | 0.18994   |
| 7     | chumajin | squeeze former         | Huberloss | 16M              | dim 128              | 0.76121  | 0.76278     | 0.75497      | -0.24416  |
| 8     | tenten   | 1dConv-Transformer-GRU | MSEloss   | 16M              |                      | 0.76108  | 0.75965     | 0.75654      | 0.07312   |



**final ensemble cv : 0.78116375, public lb : 0.78052, private lb : 0.77477**


# 2. tenten's part

- overview



- Input
    - StandardScaler
    - Scalars are expanded to match the series length (60).
- Model
    - 1dUnet
        - down_channels: [128, 256, 512]
    - 1dConv-Transformer-GRU
- Training
    - Loss function: HuberLoss(delta=1.0)
    - Optimizer: AdamW
    - Scheduler: Warmup Cosine Annealing
    - Data: All low-resolution data
- Postprocess
    - I used torch.min to limit all constant targets except cam_out_FLWDS to non-negative values (I apply this during training before calculating the loss)
    - ptend_q0002_x =  -state_q0002_x * 1200 (12≤x≤27)
- What didn’t work well
    - Auxiliary loss (latlon, grid id)
    - Increase the sequence length by upsampling
    - Robust scaler
    - Tuning the loss weight 

# 3. saeNeko's part



- Models:


    ・ 1D CNN + GRU


    ・ 1D CNN + LSTM

- Preprocessing:
 
    Applied StandardScaler to both features and targets.

- Training Data:
    ・Used Kaggle’s train.csv.
    ・6M additional data (Thanks to @khirokifor extracting data from Hugging Face!)

- Training:
 
    Trained separately for series targets and scalar targets.

- Loss Function:
 
    RMSE

- Postprocessing:
 
    For ptendq0002_x(1<=x<=27), used a linear model.
    ptend_q0002_x = state_q0002_x * coef + intercept_

# 4. chumajin's part

- Preprocess

    ・ The diff of the sequential part and StandardScaler
    ・ The scalar part was repeated after applying the StandardScaler.

- Training Data:
    
    ・Used Kaggle’s train.csv.
    ・6M additional data


- Architecture

    SqueezeFormer from the 2nd place solution of the Stanford Ribonanza RNA Folding competition, (https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460316) 

- Mask of loss

    I did not use the loss for the unpredictable target.

- Re-training

    Additional 10 epochs fine-tuning after 20 epochs.
    cv 0.767055 → 0.768034112 (I call this chumajin model1.)

- Insights from my teammates
    ・ Huberloss(delta=1)
    cv improved by about 0.01. 

    ・ Re-training(re-finetune) by separating the loss
    By training again with limiting the loss, and creating three additional models from the above chumajin model1, I averaged their predictions which was inferred from the limited loss parts of the model with chumajin model1

     1.   model 2 : limited loss of ptend_q0001, q0002, and q0003
     2.  model 3 : limited loss of ptend_t, ptend_u, and ptend_v
     3.  model 4 : limited loss of scalar values 

By doing this, I was able to improve my model's CV from 0.7680 to 0.7696. As a result, re-training led to a CV improvement of +0.0026.

- Postprocess

    ・ Applying tenten's postprocess of torch.min for scaler part.
    ・ Applying sae's postprocess, the CV improved by about 0.002.

- Regrets and Acknowledgements

    With tenten's insights, the improved CV SqueezeFormer was created about a day and a half before the deadline. It took approximately 24 hours to train on 16M train data with an A100, so I couldn't train on all the 73.5M low-resolution data...But I learned a lot expecially from team up. Thank you for team mate!!
