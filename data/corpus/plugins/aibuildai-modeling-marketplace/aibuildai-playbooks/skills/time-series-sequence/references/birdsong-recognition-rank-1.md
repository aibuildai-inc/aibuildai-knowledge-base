# 1st Place Solution

Competition: birdsong-recognition
Rank: #1
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183208

Most of my solution was based on the baseline SED model provided by @hidehisaarai1213 . Without his kernel I wouldn't have achieved the result I did. So I am really grateful to him. Thanks for sharing a lot during the competition, I learnt a lot. 

## Data Augmentation


No external data.

- Pink noise
- Gaussian noise
- Gaussian SNR
- Gain (Volume Adjustment)

## Models

I noticed that the default SED model had over 80 million parameters so I switched all my models to use a pretrained densenet121 model as the cnn feature extractor and reduced the attention block size to 1024. Since it was much smaller and wouldn't overfit as much as we only had around 100 files for each audio class. I mainly tried densenet as previous top solutions to audio competitions used a densenet like architecture. I also replaced the clamp on the attention with tanh as mentioned in the [comments on the SED notebook](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection/comments#962915)

- 4 fold models without mixup
- 4 fold models with mixup
- 5 fold models without mixup

## Training

- Cosine Annealing Scheduler with warmup 
- batch size of 28
- Mixup (on 4 of the final models)
- 50 epochs for non-mixup models and 100 epochs for mixup models
- AdamW with weight_decay 0.01
- SpecAugmentation enabled
- 30 second audio clips during training and evaluating on 2 30 second clips per audio.

### Loss Function

My loss function looked something like the below. I wanted to experiment with different parameters but in the end I mainly used the default values, which was just BCELoss. I used a different loss function for 2 of the non-mixup models and it was based on randomly removing the primary label predictions from the loss function, to try increase the secondary_label predictions but I gave up on the approach for the rest of the models since I was running out of time and resources.

```python
class SedScaledPosNegFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha_1=1.0, alpha_0=1.0, secondary_factor=1.0):
        super().__init__()

        self.loss_fn = nn.BCELoss(reduction='none')
        self.secondary_factor = secondary_factor
        self.gamma = gamma
        self.alpha_1 = alpha_1
        self.alpha_0 = alpha_0
        self.loss_keys = ["bce_loss", "F_loss", "FScaled_loss", "F_loss_0", "F_loss_1"]

    def forward(self, y_pred, y_target):
        y_true = y_target["all_labels"]
        y_sec_true = y_target["secondary_labels"]
        bs, s, o = y_true.shape

        # Sigmoid has already been applied in the model
        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)
        y_true = y_true.reshape(bs*s,o)
        y_sec_true = y_sec_true.reshape(bs*s,o)
        
        with torch.no_grad():
            y_all_ones_mask = torch.ones_like(y_true, requires_grad=False)
            y_all_zeros_mask = torch.zeros_like(y_true, requires_grad=False)
            y_all_mask = torch.where(y_true > 0.0, y_all_ones_mask, y_all_zeros_mask)
            y_ones_mask = torch.ones_like(y_sec_true, requires_grad=False)
            y_zeros_mask = torch.ones_like(y_sec_true, requires_grad=False) *self.secondary_factor
            y_secondary_mask = torch.where(y_sec_true > 0.0, y_zeros_mask, y_ones_mask)
        bce_loss = self.loss_fn(y_pred, y_true)
        pt = torch.exp(-bce_loss)
        F_loss_0 = (self.alpha_0*(1-y_all_mask)) * (1-pt)**self.gamma * bce_loss
        F_loss_1 = (self.alpha_1*y_all_mask) * (1-pt)**self.gamma * bce_loss

        F_loss = F_loss_0 + F_loss_1

        FScaled_loss = y_secondary_mask*F_loss
        FScaled_loss = FScaled_loss.mean()

        return FScaled_loss, {"bce_loss": bce_loss.mean(), "F_loss_1": F_loss_1.mean(), "F_loss_0": F_loss_0.mean(), "F_loss": F_loss.mean(), "FScaled_loss": FScaled_loss }
````

## Thresholds

I used a threshold of 0.3 on the `framewise_output` and 0.3 on the `clipwise_output` to reduce the impact of false positives. So if the 30 second clip contained a bird according to the clipwise prediction and the 5 second interval based on framewise prediction also said it had the same bird then it would be a valid prediction. During inference I also applied 10 TTA by just adding the same audio sample 10 times in the batch and enabling Spec Augmentation.

## CV vs LB 

My CV didn't match the public LB at all, so I mainly relied on the LB for feedback. During training I monitored the f1 score of the clipwise prediction, framewise prediction and the loss associated with classes existing in the audio (i.e the value of `F_loss_1` in the above loss function). When loss value of `F_loss_1` increased it generally meant that it would do worse on the LB even though the f1 score was increasing too. 

## Ensemble

I used voting to ensemble the models. My voting selection was based on LB score so in total I had 13 models with 4 votes to consider if the bird existed or not. 
On the public LB, the 3 votes approach scored 0.617 which was slightly better than 4 votes model of 0.616, but I didn't select the 3 votes approach as I thought it was too risky which turnout out to be the correct choice as the 4 votes approach achieved 0.002 better than the 3 votes model on the private LB. My second selected submission was an an ensemble of the nomix up models (9 models) with 3 votes which scored 0.676 private, 0.613 public LB.

My individual models were pretty bad on the public LB. I didn't check some of them individually as I was running out of submissions but they generally ranged between 0.585-0.605 on the Public LB.  I mainly relied on my ensemble technique to get the score boost.

Thanks to the hosts and Kaggle for this interesting competition. 

**Inference Notebook**: https://www.kaggle.com/taggatle/cornell-birdcall-identification-1st-place-solution 
**Training Code**: https://github.com/ryanwongsa/kaggle-birdsong-recognition
**Example on How to train the model on Kaggle Kernels**: https://www.kaggle.com/taggatle/example-training-notebook
