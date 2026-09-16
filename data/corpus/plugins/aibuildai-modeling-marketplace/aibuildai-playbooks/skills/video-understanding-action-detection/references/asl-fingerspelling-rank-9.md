# 9th Place Solution

Competition: asl-fingerspelling
Rank: #9
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434871

The below is result of a team effort with @rafiko1 and @group16 and myself.

We want to thank to the hosts for putting together a great competition, it was really fun and we were glad to end up in the gold zone! We've already enjoyed reading some of the top teams solutions- congrats to all who participated.

## tl;dr

We used a similar architecture to @hoyso48's solution from the ASL-sign competition. Our model had six Conv1D/Transformer blocks, was trained with CTC loss over 500+ epochs, and utilized heavy augmentations.

This competition posed challenges beyond achieving model accuracy. We had to meet model constraints (5-hour submission limit and 40mb model size). We also needed to make sure all of our code converted nicely to tflite, which at times was frustrating. This required us to find a good balance between model size and inference time with minimal postprocessing.

## Data and Preprocessing:

We used the training data as the base dataset, but also trained using the supplemental and external (ChicagoWild/Plus) during fine tuning. Preprocessing was the same for all datasets:

- Standard scaling using the mean and standard deviation for each point.
- x, y, z coordinates for each point.
- Points used included: RHAND, LHAND, LIP, POSE, REYE, and LEYE.
- We tried different settings for the max frame length since it had the most impact on our preprocessing and ened up using 356.
- Downsampling non-hand frames: To handle samples that has more than 356 frames, we first removed frames with missing hands at even intervals. This improved performance when we added it to our inference- so eventually we added it to our training pipeline.
- Resize: Examples exceeding our max frame length post-downsampling were resized.

## Augmentations:
Heavy augmentations helped us to train long without the risk of overfitting.

- Mirror/Flip
- Random resampling
- Random rotation
- Random spatial and temporal masking
- Temporal cropping
- Minimal Random noise
- Random scaling

## Training
Given that training each model took a very long time to train (days running on colab TPUs), we resumed from the best checkpoints to experiment with different augmentations, data ratios, and learning rate schedules.  It’s hard to exactly retrace the steps we took to get to our final model but the general idea was:
- Base training without augmentations at a consistent learning rate until convergence (~150 epochs).
- Introducing heavy augmentations and continuing until the validation score plateaued (200-400+ epochs).

Example of how our validation score improved from different experiments over the final week of the competition:


Even up until the end of the competition we were able to get small improvements in our score from more training.

## Model Architecture:
Initially, I tried an efficientnet approach similar to the 2nd place in the signs competition. Later, I pivoted to the 1st place architecture similar to public notebooks. After merging teams we found both of us had similar architectures but @rafiko1 made some modifications that improved it's performance on this task:
- Switched from 3 to 2 Conv1D per Conv/Transformer block.
- Increated to 6 blocks (resulting in a 10.5M parameter model).
- Parameters: 384 max frames, 320 dimensions, 8 heads per transformer. 
- Did not use a final MaxPooling layer.

We also had experiments that used DebertaV2 instead of the transformer block, But the final model didn't use them.

## Increasing model size without retraining:
Towards the competition's end, realizing we didn't have a great way to ensemble models, we wanted to get the biggest model possible without having to retrain. We found that instead of retraining from scratch we could simply expand the number of Conv1D/Transformer blocks we had - and copy the weights over from our previous best model, repeating the final layer weights. This allowed the model to converge very fast and squeeze out some additional accuracy while saying below the 5 hour inference limit.

## Attempts not in our final solution:
- **Cutmix based on pseudo labels:** The idea was to use model predictions to pinpoint frames displaying each letter in videos, then applying cutmix at the frame level during training. It functioned initially but became required us to recreate the tfrecords each time we modfiied preprocessing or model archetecture, so it wasn't used in our final solution. When it was used we only applied cutmix across similar phrase types (phone numbers mixed only with phone numbers, etc) but it was interesting to read the top team found it worked best mixing only between the same signer.
- **Ensembling techniques:** Couldn't find a good solution for this with CTC loss in tflite.
- **Beam search**
- **Efficientnet/Transformer**
- **DebertaV2**

## Postprocessing / Ensemble:
**Fill word**: As others have noted, there were a significant number of examples in the dataset with very few frames. These had low or no signal in them related to the target. We found it was better to just replace them with a fill word ('+w1- ea-or') We also found that it worked best to apply these to examples that had less than 30 frames AND a prediction phrase that was <= 6 characters.
**Model weight averaging** Instead of using the final epoch of each training run, averaging the weights from the final few epochs really helped the CV and LB score. This became less powerful later in the competition when our models were stronger.

## Strong Validation/LB Correlation:



We tracked each submission's execution time and CV/LB correlation. This competition had a really strong correlation between our validation set (one parquet file) and the LB. Our best private LB score was 0.788 but we didn't select it because it did not perform as well on the public LB.
