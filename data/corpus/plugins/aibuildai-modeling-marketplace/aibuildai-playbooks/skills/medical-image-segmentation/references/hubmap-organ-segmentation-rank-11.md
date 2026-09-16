# 11th place solution

Competition: hubmap-organ-segmentation
Rank: #11
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/354701

Thanks to the organizers and congrats to all the winners!

### Overview
In this competition, since CV based on the HPA labelled data is not reliable for the private LB score improvement, I made as many submissions as possible to confirm the effectiveness of my experiments by the Hubmap only LB score. Finally, the following three points were the key item of my solution.
- Use GTEx data
- Ensemble and TTA as many as possible (12 models x 20 TTAs)
- Post-process

### Models
Total 12 models were used for ensemble.
- EfficientNet B4/B5/B6/B7 - Unet
- EfficientNetV2 S/M/L - Unet
- ConvNeXt T/S/B - DeeplabV3+
- Swin B/L - Unet

### Training
- Data were resized to x0.25 scale (for GTEx data, after rescaled to 0.4 um pixel size) and random cropped to 512x512 size
- 120 epochs
- Loss: BCE loss + Dice loss
- Optimizer: Adam
- LR schedule: Cosine decay with warmup
- Augmentation stronger than usual competitons was used
- ImageNet pretrained backbone
- Trained on TPU of Kaggle kernel or Colab Pro
- All models were trained with TensorFlow
- All data were used for training (Not using kFold-CV)

### External data
I downloaded external data from GTEx Portal, and made the pseudo labels of them. Since the GTEx .svs images have too large resolution, they were cropped as tile before making the pseudo labels. If the pseudo label's mask area was too small or there was no mask, this tile was excluded. The pseudo labels were manually checked and modified or removed if it was obviously wrong. These data were used for training by mixing with the original labeled data. Finally, total 650 tiles from 40 samples were used for training. It improved my Hubmap only LB score ~0.02.

### TTA
Although I could use maximum 8 combination patterns, I reduced them to four because of the 9 hours run-time limitation. The scales were chosen based on x0.25 training scale.
- 5 scales x 4 patterns = 20 TTAs per model
- 5 scales: 0.19, 0.21, 0.23, 0.25, 0.28
- 4 patterns: original, lr-flip, ud-flip, rot90

### Post-process
Although there is no clear evidence, especially for "spleen" and "lung", the boundary of the masks is fuzzy, so I tried to enlarge or shrink the masks and confirmed the effect in LB score. As a result, shrinking spleen and lung prediction masks improved my LB score. I did not include post-processing in one of the final submissions because I thought there was a possibility of overfitting to the public data. However, the resulting private LB score was better if I include the post-processing. The post-processing was impremented as following code.

```
SHRINK = {'spleen': 10, 'lung': 10}

def post_proc(preds, organ):
    # preds: predicted mask after binarization
    
    if organ in SHRINK.keys():
        contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        preds = preds.astype(np.int8)
        preds_cont = np.zeros_like(preds)
        preds_cont = cv2.drawContours(preds_cont, contours, -1, 1, SHRINK[organ]*2)
        preds = (preds - preds_cont).clip(0, 1).astype(np.uint8)
    return preds
```
