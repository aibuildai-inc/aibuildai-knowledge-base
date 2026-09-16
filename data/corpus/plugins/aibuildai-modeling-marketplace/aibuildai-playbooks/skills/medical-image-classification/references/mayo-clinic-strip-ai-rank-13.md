# 13th Place Solution

Competition: mayo-clinic-strip-ai
Rank: #13
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/358203

Hi there, this is a pretty big shock. I put together a solid initial baseline early on in this competition and got busy with other things. I open-sourced this approach [**inference here**](https://www.kaggle.com/code/dschettler8845/13th-tf-mcsai-no-tiling-model-inference/notebook?scriptVersionId=100779671) and [**training here**](https://www.kaggle.com/code/dschettler8845/mcsai-no-tiling-model-tpu-tf).

This notebook was created almost 3 months ago (within the first few weeks ... maybe the first week ... of the competition). 

<br>

**The basics are as follows:**
* Pretrained EfficientNetB6 fine-tuned w/ TPU on WSI
* Model Head --> Dropout @ 0.5 > 2 Node CC w/ Class Weights
* Learning Rate Ramps Up and Decays (12 Epochs)
* Whole Slides Used (no tiling)
* Images are resized to (512,512,3)
* Use pyvips for WSI processing
* Train Augmentation
```python
def augment_batch(img_batch):
    img_batch = tf.image.random_brightness(img_batch, 0.2)
    img_batch = tf.image.random_contrast(img_batch, 0.5, 2.0)
    img_batch = tf.image.random_saturation(img_batch, 0.75, 1.25)
    img_batch = tf.image.random_hue(img_batch, 0.1)
    return img_batch
```

If anyone has any questions please let me know! 
Also, thanks to Kaggle for constantly hosting interesting Biology competitions.

<br>

ps. This levelled me up to a Competitions Master. Woo!
