# How to take the ArcFace Baseline from private lb: 0.470 (public: 0.522) to 0.804 (public: 0.834)

Competition: happy-whale-and-dolphin
Rank: #29
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/319787

I thought others might be interested to learn the steps I took to take the [public solution by ks](https://www.kaggle.com/code/ks2019/happywhale-arcface-baseline-tpu) from 0.470 on the private leaderboard to 0.804. I'm sure others were able to push it much further!

**Link to kernel [here](https://www.kaggle.com/code/lextoumbourou/happywhale-tpu-baseline-to-0-804-elasticface)**

I had to train it in GCP due to the TPU wait time, but it should train just fine in Kaggle.

## Summary of changes

- Image size: 864x846.
- Pretrained model: EfficientNet b5 with noisy-student.
- Pooling: concat pooling (from fastai)

    ```
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    pretrained_out = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    ```

- Head:
  - Dual-head model: Species classification output and metric.
  - Add [Multi-Sample Dropout](https://arxiv.org/abs/1905.09788) (idea thanks to @dhakshiin) before metric.

- Metric function: [Elastic Margin Loss](https://arxiv.org/abs/2109.09416) (w=0.3, std=0.025, s=30) (minor improvement over ArcFace).
- Embed size: 1024.
- Data:
    - Clean species label thanks to [this post](https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/305574) by @kwentar
    - Train on entire dataset (no validation) with 12157 extra pseudo labels.
    - Use pseudo labelling-algorithm by @dhakshiin: Use multiple models, find top-1 examples that the majority agree on, ensuring 0.2 separations between the confidence of 1st class and 2nd class.
    - Select between 8x crops randomly selected at training time: full-body crops, detic crops, original yolov5 crops, 3x TokenCut crops and full-sized images ([See my TFRecords notebook here](https://www.kaggle.com/lextoumbourou/happywhale-generate-tfrecords-with-pseudo))
- Augmentations: `random_flip_left_right`, `random_hue`, `random_saturation`, `random_contrast`, `random_brightness` and `random_rgb_to_gray` (parameters tuned slightly from original kernel)
- Optimiser: Adam (unchanged from original)
- Learning Rate: exponential decay with 4 epochs warm-up (unchanged from original)
- Epochs: 30

- Inference:
  - Generated embeddings for each crop with a horizontal flip.
  - Take a weighted mean of all crop embeddings to create final embeddings.
  - Inference: standard KNN inference (use KNN=150 when added to the final ensemble)
