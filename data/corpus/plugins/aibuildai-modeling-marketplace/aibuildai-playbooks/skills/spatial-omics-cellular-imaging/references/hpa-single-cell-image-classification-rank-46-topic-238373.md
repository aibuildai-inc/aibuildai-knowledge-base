# 46th place solution - Simple Image Level Multilabel Classifier

Competition: hpa-single-cell-image-classification
Rank: #46
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238373

Congrats to all winners! Thanks to the organizers of this competition.

And specially thanks to @rdizzl3, @phalanx, @alexanderriedel, @thedrcat, @linshokaku, @its7171, @dschettler8845, @samusram for sharing knowledge and datasets. I have learned a lot from you.

## Summary

"Simple Image Level Multilabel Classifier"

The label was determined by applying a classifier to the single cell mask obtained by HPA-Cell-Segmentation.

Classifier was trained using the full dataset.

## Tools

- Colab Pro, GCE, Tesla V100 16GB single GPU
- GCS
- Pytorch Lightning
- Neptune
- Kaggle API

## Dataset

I used both the Competitions default dataset and the extra dataset.

[HPA 512 PNG Dataset](https://www.kaggle.com/phalanx/hpa-512512) by [@phalanx](https://www.kaggle.com/phalanx)

[HPA 768 PNG Dataset](https://www.kaggle.com/phalanx/hpa-768768) by [@phalanx](https://www.kaggle.com/phalanx)

[HPA 1024 PNG Dataset](https://www.kaggle.com/sunghyunjun/hpa-1024-png-dataset)

[HPA Public Data 768x768 "rare classes" dataset](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/223822) by [@Alexander Riedel](https://www.kaggle.com/alexanderriedel)

The extra dataset was downloaded by referring to the public note.
Images saved to 768px png. The size is approximately 200 GB.

[HPA public data download and HPACellSeg](https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg)

## Validation

MultilabelStratifiedKFold, 5-fold split was used.

The performance of Multilabel Classifier was verified with Macro-F1, Micro-F1 Score.

[iterative-stratification](https://github.com/trent-b/iterative-stratification)

## Model training

3-channel RGB images
The image size is 1024px, and trained with the following dataset.

- 1024px Competition default dataset + 768px rare classes dataset(1024 resized)
- 1024px Competition default dataset + 768px extra dataset(1024 resized)
- AdamW
- CosineAnnealingLR
- epochs = 5 for full, 10 for rare
bce, focal loss was used.

|model|dataset|folds|loss|batch_size|init_lr|weight_decay|macro F1|micro F1|public LB|private LB|
|---|---|---|---|---|---|---|---|---|---|---|
|efficientnet_b0|full|2 of 5|bce|16|6.0e-4|1.0e-5|0.7663|0.8171|0.454|0.429|
|efficientnet_b0|rare classes|single|bce|16|6.0e-4|1.0e-5|0.8154|0.8368|0.394|0.360|
|seresnext26d_32x4d|full|single|alpha=0.75, gamma=0.0|14|6.5e-5|1.0e-5|0.7317|0.7956|0.381|0.335|
|**final ensemble**|||||||||**0.471**|**0.433**|

## Segmentation

HPA-Cell-Segmentation was used, and the speed was improved by referring to @linshokaku 's notebook.

The input image was resized by 1/4, and the CellSegmentator scale_factor=1.0.

The related values of the label_cell function have been adjusted to 1/4.

[HPA-Cell-Segmentation](https://github.com/CellProfiling/HPA-Cell-Segmentation)

[Faster HPA Cell Segmentation](https://www.kaggle.com/linshokaku/faster-hpa-cell-segmentation)
by @linshokaku

## Augmentation

```python
A.Compose(
    [
        A.Resize(height=resize_height, width=resize_width),
        A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
        A.PadIfNeeded(
            min_height=resize_height,
            min_width=resize_width,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0,
        ),
        A.RandomCrop(height=resize_height, width=resize_width, p=1.0),
        A.RandomBrightnessContrast(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ]
)
```

## TTA 4x

HorizontalFlip, VerticalFlip, Resize 0.8, Resize 1.2

## What did not work

- Label Smoothing

- pos/neg balanced weighted loss

    X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers.
(Dec. 2017). "ChestX-ray8: Hospital-scale chest X-ray database and
benchmarks on weakly-supervised classification and localization of common thorax diseases.", (p. 5) [https://arxiv.org/abs/1705.02315](https://arxiv.org/abs/1705.02315)

## Using GCS and errors

The dataset size of this competition is really huge. I had some difficult to download extra public data. It took a lot of time. Multiprocessing was not helped because more than two files couldn't be downloaded at the same time.

I use colab-pro.I usually downloaded dataset to colab VM for convinience. But to train huge extra dataset I loaded dataset directly from my GCS bucket.

I refer the article [Training Faster With Large Datasets using Scale and PyTorch](https://medium.com/pytorch/training-faster-with-large-datasets-using-scale-and-pytorch-946dfe774d8c) And I didn't implement Asynchronous dataload. In my case, multiprocessing of torch.utils.data.DataLoader is enough for latency hiding.

But training from GCS had got some rare errors. (504 GatewayTimeout, 104 Connection reset by peer)

I don't know exact reason but it seems relate belows.

- opencv multithreading deadlock with pytorch DataLoader (num_workers>0, pin_memory=True)
[https://stackoverflow.com/questions/54013846/pytorch-dataloader-stucked-if-using-opencv-resize-method](https://stackoverflow.com/questions/54013846/pytorch-dataloader-stucked-if-using-opencv-resize-method)
[https://github.com/pytorch/pytorch/issues/1355#issuecomment-675018985](https://github.com/pytorch/pytorch/issues/1355#issuecomment-675018985)
solution: cv2.setNumThreads(0)

- CPU memory leaks of copy on write
[https://github.com/pytorch/pytorch/issues/13246#issuecomment-737442812](https://github.com/pytorch/pytorch/issues/13246#issuecomment-737442812)
solution
[https://gist.github.com/vadimkantorov/86c3a46bf25bed3ad45d043ae86fff57](https://gist.github.com/vadimkantorov/86c3a46bf25bed3ad45d043ae86fff57)

## Source Code
Source code is available at [https://github.com/sunghyunjun/kaggle-hpa](https://github.com/sunghyunjun/kaggle-hpa)

Submission notebook is [HPA faster final ensemble w/o rot exp 2](https://www.kaggle.com/sunghyunjun/hpa-faster-final-ensemble-w-o-rot-exp-2)
