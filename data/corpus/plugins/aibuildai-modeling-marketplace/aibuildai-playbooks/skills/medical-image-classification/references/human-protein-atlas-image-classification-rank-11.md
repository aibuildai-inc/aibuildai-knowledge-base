# 11th place solution(custom f1 loss)

Competition: human-protein-atlas-image-classification
Rank: #11
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77289

Dream comes true! Here I briefly describe the approach I used.  [Here is the details of our team's solution.][1]

I use custom f1 loss to train model, and ensemble with my teammates' models which use bce loss.

Single se-resnext50(5 folds ensemble) with f1 loss gets 0.562 on private LB(which is higher than our final ensemble score, but strangely, it's public LB score is low, only 0.606, so we didn't use it for final submission). Resnet18(5 folds ensemble) gets 0.612 on public LB, and 0.536 on private LB. All these scores come from 512*512 RGBY input, incluing HPA v18 external data.

When using f1 loss, I oversample the rare classes. I apply data augmentation based on class frequency(images that have rare class get more chances to be augmented).

I noticed that when a batch of images lacks some classes(which is common for rare classes), there is no gradient backpropagates to those classes. So I add bce to those classes in that case. In addition, I clamp the network output at 0.01 to force network focus on the hard samples.

I use thresholds 0.205 for all classes.

Inspired by [Tilli's idea][2], we select about 300 pairs of similar images in test set, and replace poor quality images with good quality ones. This approach yields about 0.002 score improvement.

    def f1_loss(predict, target):
        loss = 0
        lack_cls = target.sum(dim=0) == 0
        if lack_cls.any():
            loss += F.binary_cross_entropy_with_logits(
                predict[:, lack_cls], target[:, lack_cls])
        predict = torch.sigmoid(predict)
        predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
        tp = predict * target
        tp = tp.sum(dim=0)
        precision = tp / (predict.sum(dim=0) + 1e-8)
        recall = tp / (target.sum(dim=0) + 1e-8)
        f1 = 2 * (precision * recall / (precision + recall + 1e-8))
        return 1 - f1.mean() + loss


  [1]: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77282
  [2]: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/74068
