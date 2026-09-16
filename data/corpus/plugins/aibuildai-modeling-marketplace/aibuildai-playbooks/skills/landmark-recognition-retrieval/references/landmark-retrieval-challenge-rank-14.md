# 14th place solution

Competition: landmark-retrieval-challenge
Rank: #14
Source: https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/58167

Hi, Everyone!

We’d like to share our solution. 

Our approach consists of 4 main steps: global CNN descriptor, nearest neighbor search, re-ranking and query expansion. 

1. As global descriptor we’ve finetuned ImageNet-pretrained [PyTorch ResNet50][1] model on dataset from [Landmark Recognition Challenge][2] with [GeM pooling layer][3] from [“Fine-tuning CNN Image Retrieval with No Human Annotation”][4] by Radenovic et.al. and [hard-in-batch triplet margin loss][5].
2. Fast nearest neighbor search done by [Faiss][6] library.
3. We have re-ranked top-100 images by performing classic image matching: detect keypoints -&gt; extract local patches -&gt; describe -&gt; match -&gt; [RANSAC][7] geometric verification.
4. The final stage is query expansion by diffusion, Iscen et.al [“Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations”][8] implemented in python here [https://github.com/ducha-aiki/manifold-diffusion][9] . 

For more details and links, please, take a look at [https://medium.com/@ducha.aiki/14th-place-solution-for-kaggle-google-landmark-retrieval-challenge-6d43e38eb513][10]


  [1]: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L13
  [2]: https://www.kaggle.com/c/landmark-recognition-challenge
  [3]: https://github.com/filipradenovic/cnnimageretrieval-pytorch
  [4]: https://arxiv.org/abs/1711.02512
  [5]: https://github.com/DagnyT/hardnet/blob/master/code/Losses.py#L87
  [6]: https://github.com/facebookresearch/faiss
  [7]: http://cmp.felk.cvut.cz/software/LO-RANSAC/Lebeda-2012-Fixing_LORANSAC-BMVC.pdf
  [8]: https://arxiv.org/pdf/1611.05113.pdf
  [9]: https://github.com/ducha-aiki/manifold-diffusion
  [10]: http://Hi,%20Everyone!%20We%E2%80%99d%20like%20to%20share%20our%20solution.%20Our%20approach%20consists%20of%204%20main%20steps:%20global%20CNN%20descriptor,%20nearest%20neighbor%20search,%20re-ranking%20and%20query%20expansion.%20%20%201.%20As%20global%20descriptor%20we%E2%80%99ve%20finetuned%20ImageNet-pretrained%20PyTorch%20ResNet50%20model%20on%20dataset%20from%20Landmark%20Recognition%20Challenge%20with%20GeM%20pooling%20layer%20from%20%E2%80%9CFine-tuning%20CNN%20Image%20Retrieval%20with%20No%20Human%20Annotation%E2%80%9D%20by%20Radenovic%20et.al.%20and%20hard-in-batch%20triplet%20margin%20loss.%202.%20Fast%20nearest%20neighbor%20search%20done%20by%20Faiss%20library.%203.%20We%20have%20re-ranked%20top-100%20images%20by%20performing%20classic%20image%20matching:%20detect%20keypoints%20-%3E%20extract%20local%20patches%20-%3E%20describe%20-%3E%20match%20-%3E%20RANSAC%20geometric%20verification.%204.%20The%20final%20stage%20is%20query%20expansion%20by%20diffusion,%20Iscen%20et.al%20%E2%80%9CEfficient%20Diffusion%20on%20Region%20Manifolds:%20Recovering%20Small%20Objects%20with%20Compact%20CNN%20Representations%E2%80%9D.%20%20%20For%20more%20details%20and%20links,%20please,%20take%20a%20look%20at%20https://medium.com/@ducha.aiki/14th-place-solution-for-kaggle-google-landmark-retrieval-challenge-6d43e38eb513
