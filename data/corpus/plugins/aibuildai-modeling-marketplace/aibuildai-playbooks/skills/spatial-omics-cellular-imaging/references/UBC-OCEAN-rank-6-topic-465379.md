# 3th on Public and 6th on Private, A Very Simple Solution: Big Pretrained-Model is All You Need!

Competition: UBC-OCEAN
Rank: #6
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465379

Hello, everyone, I'm here to share our solution! We only used the simplest pre-trained weights from iBOT-ViT-Base. Thanks a lot for this excellent work! Here is the project: [iBOT-ViT](https://github.com/owkin/HistoSSLscaling). Specifically, our algorithm consists of five steps:
1、Tiling the WSL image ( or TMA image), we random select 1000 patches (tiles) per image, if not enough, copy them;
2、Using the pre-trained model to extract features, dimension: 1000x768 per image;
3、Training a MIL (Multi-Instance Learning) model, we use the recommended chowder model [Chowder](https://arxiv.org/pdf/1802.02212.pdf) mentioned in the above iBOT-ViT method;
4、Model ensemble, (we use 7 different trained chowder models) and use the average entropy ( E=-sum(p*logp) ) for detecting "other";
5、 Adjust the threshold of "other";

Some tips:
1、We found that patch selection has an important impact on performance. Still, we just use the simplest random selection, Recent work: [PathDINO](https://rhazeslab.github.io/PathDino-Page/) proposed a fast patch selection method, but we didn't get any improvements.
2、Deep ensemble and uncertainty estimation through entropy help us from 0.59 to 0.65 on public data, but it doesn't seem to have earned me a bonus😔
3、Here is our source code: [UBC_Challenge](https://github.com/yangzhou321/UBC_Challenge/blob/main/ubc_ours.ipynb)

Any idea or discussion is highly welcomed!
