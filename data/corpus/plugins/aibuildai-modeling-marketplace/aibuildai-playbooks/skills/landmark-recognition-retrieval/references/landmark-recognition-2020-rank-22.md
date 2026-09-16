# [22nd Place + Code] Margin Scheduling – First time on Kaggle :D

Competition: landmark-recognition-2020
Rank: #22
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187731

First, **congratulations to the top teams!!!** 🎉🎉🎉🎉🎉🎉🎉 What they achieved is incredible! 

This year's [Landmark Retrieval][glret] and [Recognition][glrec] challenges are the first ML competitions that I've ever participated in. Haven't touched anything ML-related for a year, so this was refreshing.

Unfortunately, I joined the Retrieval competition late (8 days before the end) and got only around 2 weeks to participate in this one part-time. I've learned a lot during this competition. If there will be a landmark challenge next year, I'll give it a better fight!

Also, I would like to thank @keetar for being an inspiration – what he achieved with just Colab Pro showed me that you don't need fancy hardware to compete, even in such a hard competition with a ton of data. His [excellent writeup][keetar-glr] was the sole reason why I decided to participate in the Recognition track.

## Retrieval Track (39th, Silver medal)

- Nothing special, just a weighted ensemble of `0.271` and `0.277` models and got silver medal lol lol lol 😆😆😆 I also tried some filtering using Places365 VGG model, but it doesn't work out.

----------------------------------------------------------------

## Recognition Track (22nd, Silver medal) – margin scheduling

I realized that I've been spamming the Discussion section a lot, so I'll keep this short:

- I trained a [ResNet152V2][resnet152v2] with [GeM][gem] (p=3) + [ArcFace][arcface] and a [EffNetB6][effnetb6] with GAP + [ArcFace][arcface] global descriptors on Cleaned GLDv2. Didn't have time to train to full convergence ☹️☹️☹️ 

- For [ArcFace][arcface], I started with a small margin (i.e. `1e-3`), and gradually increased it at the end of every period of Cosine LR, up to `0.31`. It was harder for me to train with a fixed margin.

- Training hardware: Colab Pro only!

- I simply used a publicly available object detector [FastRCNN+InceptionResNetV2][inceptresnet] trained on [Open Images V4][oims] for non-landmark removal. This gave a boost of around **+0.005** on pub. LB.

- Because I started late (~2 weeks before the end) and didn't had enough time to experiment around, I used [Data Echoing][dataecho] technique to speed up training **by ~20%**.

- Standard re-ranking: Retrieval step (using KNN) + Non-Landmark Removal (as described above) + Local Descriptors extraction (using baseline DELG model) + RANSAC.

- During inference, I used index embeddings only! I did not have time to generate embeddings for training samples.

What I didn't have time to finish:

- Local descriptors. I only implemented the local descriptors [36 hours before the end][panictraining].
- Multi-Gradient Descent ([MGDA-UB][mgda-ub]). I figured this could allow the local descriptor to train alongside with the global ones without `stop_gradient`.
- Pre-computed embeddings. Ironically, [I was the first who have asked if this is legal][precompdisc].

My training code is [available on Github](https://github.com/hav4ik/google-landmarks-2020). I've also published my [final submission](https://www.kaggle.com/chankhavu/glob-effb6-e48-rn152-e40-det-frcnn-ss-orig).

----------------------------------------------------------------

## Fun fact: you could've gotten a Silver medal using just baselines!

- Replacing the global descriptor from the [Public Recognition Baseline Example][recbase] with the model from [Public Retrieval Baseline][b277] will get you `0.4872/0.5081`, or a **Bronze** medal.

- Adding a simple non-landmark removal with object detectors will boost the above described baseline-based solution to a **Silver** medal.




[recbase]: https://www.kaggle.com/paulorzp/baseline-landmark-recognition-lb-0-48
[b277]: https://www.kaggle.com/nvnnghia/main-0806
[inceptresnet]: https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
[resnet152v2]: https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152V2
[effnetb6]: https://github.com/qubvel/efficientnet
[delg]: https://arxiv.org/pdf/2001.05027.pdf
[arcface]: https://arxiv.org/abs/1801.07698
[gem]: https://arxiv.org/abs/1711.02512
[dataecho]: https://arxiv.org/abs/1907.05550
[glret]: https://www.kaggle.com/c/landmark-retrieval-2020/
[glrec]: https://www.kaggle.com/c/landmark-recognition-2020/
[spoint]: https://arxiv.org/abs/1712.07629
[sglue]: https://arxiv.org/abs/1911.11763
[fmatches]: https://local-features-tutorial.github.io/pdfs/Local_features_from_paper_to_practice.pdf
[cyclegan]: https://junyanz.github.io/CycleGAN/
[oims]: https://ai.googleblog.com/2020/02/open-images-v6-now-featuring-localized.html
[panictraining]: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187344
[mgda-ub]: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
[precompdisc]: https://www.kaggle.com/c/landmark-recognition-2020/discussion/176697
[keetar-glr]: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176037
