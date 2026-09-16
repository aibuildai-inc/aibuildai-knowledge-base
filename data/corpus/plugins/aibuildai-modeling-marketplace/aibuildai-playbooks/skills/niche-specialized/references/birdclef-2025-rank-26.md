# Public 20th / Private 26th Solution LB: 0.908

Competition: birdclef-2025
Rank: #26
Source: https://www.kaggle.com/c/birdclef-2025/discussion/583306

Thanks everyone for the interesting competition, and congratulations to the winners ! I am happy that as a community we were able to detect these species to that much accurate. 

**Overview**
My solution is a simple 2 stage solution. 

**First Stage Models**
- I was able to get LB 0.865 to 0.872 with the following settings. I had 3 CNNs and 1 SED trained and ensemble achieved LB 0.886.
- In this stage, I filtered human sounds as everyone else in the training data but randomly augmented a CSA recording with 30% of human sound with 50% probability. 
- Another important / thing for me was to use the following settings. 
`{'sample_rate': 32000, 'n_mels': 384, 'f_min': 0, 'f_max': 16000, 'n_fft': 3072, 'normalized': True, 'hop_length': 420}`
`{'sample_rate': 32000, 'n_mels': 448, 'f_min': 50, 'f_max': 16000, 'n_fft': 4096, 'normalized': True, 'hop_length': 334}`
- After apply AmplitudeToDB, I had to clip the values b/w 0 to -80. 
- During training, I pick 15 seconds of audio chunk from complete audio file by RMS sampling, and then divide it into 3 x 5 seconds segments. Model predicts logits on 3 x 5 seconds of recordings, 3 x 206 and then I pick the max of these 3 recordings and then back propagate on those max logits of these 3 segments.

```python
BS, K, C, H, W = spec.size()
spec = spec.view(BS * K, C, H, W)
logits = model(spec)
logits = logits.view(BS, K, 206)
logits, _ = torch.max(logits, dim=1)
```

- RMS sampling worked much better than random sampling for me.
- Loss: FocalBCE
- Augmentation: Mixup (p=1), LocalGlobal Stretch, Time / Frequency Shift, Time / Frequency Masking, Gaussian Noise. [All on spectrograms]
- Batch Size of 32.
- Model Soup from 12 to 15 epochs.


**Second Stage Models**
- I was able to get Public LB 0.909 / Private LB 0.908 with 2nd stage models.
- I generated probabilities for 5 second chunks all of the train audios, and filtered if max prob of the segment matches with primary label of that audio.
- I also generated pseudo labels of the train segments.
- I trained 4 CNNs here with these segmented audio chunks, where I had 96 Batch Size of Train Audio Segments, and 5 Batch Size = (5 x 12) Batch Size of Train Soundscapes. 
- These 4 CNNs are then further fine-tuned on just pseudo labels for 7 epochs. 


**Missed**
- My model's Public LB performance was impacted when I added No Call in the training on stage 1, I tried it in multiple ways but it didn't work, and now I saw in private LB that my single stage 1 model was able to achieve 0.885 with No Call, Adding that and fine-tuning that should improve the results I think. 

[Inference Notebook](https://www.kaggle.com/code/salmanahmedtamu/fork-of-fork-of-less-smoothing-ensemble-eff-and-nf?scriptVersionId=243764033)
[Training Notebook](https://www.kaggle.com/code/salmanahmedtamu/20th-place-training/)
