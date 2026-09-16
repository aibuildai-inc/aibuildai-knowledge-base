# 31st place silver medal solution - My first competition medal

Competition: bengaliai-speech
Rank: #31
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/448030

First, we want to thank Bengali.ai and Kaggle for hosting such an excellent competition on Bengali ASR and releasing a large-scale dataset for this domain. This competition was challenging, to say the least. But I enjoyed the last three months working on it ngl.  It was a pretty good learning curve for me. I got the opportunity to contribute to the community and also engage with the community through this competition. I got 1 Gold, 2 silver and 8 bronze medals for my notebooks in this competition. So overall, a very busy and happy three months!

## Summary : 

**Acoustic Model**: We fine-tuned the ai4bharat/indicwav2vec_v1_bengali model
**Dataset** : Competition Data + Openslr53+openslr37+ TTS dataset
In the first phase, we fine-tuned the model with the whole competition data. which was making things worse, dragging the LB performance below the best public NB ( 0.445)
Then we filtered out the bad-quality audio from the competition data using the training metadata provided by the host ( We took the audios with MOS>=2). 
We did a random split with both the train and validation data(since it was evident MaCro validation audio quality was better than train audio)  and trained with 95% of the audios. Then added the other datasets. This improved the LB score to 0.430
Augmentation : reverberation, Speed perturbation,Volume perturbation (0.125x ~ 2.0x)
Adding background noise from the example audios(Improved performance slightly)

**Language Model**: We built a 5-gram LM using competition sentences + [banglanmt](https://github.com/csebuetnlp/banglanmt) + IndicCorp_V2

## What did not work: 
1. **Punctuation model** : I tried to add the [xashru-punctuation-restoration](https://github.com/xashru/punctuation-restoration) model but it always gave CUDA OOM. Now it's bugging me to see others implemented it and got a huge boost with it. I also tried to fine-tune T5 model for this task but didn't see much improvement. So I gave up the idea.
2. **Speech enhancement** : 
- [DeepFIlterNet](https://github.com/Rikorose/DeepFilterNet)
- [Speechbrain wham](https://huggingface.co/speechbrain/sepformer-wham-enhancement)

I spent a significant amount of time on them. First I tried to add a speech enhancer model to my inference pipeline. It didn't improve the performance. Then I tried to train with enhanced audio, but it didn't help either. The noise cancellation performance by these models was quite good, but it also affected the pitch and distorted the speech a little bit. 

**What could have been done to improve further/ Next steps** : 
1. Adding the punctuation model.
2. Train the acoustic model with more data (MADASR,Shrutilipi)
3. Add more texts to the LM
