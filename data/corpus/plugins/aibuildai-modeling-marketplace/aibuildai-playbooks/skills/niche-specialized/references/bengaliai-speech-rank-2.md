# 2nd place solution

Competition: bengaliai-speech
Rank: #2
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/447976

Many thanks to Kaggle and Bengali.AI for hosting such an interesting competition. As I was about to start my PhD  research on speech processing and I knew nothing about the field, the competition came as the perfect opportunity for me to learn. In the end it was a highly rewarding experience.

The solution consists of 3 components:
- ASR model
- Language model
- Punctuation model

**1. ASR model**

I used `ai4bharat/indicwav2vec_v1_bengali` as pretrained model.

**Datasets:**
- Speech data: Competition data, Shrutilipi, MADASR, ULCA (for ULCA data most of the links are dead but some are still downloadable, only a few thousands samples though)
- Noise data: music data from MUSAN and noise data from DNS Challenge 2020
- All data are normalized and punctuation marks are removed except for dot (`.`) and hyphen (`-`)

**Augmentation:**
- use augmentation from `audiomentations`, apply heavy augmentation for read speech (comp data, MADASR) and lighter augmentation for spontaneous speech (Shrutilipi, ULCA)
Example of read speech augmentation:
```
augments = Compose([
    TimeStretch(min_rate=0.8, max_rate=2.0, p=0.5, leave_length_unchanged=False),
    RoomSimulator(p=0.3),
    OneOf([
        AddBackgroundNoise(
            sounds_path=[
                '/path_to_DNS_Challenge_noise',
            ],
            min_snr_in_db=5.0,
            max_snr_in_db=30.0,
            noise_transform=PolarityInversion(),
            p=1.0
        ),
        AddBackgroundNoise(
            sounds_path=[
                '/path_to_MUSAN_music'
            ],
            min_snr_in_db=5.0,
            max_snr_in_db=30.0,
            noise_transform=PolarityInversion(),
            p=1.0
        ),
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
    ], p=0.7),
    Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.2),
    ])
```
For spontaneous speech augment probabilities are smaller and `TimeStretch` rate much less extreme.
- concat augment: randomly concatenate short samples together to make length distribution of training set closer to OOD test set.
- SpecAugment: mask_time_prob = 0.1, mask_feature_prob = 0.05.

**Training:**
- First fit on all training data, then remove about 10% with highest WER after fitted from train set.
- Don't freeze feature encoder.
- Use cosine schedule with warmups and restarts: 1st cycle 5 epochs peak lr 4e-5, 2nd cycle 3 epochs peak lr 3e-5, third cycle 3 epochs peak lr 2e-5.

**Inference:**
Use `AutomaticSpeechRecognitionPipeline` from `transformers` to apply inference with chunking and stride:
```
text = pipe(w, chunk_length_s=14, stride_length_s=(6, 3))["text"]
```

**2. Language model**

6-gram kenlm model trained on multiple external Bengali corpus:
- IndicCorp V1+V2.
- Bharat Parallel Corpus Collection.
- Samanantar.
- [Bengali poetry dataset](https://www.kaggle.com/datasets/truthr/free-bengali-poetry).
- [WMT News Crawl](https://data.statmt.org/news-crawl/).
- Hate speech corpus from https://github.com/rezacsedu/Classification_Benchmarks_Benglai_NLP.

**3. Punctuation model**

Train token classification model to add the following punctuation set: `।,?!`

- use `ai4bharat/IndicBERTv2-MLM-Sam-TLM` as backbone
- add LSTM head
- train for 6 epochs, cosine schedule, lr 3e-5 on competition data + subset of IndicCorp
- mask 15% of the tokens during training as augmentation
- ensemble 3 folds of model trained on 3 different subsets of IndicCorp
- beam search decoding for inference.

Thank you very much for reading and please let me know if you have any questions.

Update: 
- training code: https://github.com/quangdao206/Kaggle_Bengali_Speech_Recognition_2nd_Place_Solution
- inference notebook: https://www.kaggle.com/code/qdv206/2nd-place-bengali-speech-infer/
