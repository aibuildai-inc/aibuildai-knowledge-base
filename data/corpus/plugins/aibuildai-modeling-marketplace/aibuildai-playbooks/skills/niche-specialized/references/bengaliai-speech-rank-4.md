# 4th place solution

Competition: bengaliai-speech
Rank: #4
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/447995

Many thanks to the organizers for this interesting competition. Speech recognition is a very interesting direction and I did learn a lot from the discussion and public code of the many contestants in this competition. The past three months have been stressful but rewarding. I will try to make my solution clear in my broken English

## Summary
My solution is relatively simple, using a wav2vec2 1b model as the pretrain model and training a Wav2Vec2ForCTC model. During the post-processing stage, a 6-gram language model is trained using KenLM, followed by further post-processing of normalization and dari on the output results.

## Wav2Vec2ForCTC Training
Specifically, I use [facebook/wav2vec2-xls-r-1b](https://huggingface.co/facebook/wav2vec2-xls-r-1b) as the pretrain model. Training this model requires three stages, with different random seeds and  consistent data augmentation and parameters in each stage:
* Optimizer: AdamW (weight_decay: 0.05; betas: (0.9, 0.999))
* Scheduler:  Modified linear_warmup_cosine scheduler (init_lr: 1e-5; min_lr: 5e-6; warmup_start_lr: 1e-6; warmup_steps: 1000; max_epoch: 120; iters_per_epoch: 1000)
```python
class LinearWarmupCosine3LongTailLRScheduler:
    def __init__(
            self,
            optimizer,
            max_epoch,
            min_lr,
            init_lr,
            iters_per_epoch,
            warmup_steps=0,
            warmup_start_lr=-1,
            **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.iters_per_epoch = iters_per_epoch
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.max_iters = max_epoch * iters_per_epoch

    def step(self, cur_epoch, cur_step):
        # assuming the warmup iters less than one epoch
        total_steps = cur_epoch * self.iters_per_epoch + cur_step
        if total_steps < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        elif total_steps <= self.max_iters // 4:
            cosine_lr_schedule(
                epoch=total_steps,
                optimizer=self.optimizer,
                max_epoch=self.max_iters // 4,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )
        elif total_steps <= self.max_iters // 2:
            cosine_lr_schedule(
                epoch=self.max_iters // 4,
                optimizer=self.optimizer,
                max_epoch=self.max_iters // 4,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )
        else:  # total_steps > self.max_iters // 2
            cosine_lr_schedule(
                epoch=total_steps - self.max_iters // 2,
                optimizer=self.optimizer,
                max_epoch=self.max_iters // 2,
                init_lr=self.min_lr,
                min_lr=0,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
```

* base DataAugmentation (denode as `base_aug`): 
```python
def get_transform(musan_dir):
    trans = Compose(
        [
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
            Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
            OneOf(
                [
                    # AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=5.0,
                    AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=3.0, max_snr_in_db=30.0,
                                       noise_transform=PolarityInversion(), p=1.0),
                    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
                ] if musan_dir is not None else [
                    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0), ],
                p=0.5,
            ),
        ]
    )
    return trans
```

* composite DataAugmentation (denode as `comp_aug`)。I use three comp_augs：
    1. split an audio wave evenly into 3 segments and perform base_aug on each segment;
    2. randomly select two speeches from the dataset, perform base_aug on each of them separately, and then concatenate them together;
    3. combine the above two data augmentation methods.

* dataset
For Wav2Vec2ForCTC training, I didn't use external data, I filtered the competition dataset in the following steps:
    1. train a model based on [arijitx/wav2vec2-xls-r-300m-bengali](https://huggingface.co/arijitx/wav2vec2-xls-r-300m-bengali) 
    2. use the model above to inference the whole dataset, sort all sample scores from small to large and retain the top 70% of the data


## KenLM training
* dataset:
I use IndicCorpv1 and IndicCorpv2 as corpus. After cleaning, the two corpus are combined without reduplicates.

* corpus cleaning:
I clean each sentence in the corpus using the below code:
```python
chars_to_ignore = re.compile(r'[^\u0980-\u09FF\s]')
long_space_to_ignore = re.compile(r'\s+')
bnorm = Normalizer()

def fix_text(text: str):
    # remove punctuations
    text = re.sub(chars_to_ignore, ' ', text)
    # match multiple spaces and replace them with a single space
    text = re.sub(long_space_to_ignore, ' ', text).strip()

    return text

def norm_sentence(sentence):
    sentence = normalize(sentence)
    sentence = fix_text(sentence)
    words = sentence.split()
    try:
        all_words = [bnorm(word)["normalized"] for word in words]
        all_words = [_ for _ in all_words if _]
        if len(all_words) < 2:
            return ""
        return " ".join(all_words).strip()
    except TypeError:
        return None
```

* 6gram language model is trained.

## Tried but not work
1. use external data, like openslr, shrutilipi
2. use deepfilternet to denose the audio
3. use larger model (wav2vec2-xls-r-2b)
4. use whisper-small
5. use more corpus to train an language model (BanglaLM)
6. train a spelling error correction model
7. train the fourth stage model
8. ...
There's a lot more, off the top of my head

## Update
* I just found the 7gram-language kenlm model is chosen to be the final submission version, not the 6gram version, even though their scores are the same.
* inference code is made public: https://www.kaggle.com/code/hanx2smile/4th-place-solution-inference-code
* training code is made public: https://github.com/HanxSmile/lavis-kaggle
