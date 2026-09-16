# 21st Place Write-up [Stack-up]

Competition: birdsong-recognition
Rank: #21
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183209

First of all, I would like to thank kaggle and the organizers for hosting such an interesting competition.

thanks for my wonderful team @seshurajup and @tarique7. this was great learning and collaboration.

@seshurajup and I worked hard for this from the last 20days

#### [Big Shake-up]
- LB : 0.533 ---> Private LB : 0.632
- From 364th place ---> 21st place




### [Summary]

- converted mp3 audio file into wav with sample_rate 16k, because of colab drive space and birds sing up to 12kHz
- we trained **stage1** model based on 5sec random clip.
- It gives average of 0.65+ AMP on 5-fold
- we created 1sec audio clips dataset.
- predicted 5fold **stage1** models on 1sec dataset.
- selected `df[(df.ebird_code == df.pred_code) & (df.pred_prob >= 0.5)]` selected those 1sec clips for **stage3**
-  we trained **stage2** model on topof **stage1** using public data
- again predicted **stage1** and **stage2** models on public 1sec clips
- again select `df2[(df2.ebird_code == df2.pred_code) & (df2.pred_prob >= 0.5)]`selected those 1sec clips for **stage3**
- **stage3** dataset is `stage3_df = df.append(df2)` we endup with 612K 1sec clips with sudo labels
- we created 1sec noise clips using PANN **Cnn14_16k** model.
- we predict **Cnn14_16k** model on 1sec dataset and select some noise labels from PANN labels 
- those labels are `['Silence', 'White noise', 'Vachical', 'Speech', 'Pink noise', 'Tick-tock', 'Wind noise (microphone)','Stream','Raindrop','Wind','Rain', ...... ]` selected those labels as noise labels
- based on those noise labels and **stage1** model predicted probabilities we select 1sec noise clips data.
- `noise_df = df[(df.filename.isin(noise_labels_df.filename)) & df.pred_prob < 0.4]`
- we end up with 103k noise 1sec clips you find dataset [link](https://www.kaggle.com/gopidurgaprasad/birdsong-stage1-1sec-sudo-noise)
- we know in this competition our main goal is to predict a mix of bird calls.
- now the main part, at the end we have **stage1**, **stage2** models, **1sec** dataset with Sudo labels, and **1sec noise data**. we trained the **stage3** model using all of those.
- now in front of us, we need to build **CV** and train a model that more reliable on predicting a mix of bird calls.

### [CV]
- we created a **cv** based on **1sec bird calls** and **1sec noise data**
- In the end, we need to predict for **5sec** so we take 5 random birdcalls and noise stack them and give labels based on birdcall clips.

```
call_paths_list = call_df[["paths", "pred_code"]].values
nocall_paths_list = nocall_df.paths.values

def create_stage3_cv(index):
    k = random.choice([1,2,3,4,5])
    nocalls = random.choises(nocall_paths_list, k=k)
    calls = random.choises(call_paths_list, k=5-k)
    audio_list = []
    code_list = []
    for f in nocalls:
        y, _ = sf.read(f)
        audio_list.append(y)
    for l in calls:
        path = l[0]
        code = l[1]
        y, _ = sf.read(path)
        audio_list.append(y)
        code_list.append(code)
    random.shuffle(audio_list)
    audio_cat = np.concatenate(audio_list)
    codes = "_".join(code_list)
    sf.write(f"{index}_{codes}.wav", audio_cat, sample_rate=16000)

_ = Parallel(n_jobs=8, backend="multiprocessing")(
    delayed(create_stage3_cv)(i) for i in tqdm(range(160000//5)))
)
```

Ex: `10000_sagthr_normoc_gryfly.wav` in this file you find 3bird calls and 2noise as 5sec clip.
you find the cv dataset at [link](https://www.kaggle.com/gopidurgaprasad/birdsong-stage3-cv)

### [Stage3]
- on top of **stage1** and **stage2** models we trained **stage3** model using **1sec birdcalls** and **1sec noise**.
- the training idea is very simple as same as **cv**.
- at dataloder time we are taking 20% of **1sec noise** clips and 80% of **1sec birdcalls** clips
```
if np.random.random() > 0.2:
	y, sr = sf.read(wav_path)
	labels[BIRD_CODE[ebird_code]] = 1
else:
	y, sr = sf.read(random.choice(self.noise_files))
	labels[BIRD_CODE[ebird_code]] = 0
```
- at each batch time, we did something like shuffle and stack, inspired from cut mix and mixup
- In each batch, we have 20% noise and 80% birdcalls shuffle them and concatenate.
```
def stack_up(x, y, use_cuda=True):
	batch_size = x.size()[0]
	if use_cuda:
		index0 = torch.randperm(batch_size).cuda()
		index1 = torch.randperm(batch_size).cuda()
		index2 = torch.randperm(batch_size).cuda()
		index3 = torch.randperm(batch_size).cuda()
		index4 = torch.randperm(batch_size).cuda()
	ind = random.choice([0,1,2,3,4])
	if ind == 0:
		mixed_x = x
		mixed_y = y
	elif ind == 1:
		mixed_x = torch.cat([x, x[index1,  :]], dim=1)
		mixed_y = y + y[index1,  :]
	elif ind == 2:
		mixed_x = torch.cat([x, x[index1,  :], x[index2]], dim=1)
		mixed_y = y + y[index1,  :] + y[index2,  :]
	elif ind == 3:
		mixed_x = torch.cat([x, x[index1,  :], x[index2], x[index3,  :]], dim=1)
		mixed_y = y + y[index1,  :] + y[index2,  :] + y[index3,  :]
	elif ind == 4:
		mixed_x = torch.cat([x, x[index1,  :], x[index2], x[index3,  :], x[index4,  :]], dim=1)
		mixed_y = y + y[index1,  :] + y[index2,  :] + y[index3,  :] + y[index4,  :]
	mixed_y = torch.clamp(mixed_y, min=0, m[](url)ax=1)
	return mixed_x, mixed_y
```
- for **stage3** model we mouniter row_f1 score from this [notebook](https://www.kaggle.com/shonenkov/competition-metrics) 
- for every epoch our **cv** increased, then we conclude that we are going to trust this **cv**
- at the end the best **cv** **row_f1** score in between **[0.90 - 0.95]**
- this all processes are done in the last 3days so we are managed to train up to 5 models.
- at the end we don't have time as well as submissions, so we did a simple average on 5 models and using a simple threshold 0.5
- our average **row_f1** score is 0.94+ on 5 models.
### [BirdSong North America Set]
- for some folds in  **stage3** we are only trained on North America birds it improves our **cv**
- you can find North America bird files in this [notebook](https://www.kaggle.com/seshurajup/birdsong-north-america-set-stage-3) 

### [Stage3 Augmentations]
```
import audiomentations as A

augmenter = A.Compose([
	A.AddGaussianNoise(p=0.3),
	A.AddGaussianSNR(p=0.3),
	A.AddBackgroundNoise("stage1_1sec_sudo_noise/", p-0.5),
	A.Normalize(p=0.2),
	A.Gain(p=0.2)
])
```

### [Stage3 Ensamble]
- we trained our stage3 models 2dyas before the competition ending so we are managed to train 5 different models.
- 1. `Cnn14_16k`
- 2. `resnest50d`
- 3. `efficientnet-03`
- 4. `efficientnet-04`
- 5. `efficientnet-05`
- we did a simple average of those 5-models with simple threshold `0.5` our *cv* `0.94+` on LB: `0.533`
- in the end, we satisfied our selves and trust our **CV** and we know we need to predict a mixed bird calls.
- so we selected as our final model and it gives Private LB: 0.632 
- we frankly saying that we are not able to beat the public LB score but we trusted our cv and training process
- that brings us 21st place in Private LB.

> inference notebook : [link](https://www.kaggle.com/gopidurgaprasad/birdcall-stage3-final)
