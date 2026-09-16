# 1st place solution

Competition: bengaliai-speech
Rank: #1
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/447961

STT Model:
- OpenAI whisper-medium
- Huggingface trainer
- Trained on 8x 48GB RTX A6000
- bs=8 and lr=1e-5
- Train steps 50k
- Spectrogram dithering
- Spectrogram time and frequency masking
- Resampling 16khz->8khz->16khz as augmentation
- Inference with max_length=260, num_beams=4 and chunk_length_s=20.1s
- Libsonic based speed/pitch augmentation
- Datasets: OpenSLR 37, OpenSLR 53, MadASR, Shrutilipi, Macro, Kathbath, GoogleTTS generated audios and pseudo labeled YouTube videos


Punctuation Model:
- AutoModelForTokenClassification google/muril-base-cased
- Huggingface trainer
- Labels: period, comma and question mark
- bs=64, lr=2e-4 and max_seq_length=512
- Ensemble of 4 models (using 6, 8, 11 and 12 layers of google/muril-base-cased)
- Normalized IndicCorp v2 Bangla dataset

In my daily job, I do speech speech recognition for low resource central Asian languages. From my experience, OpenAI Whisper works really well for OOD audios and can even transribe song lyrics. The downside is it is very sensitive to the annotation noise. So fixing the annotation noise, is the most crucial part of this competition.

Because the competition dataset was not validated, the initial model was trained on OpenSLR datasets. We normalized the texts and filtered out texts containing Bengali digits. All punctuation was also removed. Additionally, we sampled 420k texts from the IndicCorp and synthesized audios using GoogleTTS, which were then used as training datasets.

Following the training of an initial Whisper-medium model on OpenSLR and GoogleTTS, we conducted inference on MadASR, Shrutilipi, Macro, and Kathbath. We included audios with a WER of less than 15% in the next training phase. After three rounds of training, the model achieved an 8% WER on the Macro validation dataset and a public leaderboard score of approximately 0.380.

Since most of the training set audios were short, we merged some short audios to create around 70k longer audios. Subsequently, we achieved a public leaderboard score of approximately 0.370.

Whisper with the original tokenizer was slow on Bengali audios. Therefore, we trained a Whisper tokenizer with a 12k vocabulary on Bengali texts. With this tokenizer, we were able to perform inference with a num_beam value of up to 8 and a chunk_length_s of 20.1 seconds in less than 7 hours.

In the next step, we applied pseudo labeling to some YouTube videos, which enabled us to achieve a public leaderboard score of approximately 0.360. When we combined the predictions of four punctuation models, our public leaderboard score improved to around 0.325.

By adding more pseudo-labeled YouTube videos, our public leaderboard score further improved to 0.312 (private LB 0.372)

model weight and inference notebook: https://www.kaggle.com/competitions/bengaliai-speech/discussion/447970
cleaned/long/pseudo data: https://www.kaggle.com/competitions/bengaliai-speech/discussion/448110
