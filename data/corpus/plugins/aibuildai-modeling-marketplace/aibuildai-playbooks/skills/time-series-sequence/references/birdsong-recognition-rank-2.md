# 2nd place solution

Competition: birdsong-recognition
Rank: #2
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183269

Hi. This is my first Kaggle competition, and I really enjoyed participating.
Thank you to the organizers for an interesting competition, @hidehisaarai1213 for the basic solution, and to all the teams for an interesting and intense fight.
# My decision
**github**: https://github.com/vlomme/Birdcall-Identification-competition
**datasets**: https://www.kaggle.com/vlomme/my-birdcall-datasets
**kaggle notebook**: https://www.kaggle.com/vlomme/surfin-bird-2nd-place
-  Due to a weak PC and to speed up training, I saved the Mel spectrograms and later worked with them
- **IMPORTANT**! While training different architectures, I manually went through 20 thousand training files and deleted large segments without the target bird. If necessary, I can put them in a separate dataset.
- I mixed 1 to 3 file
- **IMPORTANT**! For contrast, I raised the image to a power of 0.5 to 3. at 0.5, the background noise is closer to the birds, and at 3, on the contrary, the quiet sounds become even quieter.
- Slightly accelerated / slowed down recording
- **IMPORTANT**! Add a different sound without birds(rain, noise, conversations, etc.)
- Added white, pink, and band noise. Increasing the noise level increases recall, but reduces precision.
- **IMPORTANT**! With a probability of 0.5 lowered the upper frequencies. In the real world, the upper frequencies fade faster with distance
- Used BCEWithLogitsLoss. For the main birds, the label was 1. For birds in the background 0.3.
- I didn't look at metrics on training records, but only on validation files similar to the test sample (see dataset). They worked well.
- Added 265 class nocall, but it didn't help much
- The final solution consisted of an ensemble of 6 models, one of which trained on 2.5-second recordings, and one of which only trained on 150 classes. But this model did not work much better than an ensemble of 3 models, where everyone studied in 5 seconds and 265 classes.
- My best solution was sent 3 weeks ago and would have given me first place=)
- Model predictions were squared, averaged, and the root was extracted. The rating slightly increased, compared to simple averaging.
- All models gave similar quality, but the best was efficientnet-b0, resnet50, densenet121.
- Pre-trained models work better
- Spectrogram worked slightly worse than melspectrograms
- Large networks worked slightly worse than small ones
- n_fft = 892, sr = 21952, hop_length=245, n_mels = 224, len_chack 448(or 224), image_size = 224*448
- **IMPORTANT**! If there was a bird in the segment, I increased the probability of finding it in the entire file.
- I tried pseudo-labels, predict classes on training files, and train using new labels, but the quality decreased slightly
- A small learning rate reduced the rating

I'll try writing more later here or on GitHub. I will be happy to answer your questions, because I almost didn't sleep and forgot to write a lot
