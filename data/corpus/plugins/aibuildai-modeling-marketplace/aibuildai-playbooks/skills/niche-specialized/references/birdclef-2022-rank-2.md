# [Public #1 Private #2] + [Private #7/8 (potential)] solutions. The host wins.

Competition: birdclef-2022
Rank: #2
Source: https://www.kaggle.com/c/birdclef-2022/discussion/326950

As I promised, I will publish my solution right after the competition. This will be a long topic as the two solutions are totally different. As kaggle says > You've been sharing this link too often. We prevent redundant posts to reduce spam. So I put some link in the code block.

Before the solutions, thanks to my teammate @jionie, our second gold medal, and first-time money zone!!!!!!!!!! Cheers!!!!!!!!!!

I will start with our [Private #7/8 (potential)] Solution first since I **promise **you will say the "F" word after seeing the other solution.

**[Private #7/8 (potential)] Solution**

Training:
Our first solution is based on @kaerunantoka 's [public notebook](https://www.kaggle.com/code/kaerunantoka/birdclef2022-use-2nd-label-f0) and last year's second-place solution. The training code is pretty the mixture of the two pipelines mentioned above. 
Some key points to mention:
1. data augmentation:
`OneOf([
                        Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
                        GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
                    ]),
                    OneOf(
                        [
                            NoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=5, max_snr=20),
                            PinkNoise(p=1, min_snr=5, max_snr=20),
                            AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                            AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
                        ],
                        p=0.3,
                    ),
                    AddBackgroundNoise(
                        sounds_path=self.config.BACKGROUND_PATH, min_snr_in_db=0, max_snr_in_db=2, p=0.5
                    ),
                    Normalize(p=1),`

2. cut mix + mix up
3. loss function: BCEWithLogits + BCEFocal2WayLoss
4. hypers setting (n_fft, n_mels, hop_length) refer to our inference notebook.

Our inference kernel and all trained models are [[publicly available](https://www.kaggle.com/code/leonshangguan/private-7-8-final-of-submission)] 
Only one key point to mention: if a bird is detected in the previous or next 5s, we will rank the probs and add the top5 bird species to the current detected birds.

Because this solution was not selected as our final submission, we don't know the actual rank, but as it scores 0.79 in private, it should rank 7-8.


**[#1 Private #2] Solution**

Actually, I am curious that no one found it, the solution is provided by the host and finally allowed by the host.

1. If you look at the discussion [Meet the host](https://www.kaggle.com/competitions/birdclef-2022/discussion/307941), you will find the host mentioned [BirdNet project](https://birdnet.cornell.edu/) in the post.
2. Click the link, and it is easy to find the host's [github repo](https://github.com/kahst/BirdNET-Analyzer).
3. Then, find the overlap classes between the [host's model](https://github.com/kahst/BirdNET-Analyzer/blob/main/checkpoints/V2.1/BirdNET_GLOBAL_2K_V2.1_Labels.txt) and the scored birds, 20 out of 21 scored birds are the same (except aniani).
4. Modify species_list.txt under example folder accordingly, I have uploaded the modified repo to kaggle [here](https://www.kaggle.com/datasets/leonshangguan/birdnet.).
5. Do some post-processing as in our [[inference notebook](https://www.kaggle.com/code/leonshangguan/birdnet-inference)]
6. Thanks to @ivanpan who opened [this issue](https://github.com/kahst/BirdNET-Analyzer/issues/40), the host says `we won't enforce the "non-commercial" clause in the license for BirdCLEF, and you can use BirdNET in your submissions.` (I don't know if they also benefit from this repo, we can wait for their solution)

Some Notes:
Honestly, we didn't originally intend to use this model as a final submission until we notice the issue mentioned above, and actually, the ensemble of our own model could also reach the gold zone. The answer is provided by the host so that's why I say it is a weird solution. I guess the BirdNet was trained on the data for the public leaderboard but not the private one since there's a huge drop between the public lb to private (0.91 --> 0.84; 0.85 --> 0.78)

The BirdNet is running on CPU less than 2h inference time, while ours cost about 8h to run on GPU. So the host wins, my hope is the first place team doesn't benefit from that model and can beat the host. Looking forward to their solution.

**Finally, a big thanks to @stefankahl for hosting this amazing competition and providing the answer as well xD.**
