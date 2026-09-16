# 44th Place Solution 🎉🎉🎉

Competition: bengaliai-speech
Rank: #44
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/450635

Before into the topic, I would like to congratulate all team members:  @nanaxing, @focuswilliam, @zhangjinru, @marcocheung0124. The first three of them are undergraduates and they are all new to Kaggle. CongratulationsBefore the topic, I would like to congratulate all team members:  @nanaxing, @focuswilliam, @zhangjinru, @marcocheung0124. The first three of them are undergraduates and they are all new to Kaggle. Congratulation again on their first medal and the success of our first audio competition🥈🥈🥈!

Then I acknowledge @takanashihumbert for his published training notebook [https://www.kaggle.com/code/takanashihumbert/bengali-sr-wav2vec-v1-bengali-training/notebook](url). Another acknowledgment is for @mbmmurad for his work to introduce the audios of the dataset [https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0](url) in [https://www.kaggle.com/code/mbmmurad/dataset-overlaps-with-commonvoice-11-bn/notebook](url).

**Dataset**: We directly used the data provided by the organizer and did not use any external data.

**Training Environment**: One Colab Pro+ account is utilized. Due to limitations of equipment and computing power, we are unable to train more models(like the punctuation model) and complete data sets. 

**Data Augmentation**: Some augmentations like HighLowPass, Noise, and PitchShift are used to increase robustness.

**Model Training**: Pretrained model is from [https://www.kaggle.com/datasets/takanashihumbert/bengali-ex002](url). We put in one-tenth of the data for training each time (feature encoder and feature extractor take turns to freeze). For freezing the feature extractor, the lr is 2e-5 for warmup. For the feature encoder, it is  6e-6. In our experiments, bs=1. After training three-fifths of the data, the public score is 0.42 and private score is 0.503.

**Decoding Parameters**: We adjust the decoding parameters and displayed them in the notebook [https://www.kaggle.com/code/yuliknormanowen/bengali-sr-wav2vec-v1-bengali-inference-for-v4?scriptVersionId=146374376](url). 

Thank you all so much for reading this. If you have any suggestions, we are happy to accept them. We would be very grateful if you could upvote this topic🥺🥺🥺🙏🙏
