# [3rd place solution with code] 17 layers squeezeformer with timerduce and ROPE

Competition: asl-fingerspelling
Rank: #3
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434393

This is an intesting game, I learned a lot from 

https://www.kaggle.com/code/hoyso48/1st-place-solution-training
https://www.kaggle.com/code/irohith/aslfr-ctc-based-on-prev-comp-1st-place
https://www.kaggle.com/code/markwijkhuizen/aslfr-transformer-training-inference

# **Preprocss**
1. **Do not throw away frames witout hands!**
2. Use as much info as possible, (hands, lips, eye, nose, pose) , use (x,y,z)
3. Use more frames, resize to 320 frames for model training.
4. I used **original + normalized + abs position(/1000.)** as input feats(total 384 * 2 + 1 = 769). 
normalized feats help converge much faster but I found original feats also helps.

# **Aug**
I followed last competion 1st solution, and found the most important aug are 
1. **time scale (interp1d)**
2. **time dim mask**
I used heavy mask here which will mask some time seq and aslo will **randomly mask 50% frames**
3. affine 
follow preve 1st solution with prob 0.75, for left-right flip only with prob 0.25

# **Training**
1. I used **train data + sup data combined training with sup data weight set to 0.1** for 400 epochs.
2. batch size 128, max lr 2e-3, lr scheduler using linear decay with 0.1 epochs warmup , adam optimizer
2. Awp training started from epochs * 0.15, with adv_lr 0.2 and adv_eps 0.
3. **Fintune with train data only for 10 epochs** with max lr 1e-4, awp start from epoch 2.  

# **Postdeal**
I used **rule for blank index(0)**


# **Model**
1. **Squeezeformer**  works best, it is a bit faster then conformer. (Code from NEMO)
2. In order to make net deeper I used **1 down sampling layer to reduce frames from 320 to 160**.
This is mainly helpful for adding more encoder layers, and also can allow using complex encoder layers.
3. **Relative pos encoding** affect the performance so much, I found **ROPE(by Jianlin Su)** perform best and super fast. (code from huggingface implementation)
4. **Dropout is super important, I used final cls_drop=0.1 only, no dropout for other layers.**
5. I learned from 1st-place-solution-training that **stochastic path** is **super important to avoid overfit for deep network**. So I used it in squeeeze former block for each layer like below(notice the InstDropout and 0.5 skip factor all super important for final performance)


Overall model arch below:


# **Others**
1. For me (using 4090), torch much faster then tf (3-4 times faster)  
2. I used tf at first and swich to torch in the last month which help to speedup experiments a lot.    
Speed is only one factor, another important thing is I could easily try opensource code of ASR like NEMO or espnet.
3. I used **nobuco** to convert torch model to keras, it works like a charm.  
4. I still used tf for prorcess/aug and post process and also use tfrecord as input format, I wrote a torch iterable dataset which wrap tfrecord reader.
# **TODOS**
Due to time limit, I could not finish more experiments at last days, but some possible improvements might be
1. Model can be deeper up to **20 layers**
20 layer model perform better then 17 layers but I only trained 300 epochs which perform not as good as 17 layers + 400 epochs.
2. More epochs training, maybe 500 or 600 ? 
For 17 layer model from 300 to 400 improve LB 4 points and PB 2 points, so might sitll could train more epochs and might use 20 layer model for more epochs help even more:)
3. From what I learned from other solutions, it seems I missed some major points here
- cutmix
this is a pity, I planned to do this at the begging but did not try it, as I found hard mask of frames(50%+) worked very well, I should have realized that cut mix might help even more.  
simple concat 2 instances  
concat with some ratio like 0.7 and 0.3  
concat with using ctc segmentation   
- seq2seq method and ctc+attention decode method    
I tried seq2seq using tf at the begging of this competition but not give good results, should have tried it after changing to use torch with squeezeformer encoder.  
- input len mask to speedup infer
- try even more max input frames from 320 to 384 or 512(with input len mask infer)
# **Code**
Opensource all codes here:  
https://www.kaggle.com/code/goldenlock/3rd-place-step1-gen-tfrecords-for-train
https://www.kaggle.com/code/goldenlock/3rd-place-step2-gen-tfrecords-for-supplement
https://www.kaggle.com/code/goldenlock/3rd-place-step-3-gen-mean-and-std
https://www.kaggle.com/code/goldenlock/3rd-place-step-4-train-squeezeformer
https://www.kaggle.com/code/goldenlock/3rd-place-step-5-torch2keras-using-nobuco
https://www.kaggle.com/code/goldenlock/3rd-place-step-6-inference  
Notice I did not reproduce my final results using kaggle notebooks, so if you want to reproduce or want to find the original code you could find it here:
https://github.com/chenghuige/Google-American_Sign_Language_Fingerspelling_Recognition
