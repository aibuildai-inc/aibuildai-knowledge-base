# 8th solution

Competition: stanford-rna-3d-folding
Rank: #8
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/8th-solution

Thanks for the host to celebrate such a interesting competition. 

To be honest, I almost forgot this competition because my result in the public ranking is not so great enough. But lastly when the private ranking was out and I was so happy because this is my first gold medal and I think it should be greatly attribute to luck because I think my finetuned got a reasonable convey 

- In the begining of the game，I did spent a lot of time digesting the public notebook proposed by [drfold-no-msa](https://www.kaggle.com/code/hengck23/lb0-321-simple-drfold-no-msa). The result is clear that such that the main challenge is to successfully run the code and transfer the XYZ coordinate. Such notebook helped me got rank 11 in the public, but later the proteinx notebook was proposed and the ranking dropped down. 

- Later, thanks for [Proteinx Code](https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/573495), I downloaded the training data and  transfer it to the needed format that used in finetuning the model, I tried a couple times using different LR, CROP lENGTH. Finally I got the best result (I can't remember the setting because I didn't document my experiment results)and submit it to the competition in the last day.

- Since I need to work 995, I have some ideas that haven't implemented yet.
1) Maybe we can randomly mask the item in the sequence just like what bert has doned in its pretraining and then finetuned the model with the training data, such idea can help the model to more robust to the missing sequence.
2) In the inference stage, because of the limitation of the GPU, proteinx will encounter error when the sequence is too long, I guess we can always use the last 300 sequence to predict the final items both in training and inference, for example, if the sequence is longer than 300, we firstly use first 300 to predict XYZ, then we move a window lengh(16,32,64),and then we start to do the inference in next startpoint.

By the way, I am looking for a job related to AI, LOL. If you are looking for someone working with you, contact me~
