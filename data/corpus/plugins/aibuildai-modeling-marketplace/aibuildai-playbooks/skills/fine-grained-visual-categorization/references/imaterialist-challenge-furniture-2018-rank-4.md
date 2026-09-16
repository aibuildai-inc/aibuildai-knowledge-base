# 4th place solution

Competition: imaterialist-challenge-furniture-2018
Rank: #4
Source: https://www.kaggle.com/c/imaterialist-challenge-furniture-2018/discussion/57939

Here is a brief overview of our solution. 

 1. We used the data got from https://www.kaggle.com/c/imaterialist-challenge-furniture-2018/discussion/54222, 192171 images for training and 6309 images for validation.
 2. At first, I used Caffe to train the models (InceptionResnetV2, SeResnXt50, Resnet101). It was easy to overfit. Then, I tried to use pytorch to train the models with the strategy learned from Dowakin. Thanks for his sharing. I tried almost all the models that I can use. You can see the attachment ( furniture - Youngkl_pytorch.pdf) for detailed training parameters and results. 
 3. We tried to ensemble the models we had. We tried average ensemble and weighted ensemble. We got the best score 0.12760 on public LB. For testing, we used TenCrop. Then, we decided use pseudo-labeled images from the testset. (However, I forgot to apply a threshold to refine the testset images.) This is  pseudo label1.  Afterwards, we got results scoring 0.12682 and 0.12656 on public LB. We got pseudo label2 and pseudo label3 from these results, and trained some models.
 4. At last, I average ensembled all the models got from pseudo label1, pseudo label2 and pseudo label3, except dpn107_ck5 and xception_ck2. 

You can go to my github for more details. https://github.com/Youngkl0726/kaggle_iMaterialist-Challenge-Furniture-at-FGVC5
