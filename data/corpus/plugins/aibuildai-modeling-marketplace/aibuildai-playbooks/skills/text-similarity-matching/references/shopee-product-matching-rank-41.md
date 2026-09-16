# 41th Place : Approach

Competition: shopee-product-matching
Rank: #41
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238382

A big thanks to Organizers for organizing such a nice competition, I like these types of competitions very much where there is less resource requirement(as I have only Kaggle Kernels and Colab😄) and many tricks can be applied.

Also Congratulations to all Winners ,  participants, contributors in this competitions. All the solutions I have read are so sophisticated  that I was thinking should I post something.

This competition is very unique of its kind where both Image and Text processing can be applied, along with that some post processing tricks. Also to establish a good validation scheme.

Here is my simple one

# Approach

## Fold Assignment
GroupKFold with count of each group is stratified, so the count distribution of the label is almost similar between train and valid. Five folds have been assigned , among which four folds used for training and 1 fold for validation (80%-20%)

## Image Processing
Efficientnet - B1 : 640x640 - only 1 fold
tried another fold with Efficientnet B5 : 512x512 but the increase in LB : 0.001(very negligible) and that too in private, public LB was same
Checkpoint using a custom callback where it calculates F1 on validation sets after doing a threshold search(cosine) after each epoch

## Text processing
Tfidf Vectorization 

## Threshold
Threshold part was bit tricky as it has to be determined at the image text combination. If  I calculate best threshold individually for image and text and then marry the result it's not optimum. In my case keeping threshold =0.45 (cosine)  in both cases was optimum in the validation. In the submission it was just 0.2 less from the validation threshold  (0.25). This has to be established after doing few submissions and this corelates with the validation then after properly.

##Combination
Concatenate Image prediction and text prediction

##Post Processing -1
Remove the FPs based on UOM

##Post Processing -2
If the match is only 1 , concatenate it the closest product based on the distance (considering both image and text, whichever is less)

Also planned to apply few more ideas like BERT, reducing threshold  and increase transitive relation etc etc, but didn't get any time due to some other priority stuff.

##Acknowledgement
https://www.kaggle.com/c/shopee-product-matching/discussion/229495
https://www.kaggle.com/ragnar123/shopee-efficientnetb3-arcmarginproduct/notebook
https://www.kaggle.com/c/shopee-product-matching/discussion/224828
https://www.kaggle.com/c/shopee-product-matching/discussion/225093

Also  a BIG THANKS ( I think it's not enough though) to @cdeotte for his contribution which influenced me a ton whenever I have participated any comp.
