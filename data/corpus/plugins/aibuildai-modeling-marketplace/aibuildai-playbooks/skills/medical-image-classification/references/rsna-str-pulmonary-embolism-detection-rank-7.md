# 7th place - EfficientNet, Transformer and a 2nd opinion [edited]

Competition: rsna-str-pulmonary-embolism-detection
Rank: #7
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193460

I used a two stage model.
1. A EfficientNet (B5, B3) used for feature extraction per image
2. A transformer used per series to predict the series related classes and the  'PE Present on Image' per image

This is the 2nd competition I use such network and an extensive description of this model can be found [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/181830)  

The targets for the EfficientNet where: 
* The original targets for images where  PE Present on Image = 1
* 0 for every other image. Except the Intermediate target which remained the same.

The loss was weighted BCE - the weights reflecting the competitions matric weights.
I used flip, rotate, random resize/crop, mean/std shift as augmentation.
I also use trainable 3 windows to convert the CT image to jpeg ([WSO](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117480) )

The transformer was a 4 layer encoder (using Pytorch's transformer encoder module). Where the relative and absolute places of the images in the series were embedded and added to the features vectors (as is done for positional embedding in NLP transformers such as BERT)
The loss function reflected the competition's matric.

This model gave an **LB of 0.166**  

Ensembling improved the **LB to 0.162**, but I could only ensemble 2 models in the time frame (I didn't use the public/private LB trick that can add 25%). to improve this I used a **2nd opinion mechanism** - instead of using a 2nd model to inference all the data again, I did what an MD will do, I chose only series where the results where the most uncertain (near 0.5), inference them with another model and ensembled - I did this 3 times for ~ 30-40% of the data each time gaining an equivalence of ensembling 4 models => **LB 0.157**

As a last step, I checked if any prediction meet the competition's [Label Consistency Requirement ](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/183473) and if it didn't, made the minimal changes needed to meet the requirements

* The full code can be found in [git](https://github.com/yuval6957/RSNA2020_final.git)

* The inference code can be found [in this notebook](https://www.kaggle.com/yuval6967/rsna2020-inference-2nd-op-final)

* The models' weights are [in this](https://www.kaggle.com/yuval6967/rsna2020-models) public dataset

* A more detailed description can be found [in this documentation](https://github.com/yuval6957/RSNA2020_final/blob/master/Documentation.md) and [this presentation](https://github.com/yuval6957/RSNA2020_final/blob/master/RSNA2020%20presentation.pdf)

* [This is a video](https://www.youtube.com/watch?v=hVgIawktZgs) which present this solution
