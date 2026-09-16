# 1st place solution

Competition: sp-society-camera-model-identification
Rank: #1
Source: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49367

## Models ##
Here is the list of our best model. Half of them is based on Andres code (forked before adding any restrictions on usage) and half on its PyTorch implementation:

- 984_densenet201_antorsaegen_29_0.98624
- 976_densenet201_antorsaegen_62_0.98271
- 977_resnet50_antorsaegen_119_val_0.9815
- 976_DenseNet201_do0.3_doc0.0_avg-epoch072-val_acc0.981250
- 967_InceptionResNetV2_do0.1_avg-epoch154-val_acc0.965625
- 962_Xception_do0.3_avg-epoch079-val_acc0.991667 (leaky validation, pls ignore)

All of the models had a full-size crop (512) + TTA 8. Choosing this crop was our biggest flaw since smaller crops allow faster training, more TTA options and produce almost the same accuracy.  

## Data ##
In total, we collected 300 Gb of photos from Flickr and Yandex.Foto. But due to the huge crop size, we could not fully utilize the entire data set and used only 20,000 photos for training. The rest 50,000 was used for blending. 
Data was filtered by resolution and camera type, also all photos with non-default software in Exif  (i.e. processed somehow) and with poor quality (lower than 95 as identified by ImageMagick) were excluded.

## Hardware ##
Nothing special for a team with five members: 5 x 1080ti + 1070

## Submissions ##
We had 2 submission strategies:

 - Averaging all TTA prediction from all models by power mean - powers of 1,2,4 as a parameter. Additionally, we explicitly made all classes equally distributed (132 manip and 132 unalt photos for each class) by using Hungarian algorithm on probabilities. As expected, this resulted in a huge LB overfit: 0.991 score on public and 0.985 on private.
 - Blending models by using hold-out predictions. There were 52 models with slightly different parameters and equal weights: XGBoost (20) + LightGBM (20) + Keras (12). This gave us 0.986 score on public and 0.989 on private. No class equalization was used with this approach except some weight tuning for Nexus 5 during training (all models had a smaller number of predictions for this phone). 

## Key takeaways ##

- Collect as much data as possible but do not forget to clean it (GIGO)
- Try smaller crops first and all architecture types
- LB probing is evil, trust your CV
