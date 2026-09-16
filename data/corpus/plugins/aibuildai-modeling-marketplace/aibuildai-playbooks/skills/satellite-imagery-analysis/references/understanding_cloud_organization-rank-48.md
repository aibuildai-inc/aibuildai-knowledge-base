# Krazy Klassifiers - 48th place solution

Competition: understanding_cloud_organization
Rank: #48
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118086

If you could predict empty mask for every empty mask and predict full mask, (i.e. predict every pixel with `rle = '1 183750'`) for every mask, then your CV is 0.686!! Therefore a perfect classifier can win without any segmentation. The following code outputs 0.686:

    train = pd.read_csv('../input/understanding_cloud_organization/train.csv')
    train['pred'] = np.where(~train.EncodedPixels.isna(),'1 183750','')
    train['dice'] = train.apply(lambda x: kaggle_dice(x['EncodedPixels'],x['pred']),axis=1)
    print( train.dice.mean() )

# Classification Models
I focused most of my energy on building classification models and finally achieved 78% classification validation accuracy (on 33% holdout set, i.e. 3-Fold CV) by ensembling two crazy classifiers. The first has 4 backbones that extract features from 4 different resized input images (half size, quarter size, one sixth size, and one eighth size)
  

  
    base_model0 = Xception(weights='imagenet',include_top=False,input_shape=(None,None,3))
    base_model1 = Xception(weights='imagenet',include_top=False,input_shape=(None,None,3))
    base_model2 = Xception(weights='imagenet',include_top=False,input_shape=(None,None,3))
    base_model3 = Xception(weights='imagenet',include_top=False,input_shape=(None,None,3))
    x0 = base_model0.output
    x0 = layers.GlobalAveragePooling2D()(x0)
    x1 = base_model1.output
    x1 = layers.GlobalAveragePooling2D()(x1)
    x2 = base_model2.output
    x2 = layers.GlobalAveragePooling2D()(x2)
    x3 = base_model3.output
    x3 = layers.GlobalAveragePooling2D()(x3)
    x = layers.concatenate([x0,x1,x2,x3])
    x = layers.Dense(4,activation='sigmoid')(x)
    model = Model(inputs=(base_model0.input, base_model1.input, base_model2.input, 
        base_model3.input), outputs=x)

My second model uses masks in addition to labels and achieves 77% accuracy by itself. The label loss is backpropagated through the mask prediction. Then instead of using the outputted labels, we predict 1 or 0 for label based on whether mask is present or not.
  


    model0 = Unet('resnet34', input_shape=(None,None,3), classes=4,
        activation='sigmoid', encoder_freeze=True)
    model0.layers[-1].name = 'out1'
    x = model0.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4, activation='sigmoid', name='out2')(x)
    model = Model(inputs = model0.input, outputs = (model0.output,x))
    model.compile(optimizer=opt, loss={'out1':loss1,'out2':loss2}, 
        metric = {'out1':metric1, 'out2':metric2})

# Segmentation Model
My segmentation model is a collage of ideas from public kernels. Without post process, it achieves Public LB 0.650. Test time augmentation (TTAx6) increases this to LB 0.655. Using 3-Folds increases this to LB 0.660. Ensembling 7 copies with different choices for 3-Fold achieves LB 0.665. And finally removing false positives with my classifier increases this to LB 0.670. My final solution has CV 0.663 and Private LB 0.663. Here are specific details:

* Unet Architecture
* EfficientnetB2 backbone
* Train on 352x544 random crops from 384x576 size images
* Train augmentation of flips and rotate
* Adam Accumulate optimizer
* Jaccard loss
* Kaggle Dice metric, Kaggle accuracy metric
* Reduce LR on plateau and early stopping
* Remove masks less than 20000 pixels
* TTA of flips and shifts
* 3-Fold CV and prediction
* Remove false positive masks with classifier

# Kaggle Notebook
I posted a Kaggle notebook showing my segmentation model [here][1]. It scores LB 0.665 by itself and LB 0.670 if you ensemble it with 7 copies of itself with different initialization seeds. It loads classification predictions from my offline classifier models for false positive removal.
  
Thank you everyone for a fun and exciting competition. I learned a lot from reading everyone's discussions and posted code. Thank you Kaggle and Max-Planck-Institite for sharing cloud data and hosting. Congratulations to all the winners.

[1]: https://www.kaggle.com/cdeotte/cloud-solution-lb-0-670
