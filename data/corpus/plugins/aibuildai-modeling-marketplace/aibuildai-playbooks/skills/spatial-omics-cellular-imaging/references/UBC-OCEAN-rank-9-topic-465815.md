# 9th place solution

Competition: UBC-OCEAN
Rank: #9
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465815

### Only use Competition Data, no External Data 

## **Split WSI and TMA:**
WSI images have black pixels (all zeros in all three channels), while TMA images do not. Therefore, if both the image width and height are less than 6000, but the area of black pixels is greater than 5% of the image (all WSI images in the training data have more than 10% black pixels), it is classified as WSI; otherwise, it is classified as TMA.

## **Make tile :**
 First, reduce the size of the WSI by 0.33 times, and then divide it into 512*512 tiles. Subsequently, categorize these tiles into three levels based on the presence of bad pixels, identified by the condition "np.sum(np.ptp(tile, axis=2) < 20)".
Inference make tile code:

```python
def resize_image_and_make_tile(name,out_path,scale):
    path=f"/kaggle/input/UBC-OCEAN/{inference}_images/{name}.png"
    p_mask=f"{pred_mask_512_folder}/{name}.npy"
    image=cv2.imread(path)
    image=cv2.resize(image,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    mask=np.load(p_mask)
    os.makedirs(f"{out_path}/{name}",exist_ok=True)
    count=0
    if (count<20):
        idxs=[(y,x) for y in range(0,image.shape[0]//512) for x in range(0,image.shape[1]//512)]
        random.shuffle(idxs)
        for k, (y, x) in enumerate(idxs):
            tile=image[y*512:(y+1)*512,x*512:(x+1)*512,:]
            #bg_count=np.sum((tile.max(axis=2)-tile.min(axis=2))<20)
            bg_count=np.sum(np.ptp(tile,axis=2)<20)
            if ((bg_count/(512*512))<=0.5):
                cv2.imwrite(f"{out_path}/{name}/{x}_{y}.png",tile)
                count+=1

            if count>=60: #60
                break
        if count<20:
            idxs=[(y,x) for y in range(0,image.shape[0]//512) for x in range(0,image.shape[1]//512)]
            random.shuffle(idxs)
            for k, (y, x) in enumerate(idxs):
                tile=image[y*512:(y+1)*512,x*512:(x+1)*512,:]
                #bg_count=np.sum((tile.max(axis=2)-tile.min(axis=2))<20)
                bg_count=np.sum(np.ptp(tile,axis=2)<20)

                if ((bg_count/(512*512))<=0.65)&((bg_count/(512*512))>0.5):
                    cv2.imwrite(f"{out_path}/{name}/{x}_{y}.png",tile)
                    count+=1

                if count>=40:
                    break
        if count<10:
            idxs=[(y,x) for y in range(0,image.shape[0]//512) for x in range(0,image.shape[1]//512)]
            random.shuffle(idxs)
            for k, (y, x) in enumerate(idxs):
                tile=image[y*512:(y+1)*512,x*512:(x+1)*512,:]

                #bg_count=np.sum((tile.max(axis=2)-tile.min(axis=2))<20)
                bg_count=np.sum(np.ptp(tile,axis=2)<20)

                if ((bg_count/(512*512))<=0.75)&((bg_count/(512*512))>0.65):
                    cv2.imwrite(f"{out_path}/{name}/{x}_{y}.png",tile)
                    count+=1

                if count>=10:
                    break
```

#### Training tile:

**Step 1:**Using all tiles if bg_count/area less 0.5

**Step 2:**If tiles of WSI image<50,add ((bg_count/area) between 0.5-0.65)   tiles until there are 50 tiles.

**Step 3:**If tiles of WSI image<20,add ((bg_count/area) between 0.65-0.75)  tiles until there are 20 tiles.

## Model Training:

Only use WSI tiles. Randomly select 6 tiles from each image for training in every batch.
Loss Function: Binary Cross-Entropy (BCE)

**Step 1:** Normal Training

**Step 2:**
Utilize the results from Step 1 to generate auxiliary labels.If the predicted value for true label is greater than 0.3, set the auxiliary label to 1; otherwise, set it to 0.
Re-train the model without using the weights from Step 1.
Loss function: Label loss (BCE) + 0.3 * Auxiliary Label loss (BCE)
Learning rate: 2e-4

**Step 3:** Fine-tuning with Step 2 Weights
Further refine the model using the weights obtained from Step 2.
Loss function: Label loss (BCE) + 0.15 * Auxiliary Label loss (BCE)
Learning rate: 5e-5

#### Models with different backbone:
efficientnetb4,efficientnet_v2s,maxvit_tiny (The model settings of different backbones are slightly different.)

### WSI
Use model to predict tiles.

### Wsi tile ensemble:
```python
tile_df["prob"]=np.max(tile_df[["pred_0","pred_1","pred_2","pred_3","pred_4"]],axis=1)
tile_df["pred"]=np.argmax(tile_df[["pred_0","pred_1","pred_2","pred_3","pred_4"]].values,axis=1)
tile_df=tile_df[["image_id","pred","prob","aux"]].groupby(["image_id","pred"])[["prob","aux"]].mean().reset_index()
idx=tile_df.groupby(["image_id"])["prob"].idxmax()
wsi_df=tile_df.loc[idx1].reset_index(drop=True)
```

### Outliers(WSI):
 The predicted mean value of aux_label<0.5(The score is almost the same as not predict "Other",maybe+0.01)

## tma:

**Step1.**Crop tma
```python
def crop_tma(img):
    ks=min(min(img.shape[0],img.shape[1])//150,20)
    
    mask=(img.max(axis=2)-img.min(axis=2))>20
    kernel = np.ones((ks, ks),np.uint8)
    mask=cv2.erode(mask.astype(np.uint8),kernel)
    nonzero_pixels = np.column_stack(np.where(mask > 0))
    
    if (nonzero_pixels.size)<(img.size//60):
        return img
    else:
    
        min_y, min_x = np.min(nonzero_pixels, axis=0)
        max_y, max_x = np.max(nonzero_pixels, axis=0)

        return img[max(0,min_y-ks):max_y+ks+1,max(0,min_x-ks):max_x+ks+1,:]
```
**Step2.**Resize to 512*512(The size of tma *0.33*0.5~512, so we can directly resize to 512 to predict)
**Step3.**Using wsi training model to predict

### Outliers(tma): 
 The predicted value of aux_label <0.5 (compared to tma without predict "Other", public score +0.03, private score +0.06)

## Ensemble different models:
Voting(Compared with a single model, maybe only +0.01)

## Maybe not work:
segmentation
