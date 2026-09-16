# 14th Place Solution

Competition: blood-vessel-segmentation
Rank: #14
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475260

# Overview  
[overview]
* 2.5D segmentation model that inputs N consecutive slices stacked in ch direction and outputs corresponding Nch masks.  
* Input images are cropped to the kidney area only and then resized.  

# Pipeline Detail　　
## 1. Preprocess  
### 1-1. Normalization  
* A histogram of luminance values is calculated for the entire kidney and normalized based on minimum and maximum values.  
* Normalization based on maximum and minimum values per image unit could cause variations in the appearance of images, resulting in unnatural switching of inference results. To counteract this, normalization based on the luminance distribution of the entire kidney was employed.  
* The code is as follows.  
```python
def get_min_max_val(image_dir):
    img_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
    
    pixels = np.zeros((65536,), dtype=np.int64)
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        _pixels = np.bincount(img.flatten(), minlength=65536)
        pixels += _pixels

    bin = 1000
    hist = []
    bins = []
    for i in range(0, 65535+bin, bin):
        hist.append(pixels[i:i+bin].sum())
        bins.append(i)
    hist = np.array(hist)
    hist_rate = hist/hist.max()
    idxes = np.where(hist_rate>0.01)[0]
    min_idx = idxes[0]
    max_idx = idxes[-1]
    return bins[min_idx]-5000, bins[max_idx]+5000  

min_val, max_val = get_min_max_val(inference_img_dir)
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
img = img.astype('float32')
img = np.clip(img, min_value, max_value)
img = (img-min_value)/max_value
```

### 1-2. Crop  
* Obtain a rectangle of the kidney region using a segmentation model that infers a mask of the entire kidney.  
[crop]
* The kidney segmentation model used a single model of Unet (backbone: efficientnet_b4) and only kidney_1_dense was used as training data.  
* Cropping only the kidney region eliminates wasted areas in the image and greatly improves the accuracy of vessel segmentation.  
* During training, the height and width of the rectangle were stochastically increased or decreased by ±5% as part of augmentation.

### 1-3. Resize  
* Because of the strict masking requirements of this competition metric, it was important to resize the image to a larger image size.  
* In my solution, I trained the model by resizing the image to as large as GPU memory would allow, in the range of 1536~1920.  
* If the mask is resized by OpenCV's resize function and then resized back to the original size again, the mask pixels are shifted to the lower right, resulting in a significant loss of accuracy. Therefore, care should be taken in resizing.
  * In my solution, I used an affine transformation that simultaneously translates by 0.5 pixel and scales the image to prevent pixel misalignment.　　
  * Incidentally, this idea is strongly influenced by the contrail competition solution.  


## 2. Vessel Segmentation  
### Model  
* Model was Unet (using smp implementation), resnest14d, resnest50d, maxvit_tiny were used for backbone, and ensemble with equal weights was used as final sub.  
* Since I wanted to use depth direction information as well, we employed a 2.5D model that takes an input image consisting of n (5 or 7) consecutive slices stacked in the ch direction and outputs the corresponding n ch masks.  

### Data
* The data was kidney_1_dense as training data and kidney_3_dense as validation data. Some models used kidney_2 and Pseudo Labeled data for external data as training data.  

### Augmentation
* Use augmentation on rotation, flipping, and brightness (using the albumentations implementation).
* The shape-changing type augmentaion (e.g., Distortion) was tried but was not used because it worsens the accuracy of both cv/lb.

### Inference
* Inference in each view in XY, XZ, and ZY directions.  
* The accuracy of both cv/lb was increased by inputting a larger size than the image size used for training during inference.  
* The threshold was determined based on CV and used 0.25.  

### Summary  
The final scores are as follows.  
| Model| N(ch) | train_data | validation_data | input_size(train) | input_size(inference) | CV | Public | Private |
| ---  | --- | ---  | ---  | ---  | ---  |  --- | ---  | --- |
| resnest14d | 7 | kidney_1_dense | kidney_3_dense | 1920x1920 | 2304x2304 | 0.909 | 0.835 | 0.659 |
| resnest50d | 5 | kideny_1_dense | kidney_3_dense | 1536x1536 | 1920x1920 | 0.903 | 0.818 | 0.599 |
| maxvit_tiny| 7 | kidney_1_dense, kidney_2(pseudo label), extra_data(pseudo label) | kidney_3_dense | 1536x1536 | 2048x2048 | 0.901 | 0.810 | 0.623 |
| Ensemble | - | - | - | - | - | 0.913 | 0.824 | 0.645 |
