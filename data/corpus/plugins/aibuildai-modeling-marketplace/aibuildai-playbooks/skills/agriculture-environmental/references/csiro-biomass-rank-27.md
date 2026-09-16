# 27th Place Solution

Competition: csiro-biomass
Rank: #27
Source: https://www.kaggle.com/c/csiro-biomass/writeups/27th-place-solution

First of all, Thanks to the host for organizing such interesting and fun competitions. This was my first serious participation, and I am very happy my solution was actually decent enough to rank 27th. The experiences and knowledge gained during this particiaption has been extremely mind-opening. So thank you for broadening our knowledge horizons. This is also my first time using cross-validation and ensembling strategies.

# ☘️ EDA
After analysing the data, I noted the following observation
* The biomass targets follow clear additive relationships `Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g` and `GDM_g = Dry_Green_g + Dry_Clover_g` 
* Target-wise distribution characteristics, 
    * `Dry_Total_g` and `GDM_g ` were the least noisy, and only these two were roughly Gaussian.
    * `Dry_Green_g` had a power law distribution with a shorter head.
    * `Dry_Dead_g` and `Dry_Clover_g` also had a power law distribution with much taller heads and long tails
* Species targets were noisy; some species were visually very similar (especially subspecies)
* The 3 auxiliary targets shown strong correlation with the values, for example
    * If there is no clover or its subspecies present, then `Dry_Clover_g` was often zero.
    * If height was high, the targets also had higher mass, except in the case of the `lucerne`, where the mass values were not scaled in the same proportions

# ☘️ Data Preprocessing and Strategy
* For biomass targets since `Dry_Dead_g` and `Dry_Clover_g` were noisy and had a lot of zeros, I decided to predict `Dry_Total_g`, `GDM_g`, and `Dry_Green_g` and derive rest two from them. I used log norm transformation for these biomass targets.

    ```python
    means = torch.tensor([3.6171854219925024, 3.217874424165861, 2.832479241567281])
    stds = torch.tensor([0.5573042636308412, 0.8185179833598041, 1.2134970356508157])
    normliazed = (torch.log(data + 1) - means) / stds
    data = torch.exp((normliazed * stds) + means) - 1
    ```
* For auxiliary regression targets, I did the following pre-processing
    * Auxiliary target `Height_Ave_cm` is also normalized and denormalized in the same way with a mean of `1.7949544186979798` and std of `0.7399843934368923`. As the height also have power law distribution with a tall head and long tails
    * Auxiliary target `Pre_GSHH_NDVI` was not transformed and used as it is
    * For Species I preprocessed and grouped similar-looking species, there was also a capitalization mismatch whic were resolved. I reduced 15 categories to 8 categories. I used the following function. This grouping gave a ~0.03 R² boost.

    ```
    def reduce_species(species):
        new_species = []
        for one_species in species.split("_"):
            if one_species in ("BarleyGrass", "Barleygrass", "Bromegrass", "SilverGrass", "SpearGrass"):
                new_species.append("grass")
            elif one_species in ("SubcloverDalkeith", "SubcloverLosa", "Clover"):
                new_species.append("clover")
            elif one_species in ("Capeweed", "CrumbWeed"):
                new_species.append("weed")
            elif one_species == "Mixed":
                new_species.append("ryegrass")
            else:
                new_species.append(one_species.lower())
        return list(set(new_species))
    ```


# ☘️ Cross Folds Creation
As the dataset was very small, I tried to create data splits where each train se set would see the entire spectrum of values. But this was not happening in single cross-folds as I was always leaking dates, state in train-val splits. So I decided to create two cross validations splits.
* 5 Cross Folds -> I called this `clean_folds`, where I stratified across `State` and `Sampling_Date`, using the following code 

    ```
    df = pd.read_csv('./datasets/csiro-biomass/train_unwrapped.csv')
    df['session_id'] = df['State'].astype(str) + "_" + df['Sampling_Date'].astype(str)
    df['biomass_bin'] = pd.qcut(df['Dry_Total_g'], q=5, labels=False)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=False)
    ```
* 3 Cross Folds -> Here I simply sorted by `Dry_Total_g` in descending order and created 3 groups so each train set will see a full spectrum of target values. Only 3 folds were used to avoid overly similar train/val splits.

# ☘️ Augmentations
### First of all, what worked
* Tested with a combination of image resolution `384, 512, 768, 1024` image size with patch size of `2x1, 4x2`. Found image size of `768x768` with two patches `left and right` worked the best, so that's how the rest of the experiments were conducted.
* For augmentation, I used some global transforms, which were applied on full image. And local transforms, which were applied patchwise. I wrote a custom compose class for this. I also experimented with switching to a weaker pipeline in the middle of training. At worst, it did nothing, and at best, gave a little bump in metrics, so I used it for the rest of the runs by default, switching the pipeline in the last 5-10 epochs

    ```
    NORM_SETTING = {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    STRONG_AUG_TRANSFORMS = {
        "type": "albu",
        "compose_cls": "tiled_compose",
        "global_transforms": [
            {
                "type": "OneOf", "p": 0.5,
                "transforms": [
                    {"type": "AutoContrast", "method": "cdf", "p": 0.5},
                    {"type": "RandomBrightnessContrast", "brightness_limit": 0.3, "contrast_limit": 0.2, "p": 0.5},
                    {"type": "RandomToneCurve", "scale": 0.3, "p": 0.3},
                    {"type": "HueSaturationValue", "hue_shift_limit": 20, "sat_shift_limit": 30, "val_shift_limit": 20, "p": 0.5},
                    {"type": "PlanckianJitter", "mode": "blackbody", "temperature_limit": [4000, 8000], "sampling_method": "gaussian", "p": 0.5},
                    {"type": "CLAHE", "p": 0.5},~~~~
                ],
            },
            {"type": "ShiftScaleRotate", "shift_limit": [-0.1, 0.1], "scale_limit": [-0.1, 0.1], "rotate_limit": [10, 10], "border_mode": 4, "p": 0.5},
            {"type": "GaussianBlur", "blur_limit": 3, "sigma_limit": 1, "p": 0.5},
            {
                "type": "OneOf", "p": 0.5,
                "transforms": [
                    {"type": "GridDistortion", "distort_limit": 0.5, "p": 1.0},
                    {"type": "OpticalDistortion", "p": 1.0},
                ],
            },
        ],
        "local_transforms": [
            {"type": "D4", "p": 1.0},
            {
                "type": "OneOf", "p": 0.5,
                "transforms": [
                    {"type": "Sharpen", "alpha": [0.1, 0.5], "lightness": [1.0, 1.0], "method": "kernel", "p": 0.8},
                    {"type": "Emboss", "alpha": [0.4, 0.5], "strength": [0.3, 0.7], "p": 0.8},
                ],
            },
        ],
        "cvt_transforms": [NORM_SETTING, {"type": "ToTensorV2"}],
        "n_patch_h": 1, "n_patch_w": 2,
        "target_size": [IMAGE_SIZE, IMAGE_SIZE * 2]
    }
    ```


# ☘️ Architectural Overview
* Used DinoV3 ViT large as my backbone, as almost everyone else in this competition did. 
* The image patches were passed to a shared backbone, and the features of shape [N, dim] from both patches were concatenated.
* First, the height, NDVI, and species were predicted from the backbone features using a simple 2-layer MLP. The same backbone features were passed into an MLP projection network, which produced an embedding. 
* Then, the NDVI, height, and sigmoid-activated species logits were concatenated and then passed to a FiLM layer, which then modulated the embedding produced by the above MLP projection network. Earlier, I was simply concatenating them, but due to the size of the embedding, it dominated. I was looking for better options, so I thought to predict the scale shift parameter and transfrom embedding that way. I ended up designing the FiLM layer. (Fun fact, I initially thought, I discovered a novel idea, as this took me from 0.68 to 0.72 in public LB. Later discovered the FiLM paper 😅)
* Finally, I predicted 3 biomass targets at once from the modulated embeddings

# ☘️ Loss & Optimization
* For loss, I used a weighted SmoothL1 loss for biomass targets and a SmoothL1 loss for height and NDVI targets. 
* For the species head, I used a simple BCE loss with the following weights for each category
    ```
    SPECIES = ["clover", "ryegrass", "phalaris", "grass", "fescue", "lucerne", "weed", "whiteclover"]
    SPECIES_WEIGHTS = [1.68, 1.86, 2.76, 3.59, 4.09, 5.01, 5.89, 7.08]  # roots sqrt(N / Nc)
    ```
* The above 3 losses were combined with weights `{"biomass": 3.0, "aux": 1.0, "species": 0.2}`
* For training, I used AdamW with lr of 1e-5 and weight decay of 0.3 (I know it's high, but it regularizes the model ), did warmup for 5 epochs, then reduced the lr for the rest of the training using cosine annealing
* The training was unstable, so I used EMA and gradient clipping to stabilize the training
* All three heads were trained together on 3CV and 5CV. These two were my final submissions.

# ☘️ What did not Work
* As the dataset provided in competetion was very small, I found a large dataset (was ~270GB with ~40k images) which I have processed and uploaded to Kaggle. It was not much of a help, though. [here](https://www.kaggle.com/datasets/pushpakbhoge/grass-clover-datasets)
* **DinoV2 Pretraining**: Fine-tuned DinoV2 (ViT-small) on the largely unlabelled data for 100 epochs. It failed to beat the original weights despite promising attention maps. Halted due to compute limits, though this remains a potential avenue for exploration.
* **Segmentation**: Trained a 16-class model on synthetic data to create guided attention maps for the downstream CNN [code](https://github.com/PushpakBhoge/csiro_biomass_regression/blob/main/components/bricks.py#L73).
* **CNN Architecture**: Initially used ConvNeXt-atto (ImageNet1k pretrained), theorizing that a smaller model suited the dataset size. Integrated the segmentation maps for attention and tested various pooling methods.
* **CNN Outcome**: Despite extensive tuning, the CNN could not outperform ViT. Realizing late that superior features were critical, I tried ConvNeXt-Base with DinoV3 weights but only achieved 0.65 Public LB, suggesting ConvNeXt features were less effective than ViT-base here. These weights for ConvNeXt were distilled from ViT, so maybe that has something to do with it
* **Meta-Averaging**: Tried meta-averaging auxiliary predictions before biomass inference, neutral effect.
* **Unfreezing-backbone**: Tried scaled lr across different backbone stages, and training small model from scratch did not work
* **Biomass Summation**: Attempted to approximate density maps by summing patch biomass. Summing raw channels caused exploding gradients, while 3x3 average pooling stabilized training but yielded no gains.
* **Spatial Pooling**: Implemented a convolutional spatial pooling layer (reducing 24x24 features to 1x1) followed by MLPs. Summing these outputs improved Public LB by 0.01, peaking at 0.59 [code](https://github.com/PushpakBhoge/csiro_biomass_regression/blob/main/components/bricks.py#L217).
* **FPN & Multi-Stage Heads**: Tested fusing backbone features at varying resolutions (FPN) and attaching heads directly to Stage 2 and 3 outputs. Neither approach improved the score.


I have written quite extensive code for this. You can find the code on my [GitHub repository](https://github.com/PushpakBhoge/csiro_biomass_regression)
This is my [submission notebook](https://www.kaggle.com/code/pushpakbhoge/csiro-image2biomas-submission-notebook) I am planning to release an exhaustively detailed article about this, which will take a week or two.

Finally, Why were CNNs consistently inferior to ViTs in this competition? I’d love to hear insights from others.
