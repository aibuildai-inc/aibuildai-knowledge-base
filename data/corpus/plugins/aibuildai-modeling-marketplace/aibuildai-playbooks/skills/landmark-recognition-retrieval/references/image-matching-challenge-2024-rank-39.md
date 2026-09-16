# [39th place] Superpoint+Lightglue & Keynet_Affnet_Hardnet_Adalam

Competition: image-matching-challenge-2024
Rank: #39
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510373

I would like to first thank our organizers and Kaggle staff for organizing this competition, authors of Hloc, Lightglue and pycolmap, my teammate @cody11null for his great work. Even though the preliminary standing doesn't meet our expectation, we have learnt a lot as a team. On behalf of the team, I also want to thank @maxchen303 for the [notebook](https://www.kaggle.com/code/maxchen303/imc2023-final-pub) he has published last year. It is the baseline with which we experiment, and I suppose a lot of teams owe him a big thank you!

P.S. Our team has experimented with lots of models and has gained some insights. Plus, I will be in Seattle area in person during the CVPR, and I really would like to be invited and present our work in workshop☺️ @oldufo

---

For an overview of Max's [baseline](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417045)

### Overview of our solution
[solution]

### Thoughts

The greatest impression our team has of IMC2024 is **randomness**. It is very hard to set up a convincing local pipeline, because colmap's randomness accounts for differences in scores, obscuring the effectiveness of the trivial trick. Moreover, trick that has zero risk of harming the result, such as unregistered image localization, multiple reconstructions don't boost the score significantly. 

Despite having great potential in camera position estimation, Dense matching is inherently not compatible with the reconstruction of the entire scene as mentioned in last year winning [solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407). We have tried quantization and some miscellaneous tricks so that view tracks may be established but they all failed. 

### Local Result

Average mAA of an ensemble of Keynet+Affnets+Hardnet+Adalam*: 23.5% ~ 24%
Average mAA of SP+LG*: 25.7% ~ 26%
Average mAA of LoFTR: < 15%
Average mAA of dense matchers + quantization: <10%

*Our experiment suggests that DoG + Affnet works the best among all, on a par with the ensemble in terms of accuracy. 

*higher score on average using SP+LG is attributed to much better result on `lizard` (more 10 additional images registered)  and slight boost on `pond`. Yet, it doesn't perform as well as Keynet+Affnets+Hardnet+Adalam in reconstructing architectures, like the rest of dataset. This also inspires us to reconstruct models once with and another time without SP+LG to squeeze some extra accuracy. 

### What works for us:
1. Choose a different and right resolution before sending to the model -   For Superpoint and Lightglue, the best resolution is 2000 when we use Hloc (We can confirm the boost only because score slight boost in every dataset). For Keynes-Affnet-Hardnet, we stack key points extracted from model based on 1024 and 1600. Essentially we have used same resolution for images smaller than 1024 since enlargement is set to false.

2. Image localization using Hloc  - For datasets such as lizard and pond, many images are not registered. We manage to restore 2 or 3 images in extra under the easiest threshold. 

3. Multi-time reconstruction:  We set up different thresholds to filter out image pairs that have too few matched points, and then reconstruct them. We estimate the score of each model using this formula: `score = num_images * num_3dpoints / projection_error`. This helps us to select the best model but it does enhance the score much on LB compared to local (~1%)

Incidentally, out best sub doesn't even use the later two tricks that locally work, suggesting again that being lucky is somehow the key 😬

### What doesn't work
1. Dense matching always ends up with 0 image registered. This is not surprising as keypoints in an image vary when that image is paired up with different images, so there are always only two 2D coordinates that can be projected to a 3D point and the model thus couldn't be robustly constructed. Regretfully, RoMa has quite great performance as I check the image pairs it produces manually. We have tried [detector-free sfm](https://github.com/zju3dv/DetectorFreeSfM) to handle this issue, but we aren't able to compile it on Kaggle, unfortunately. 

2. [Omniglue](https://github.com/google-research/omniglue) - I thought this will give a boost to the current SP+LG pipeline, but it doesn't. After trying different configurations, it still couldn't surpass LG on local datasets (approximately 30~10 more images are unregistered compared to LG). 

3.  DISK + LG - concating with keynet+affnet+hardnet key points lowers the score. Moreover, we also find that `kornia.feature` has a bug that prevents model to be loaded from local machine. @oldufo

```python
def modify_lightgluematcher():
    def new_init(self, feature_name: str = "disk", params = {}):
        super(KF.LightGlueMatcher, self).__init__(feature_name)
        self.feature_name = feature_name
        self.params = params
        self.matcher = KF.LightGlue(None, **params)
        
    KF.LightGlueMatcher.__init__ = new_init
```

### What we didn't do

1. Tricks on glass - it seems that higher resolution gives a better performance on glass in local testing, but we have never implemented it online. We have also made the assumption that camera position for glassware images in test set is similar to camera position of the one in train set, except that it's not in order. Yet, we didn't know how to restore the order... (hence thumb up to 1st place for their innovative approaches!)


If you have any question about our solution, you are welcome to comment below. ❤️
