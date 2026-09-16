# [Viettel.DGD] Train4Ever 7th place Solution

Competition: sartorius-cell-instance-segmentation
Rank: #7
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298002

Hello Kagglers, I am Truong Bui Nhat from team Train4Ever. First, I would like to thank the host for providing such an interesting instance segmentation challenge, then all the other teams for making a nice race.
Shout out for @tungvs  @namgalielei @damtrongtuyen @duykhanh99.

This our brief solution for Sartorius - Cell Instance Segmentation competition. 
Our solution is a flow with 5 stages back to back:
1. Train baseline models
2. Add LiveCell Shsy5y data, clean data, retrain and finetune models and use NMS by mask
3. Pseudo labeling with potential models
4. Pseudo labeling round 2 and Ensemble
5. Post processing model

Stage 1, we tried a lot of models with original data:
        - MaskRCNN Cascade ResneSt200 (with LIVE CELL pretrain)
	- PointRend 
	- GCNet
	- CellPose
	- MaskRCNN Swin
	- MaskRCNN Cascade Swin
	- SCNet
	- Query Instance
	- HTC
	- CBNet V2
Note that we froze batch norm layers in backbone ResneSt200 and FPN because the training batch size was only 2. We thought it helped to keep the batch statistics unchanged, thus stabilizing the training.
After validating on local valid set + submit to Public leaderboard we selected the below candidates for stage 2:
	- MaskRCNN Cascade ResneSt200 (with LIVE CELL pretrain): 0.329
	- PointRend: 0.317
	- MaskRCNN SWin: 0.292
	- MaskRCNN Swin: 0.291
	- GCNet: 0.304
	- CellPose: 0.314
(All are reported on public leaderboard scores)

Stage 2, we added clean data, retrained stage 1 candidates and finetuned:
	Add LiveCell Shsy5y data:
We used all the train, valid and test set of Shsy5y type from LiveCell data as additional training data for Shsy5y 
	Our cleaning method:
		1. Remove LIVE CELL Shsy5y images with high FN with IOU 0.5 on training set (After training a model and perform error analysis)
		2. Remove cort images with duplicate annotations 
		3. Sanity check and remove cort images that we feel missing annotations
	More fine tuning:
		1. Image size increased from 800 (shortest edge) to 1024
		2. Unfreeze all the backbone (default option of Detectron2 is freezing at the second block)
	Our best model performance on Public leaderboard after stage 2:
		- MaskRCNN ResNeSt200: 0.336	

Stage 3, we generated pseudo labels on the Semi Supervised dataset and re-trained on new data
First, both MaskRCNN ResNeSt200 and PointRend predicted the instances, then a simple ensembling technique was used to combine their predictions. The ensembling technique was to match IOU by mask to create different clusters, then within each cluster, used pixel voting to determine which pixel was kept, which was filtered.
Then MaskRCNN ResNeSt200, PointRend and MaskRCNN Swin trained on this new pseudo data + original data + LiveCell Shys5y data.
 
At this stage MaskRCNN ResNeSt200 could achieve 0.338 on LB. 

Stage 4, we generated pseudo labels on the Semi Supervised dataset with MaskRCNN ResNeSt200, PointRend and MaskRCNN Swin coming from stage 3 and used the simple ensembling technique similar to stage 3 to combine the predictions.
The new pseudo prediction was added to original data + LiveCell Shsy5y data and we retrained MaskRCNN ResNeSt200 model.

At this stage MaskRCNN ResNeSt200 could achieve 0.340 on LB. 

Then we developed a new ensemble technique to ensemble 2 MaskRCNN ResNeSt200 models trained on round 1 pseudo and round 2 pseudo data.

The new ensembling method described with the image below:

[ensembling method]

Those only boosted less than 0.1% (still 0.340), we still thought an ensembling submission would be better if a shake-up happens.

Stage 5, we trained a machine learning model with a view to filtering FPs. 
We extract some features from the prediction of stage 4 such as: basic features (instance confidence score, mask pixel scores, box area, mask area, location, size, mean/median pixel values on the original image), morphology features (rotation_angle, solidity, convex hull area, perimeter, …), neighboring features (overlap amount, overlap neighbor count, distance to top k neighbors, neighbor count within a circular area, …). The total number of features was 130.  
Model: CatBoost
Validation: We splitted the valid set (used for validating at the above 4 stages) into 2 half and used the first one to train CatBoost, the second one to validate and hyperparam tuning.
Hyperparam tuning method: Using hyperopt lib with objective function was the competition metrics on the second half of the valid set. 

This stage gave us a score of 0.341 on the leaderboard.

Our final submission notebooks:
https://www.kaggle.com/namgalielei/maskrcnn-v21-v16-ensemble-stage2catboost/notebook

	
This is the end of our solution, stay safe and enjoy the New Year’s Eve everyone.
