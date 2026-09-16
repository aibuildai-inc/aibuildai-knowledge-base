# 11th place solution: LightGBM

Competition: predict-ai-model-runtime
Rank: #11
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456092

Thanks to the host and congratulations to the winners.
I'm glad to win my 4th gold medal on this competition.
I did not use GNN, but used LightGBM with feature engineering.
The reason why I chose LightGBM is that I thought that the runtime of the model in a single TPU is equal to the sum of the runtime of each node, and learning the aggregated statistics from the graph would be sufficient to achieve good score without learning the graph structures.

## Layout

### Features

I extracted the following features from the graph.
- Number of nodes of each type for conv/dot/reshape when using the following classification method
	- Conv: If any one of the input/output number of dimensions or config values, or the 93~106th values of node_feat is different, it is of a different type
	- Dot: If any one of the input/output number of dimensions or config values, or the index features of dot operation (extracted from the .pb file on its own) is different, then it is of a different type
	- Reshape: If any one of the input/output number of dimensions or config values, or the inconsistency on the element products in a set of a certain dimension, then it is of a different type
- Number of times the element is copied
	- Determine if a copy is needed by comparing the layout of each configurable node and the configurable nodes connected to it.
- Counting binning the size of the 1st/2nd minor dimension
- Sum of padding generated at each configurable node
- (only default) Occurrence rate of config for each node relative to the total data for that model
	- Because genetic algorithms are used in the search, the more frequently a config pattern appears, the faster the config runtime will tend to be.

### Models

I trained the pointwise and pairwise models.

- Pointwise LightGBM
	- Target: Normalized rankings(0~1)
	- Loss: MAE Loss
	- Public LB: 0.715 (Private LB: 0.680)
- Pairwise LightGBM
	- Loss: Binary
	- Randomly select the same number of pairs as the number of configurations and generate train/valid.
	- Inference on test data predicts for all pairs of 1000^2 and then sort configs using the sum of the predictions.
	- Input features are as follows: 
		- The features for one config of the pair
		- The difference between the features of the two configs
	- Public LB: 0.728 (Private LB:0.701)

I used the same train/valid given by the host. (Failure to devise a better CV may have been the cause of the shake down)

## Tile

### Features

- Config features
- Node features averaged over all nodes
### Models

I trained the pointwise LightGBM model.
- Target: Normalized rankings(0~1)
- Loss: MAE Loss
- Public LB(only tile): 0.198 (Private LB(only tile): 0.195)
