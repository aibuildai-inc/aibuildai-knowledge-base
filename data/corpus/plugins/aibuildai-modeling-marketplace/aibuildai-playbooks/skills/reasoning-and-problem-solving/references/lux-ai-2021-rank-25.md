# Some useful and useless ideas(maybe silver solution- 25th place)

Competition: lux-ai-2021
Rank: #25
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/293836

🎉🎉🎉Congrats to every one who fight for Lux, wish you all good results!🎉🎉🎉  

Here I want to share our team's (probably) silver medal solution.   

On the whole, we divide our work into three parts: imitation learning, rules, and reinforcement learning. I am mainly responsible for the imitation learning part. In order to make it easier for Chinese friends to browse, I also wrote a Chinese version [here](https://www.kaggle.com/c/lux-ai-2021/discussion/293838) (which has the same content).

**Note:**The following method are some of the methods I tried in the competition, some of which have been significantly improved, and some of them were not used in the final submission.

#### Feature Engineering
First of all, I would like to thank [Lux AI with Imitation Learning](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning)for sharing the great imitation learning baseline。My feature engineering is based on the baseline.  
I added the following features to the program:

##### Independent Features
- resource map
- Distance map:distance to each cell of the map (you can set a limit size)
- Last action (5 layers - n,s,w,e,bcity)

- Action of teammate

- [Manhattan distance to nearest friendly city](https://www.kaggle.com/c/halite/discussion/183312)

- Manhattan distance to nearest enemy city

- [flood-fill](https://www.kaggle.com/c/hungry-geese/discussion/255931)

- source_map/distance_map **

- ...

  
##### Global Features

- Different layer for each type of resource rather than combine together.

- Number of opponent/friendly unit

- Number of total source remain

- Source block: we deive source into different clusters, and extract feature by block。

- ....

If you want to speed up feature extraction, please calculate the global features only once in each step(rather than once for each unit).  
In addition, due to time constraints, some of the above features can only be used when the number of agents is small.

#### Model Architecture
- ResidualBlock:
  ​	Convolution kernels of different sizes are used to extract features of different scales, and a fully connected layer is used to reduce the problem that the convolution layer will confuse accurate position information.[resource](https://www.kaggle.com/c/hungry-geese/discussion/263279)
>   ​    self.conv11 = BasicConv2d(filters, filters, (1, 1), True)
  ​    self.conv33_1 = BasicConv2d(filters, filters, (3, 3), True)
  ​    self.conv33_2 = BasicConv2d(filters *3, filters, (3, 3), True)
  ​    self.fc1 = nn.Linear(32, 32, *bias*=False) 

- Unet:
  I first saw unet applied to decision-making games in [Imitation Learning by Semantic Segmentation for Halite IV Competition (8th Gold solution)](https://www.kaggle.com/c/halite/discussion/183312 )。Since this method is very interesting (saving time and predicting all actions with only once inference), I start to achive it after seeing it. However, many problems have been encountered, and the final effect is not as good as the independent learning agent method. Later, I saw the [topic](https://www.kaggle.com/c/lux-ai-2021/discussion/289540) shared by @zaharch,which reminded me that unet can also achieve good results in lux. Thank very much for his prompt response.  

 Although in the end I failed to fully implement unet in lux, I also have some suggestions for kagglers who also want to implement this method:

- When you try to rotate all move_actions to the same direction during training, please check your rotation function carefully and debug with visualize.
- If you use the np.rotate function, please note that the rotation here is to counterclockwise rotation matrix, but because the direction of the matrix and the real map are opposite, that is, the map is rotated clockwise.
- If you rotate the input, you must remember to rotate the label and mask as well.
- If you rotate the input during prediction, you must remember to rotate the output action back before obtaining the weighted policy!
- If your map is expanded, don't forget to add shift.

    
#### Speciall Skills

- **TTA**：TTA is a common method in visual competitions, that is, when predicting, the input rotation is rotated three times, and the results are integrated. This can improve the robustness of model prediction. However, TTA is prone to encounter timeout problems in the model of independent prediction by the agent. Therefore, when predicting, we will calculate the time for each unit to perform tta and estimate the time consumed by all agents tta. If the total time exceeds 3 seconds, tta is not used.

- **Data augmentation**: The training data can be expanded eight times by rotating and flipping the training data. As I said above, you must be careful not to make a mistake when rotate the obs and label.

- **Data balance**: In this game, there are only a small number of agents in the initial stage, so only a small amount of data will be generated. At the end of the stage, hundreds of agents will produce actions in one frame. In this case, it is difficult for the model to learn a good initial strategy. I adopted a method of varying the sampling rate, using a higher retention rate at the beginning step, and gradually reducing the retention rate. Therefore, the model will pay more attention to the initial strategy learning.`if random.random()<=(40/(i+1)):`

- **Segmented learning**: It is essentially the same as the previous method. The strategy for the first place has obvious stages. We divide the data set into different stages to train different models, and the model in the initial stage can use much larger and deeper model , because the model timeout pressure will be much less.

- **Data screening**: Keep episodes with higher scores. Note that this is not a submission with a higher score (although this is definitely something to be done), but an episode, because the strength of the opponent encountered by the submission in the initial stage of submission is too low, which may actually be detrimental to the learning of the model.

### Some methods that feel meaningful but have not had time to implement:
- Hungarian algorithm
- Fusion of Unet model and independent model
- Use Unet to learn city strategies


### Some suggestions for novices:
- If you are suffering from not being able to get a medal, please check the discussion area carefully every time the game is played. In most competitions on kaggle, ideas or hints that get silver or even gold medals are basically hidden in every corner of the discussion area. The process of finding them is like digging for treasure. If you can't find it in the game, then go to the kaggle competition to share. This is why I like kaggle - because there are so many like-minded colleagues who love to share. 😄
- If the game has restrictions on model size or running time, try to squeeze this restriction as much as possible.

Maybe my sharing is not enough to provide as much enlightenment as other people's sharing, but I also want to try to give back to this community, after all, I have learned a lot from here.

Part of the improvement in our final plan also comes from the coding of the rules, including defense and offense, personnel allocation, and city strategy. My teammate @amimibear will also share it later.
