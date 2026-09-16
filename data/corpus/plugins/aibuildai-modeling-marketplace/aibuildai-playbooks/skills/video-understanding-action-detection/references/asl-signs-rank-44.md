# 44th Place Silver - How To Improve Best Public Notebook

Competition: asl-signs
Rank: #44
Source: https://www.kaggle.com/c/asl-signs/discussion/406302

Thank you Kaggle, Kagglers, PopSign, and Partners for a fun competition! Public notebooks in this competition are very strong! The host said that they need `LB >= 0.6` to be helpful and the best public notebook scores `LB = 0.73`! Great job Kagglers!

I joined this competition one week ago, so I didn't have much time to build my own model. Instead, i read the discussions and public notebooks, and attempted to improve the best public notebook. I boosted the best public notebook from public `LB = 0.73` to `LB = 0.77` and achieved 44th place Silver.

# Praise for Best Public Notebooks
I am very impressed with the shared public notebooks
* Great Transformer model architecture
* Selected which 66 Landmarks out of 543 Landmarks are important (hands, lips, pose)
* Great preprocessing to convert videos of variable length to small fixed length
* Accurate local validation scheme which estimates LB score

# Improvements to Best Public Notebook
After reading the best public notebook, I made the following 11 changes to boost CV and LB `+0.03` from `LB = 0.73` to `LB = 0.76`:

* Train 1 model => Train 4 models
* Add Time Scale augmentation
* Ensemble and apply TFLite FP16 quantization
* Change the following parameters:
* INPUT_SIZE, 64 => 12
* BATCH_ALL_SIGNS_N, 4 => 1
* N_EPOCHS, 250 => 120
* LANDMARK_UNITS, 384 => 224
* UNITS, 512 => 376
* NUM_BLOCKS, 2 => 3
* MLP_RATIO, 4 => 3
* MLP_DROPOUT_RATIO, 0.40 => 0.30
* remove random frame masking

# Published Code
I published my code [here][1] for those curious to learn "How To Improve Best Public Notebook" from `LB = 0.73` to `LB = 0.76`

# My Thought Process
Most Kagglers are probably curious why and how I discovered these modifications. So let me share my thinking. First, the easiest way to boost CV LB for NN is to ensemble NN with multiple copies of itself (trained with different seeds). So the easiest way to boost public notebook CV LB is to make the model smaller and make the model faster (because this competition has size and time constraints). Then we can include multiple copies of the NN during inference. And each NN we will train with 100% train data.

The first thing i did was reduce `LANDMARK_UNITS`, `UNITS`, and `MLP_RATIO` to reduce the model size. I slowly reduced these parameters to see how small I could go without reducing CV score. It is important to reduce the model to less than 5M (i.e. 5 million) parameters because then we can infer 4 copies of the model within the 40MB size restriction when using FP16. With 40MB size limit, the maximum number of parameters during inference with FP16 is 20M. I also tried less than 4M parameters and 3.3M parameters which allows 5 copies and 6 copies respectively (when using FP16 quantization) 

Second, I made the transformer deeper with more transformer blocks. I increased blocks from 2 to 3 which boosted the CV and LB about `+0.01` or `'0.02`. Reducing the `MLP_RATIO` from 4 to 3 significantly reduced the number of model parameters without affecting CV LB so this gave me extra parameters to allow increasing the transformer blocks.

Next, I reduced `INPUT_SIZE` to speed up the model to make it infer (and train) faster. The parameter `INPUT_SIZE` is the transformer sequence length. This competition has a inference time constraint of 1 hour (in addition to a 40MB size constraint). Reducing `INPUT_SIZE` will decrease inference time (and does not affect model size). With sequence length 12, we can easily infer 5+ models under 1 hour.

At this point, i got lucky. Reducing `INPUT_SIZE` actually boosted the CV and LB by `+0.01` or `+0.02` or so. After this I tuned the learning rate, batch size, and learning schedule and found `N_EPOCHS = 120` and `BATCH_ALL_SIGNS_N = 1` to be best. Many Kagglers overlook the fact that changing batch size can make a big difference for models. We should always try 0.25x, 0.5x, 2x, 4x batch size and change the learning rate for those experiments to be 0.25x, 0.5x, 2x, 4x respectively. Experiments demonstrate that `Batch Size = 256` is better than the original `Batch Size = 1024`. This helped `+0.0005` or so.

Lastly NN always benefit from more data and data augmentation. I tried external data and many different data augmentation. Using external data did not help me. Regarding augmentation i only got benefit from `frame drop out` and `time scale augmentation`. This helped about `+0.0005` or so. I also tried rotation, scale, shift but this did not help. Flip was not needed since preprocessing normalized all videos to be left handed.

[1]: https://www.kaggle.com/code/cdeotte/improve-best-public-notebook-lb-0-76
