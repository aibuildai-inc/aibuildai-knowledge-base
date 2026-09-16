# 3rd Place Solution: Transformer+GRU

Competition: tlvmc-parkinsons-freezing-gait-prediction
Rank: #3
Source: https://www.kaggle.com/c/tlvmc-parkinsons-freezing-gait-prediction/discussion/417717

The first and second place solution have much better fits to the individual defog/tdcs datasets;  this solution fit one model for both.

Architectures: Deberta/VisionTransformer/VisionTransformerRelPos -> LSTM/GRU
(typically 2-4 layers, with single-layer RNN)

Patch sizes are 7-13, with sequences of 192-384 patches.

All sequences have heavy augmentation--stretching, cropping, ablation, accumulated Gaussian noise, etc.

--

See data.py and model.py for details:

Inference Code: https://www.kaggle.com/code/stochoshi/fork-of-walk3
Additional Code: https://www.kaggle.com/datasets/stochoshi/walkdata4
