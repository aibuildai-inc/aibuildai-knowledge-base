# 8th place novel solution

Competition: quickdraw-doodle-recognition
Rank: #8
Source: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73967

Please pardon the quality of the following, but I’ve rushed it out. I will plan to release a detailed blog post/paper at a later date. My approach is novel, complex, and a challenge to communicate, but I wanted to get this information out while everyone was still interested!

# Key Insight
I leveraged a principal lesson in Deep Learning: Let the network learn the features rather than hand crafting them. The standard approach is to rasterize the drawings, crudely encode time as colour, and then pass to a pre-trained RGB ResNet. Instead, I implement a differentiable, trainable module to do this. Furthermore, I replace the standard nn.Conv2d(3, 64, kernel_size=7, stride=2, ...) and 2x2 MaxPool at the start of a ResNet with the same module.

High Level Visual:
![enter image description here][1]

Key benefits:

- Deep features are computed from the time component rather than some RGB hack

- Deep features also computed from stroke data, leveraging the native format before “discarding” it and switching to images. Access to connectedness information.

- Uses efficiency of convolutions on grids where a pure sequence or point cloud model fails to

# Implementation
Enormous amount of details that I will cover later.

Detailed visual #1:
![enter image description here][2]

- Begin with strokes as defined by a series of points.

- Difference the x’s and y’s to get dx, dy segment vectors. Average the ts.

- Now we have strokes as defined by a series of segments.

- Process strokes with a sequence module to generate 32 features.

- Unroll those features with a window 2 convolution to generate 64 features per original point.

Detailed visual #2:
![enter image description here][3]

- Draw points into a 32x32x64 image as per the diagram.

- When points collide their feature vectors are averaged. I think I would have max-pooled if I could have implemented it.

- At this stage in the network a 32x32 image could be thought of as equivalent to having started a normal image ResNet at 256x256.

# Sequence Module

    Conv1d(3,  32, kernel_size=3, stride=1, padding=1, dilation=1)
    BatchNorm1d(32)
    ReLU(inplace=True)
    Conv1d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2)
    BatchNorm1d(32)
    ReLU(inplace=True)
    Conv1d(32, 32, kernel_size=3, stride=1, padding=4, dilation=4)
    BatchNorm1d(32)
    ReLU(inplace=True)
    Conv1d(32, 32, kernel_size=3, stride=1, padding=8, dilation=8)
    BatchNorm1d(32)
    ReLU(inplace=True)
    Conv1d(32, 64, kernel_size=2, stride=1, padding=(1,0))


# Rasterization Module

    from apex import amp
    import torch
    from torch.autograd import Function
    
    class PointsToImage(Function):
       @staticmethod
       @amp.float_function
       def forward(ctx, i, v):
           device = i.device
           batch_size, _, num_input_points = i.size()
           feature_size = v.size()[2]
    
           batch_idx = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, num_input_points).view(-1)
           idx_full = torch.cat([batch_idx.unsqueeze(0), i.permute(1, 0, 2).contiguous().view(2, -1)], dim=0)
    
           v_full = v.contiguous().view(batch_size * num_input_points, feature_size)
           mat_sparse = torch.cuda.sparse.FloatTensor(idx_full, v_full)
           mat_dense = mat_sparse.to_dense()
    
           ones_full = torch.ones(v_full.size(), device=device)
           mat_sparse_count = torch.sparse.FloatTensor(idx_full, ones_full)
           mat_dense_count = mat_sparse_count.to_dense()
    
           ctx.save_for_backward(idx_full, mat_dense_count)
    
           return mat_dense / torch.clamp(mat_dense_count, 1, 1e4)
    
       @staticmethod
       @amp.float_function
       def backward(ctx, grad_output):
           idx_full, mat_dense_count = ctx.saved_tensors
           grad_i = grad_v = None
    
           batch_size, _, _, feature_size = grad_output.size()
    
           if ctx.needs_input_grad[0]:
               raise Exception("Indices aren't differentiable.")
           if ctx.needs_input_grad[1]:
               grad = grad_output[idx_full[0], idx_full[1], idx_full[2]]
               coef = mat_dense_count[idx_full[0], idx_full[1], idx_full[2]]
               grad_v = grad / coef
               grad_v = grad_v.view(batch_size, -1, feature_size)
    
           if isinstance(grad_output, torch.cuda.FloatTensor):
               return grad_i, grad_v
           else:
               return grad_i, grad_v.half()
    
    points_to_image = PointsToImage.apply


Various other details:

- Packing strokes of varying length into tensors of fixed size in order to do 1D CNNs is a non-trivial thing to do and beyond the scope of this post

- Pytorch 0.4.1

- Used NVIDIA’s apex amp (https://github.com/NVIDIA/apex/tree/master/apex/amp) to train exclusively at half precision

- Trained on all the data

- Raw not simplified

- LMDB for memory mapped data

- Adam optimizer

- Models typically took about 2.5-3 days to converge on a system with a 1080Ti and a Titan

- Used pre-trained imagenet weights

- Froze those weights and only trained my additional modules for the first 1k-5k iterations

- Held out 50k examples for validation during training

- Held out 1 million examples for blending

- Probably never completed even 2 complete passes through all the data during training. Convergence came first.

- Used gradient accumulation via multiple backwards calls in Pytorch to finish training at huge batch sizes

# Results

SEResNeXt50 32x4d at core.
Local validation:

- Acc@1: 0.8648

- Mapk3: 0.9073

- CE Loss: 0.5092

- Public LB: 0.94781

- Private LB: 0.94915

Best ensemble:

Six best models as measured by local CE Loss.

- 2 x SEResNeXt50 at core

- 3 x SEResNeXt101 at core

- 1 x ResNet34 at core

- Weighted arithmetic mean of probabilities. Weights = (1/loss)**24

Local validation:

- Acc#1: 0.8683

- Mapk3: 0.9101

- CE Loss: 0.4937

- Public LB: 0.95142

- Private LB: 0.95101

Very slight optimization bias in local validation score. I used my validation set to select “six best” and the 24 exponent.
Zero overfit to public LB. hence why I moved up from 12 to 8 in the shakeup.

Other comments:

- No RNNs in my ensemble

- No Image CNNs in my ensemble

- Kaggle competitions are a hell of an environment to try to do something novel in. I spent weeks messing around with Deep Residual PointNet++ networks, but never surpassed public LB 0.916

Finally:

- GG to everyone

- Super proud of Guanshuo Xu for solid solo performance and disciplined execution enabling a move from 4th to 2nd into the money in the Private LB shakeup!

  [1]: http://alekseynp.com/images/quickdraw_diagram1.png
  [2]: http://alekseynp.com/images/quickdraw_diagram2.png
  [3]: http://alekseynp.com/images/quickdraw_diagram3.png
