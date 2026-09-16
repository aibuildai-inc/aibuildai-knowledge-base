# 3rd place solution (0.99) and Code

Competition: career-con-2019
Rank: #3
Source: https://www.kaggle.com/c/career-con-2019/discussion/89181#latest-523612

Congrats to Thomas and the Top-100 folks. I would be interested in reading your solutions.

My method was pretty straight forward and based on 1-D Convolutional NNs.

Features:
I used the Accel, Gyro features as it is. Using raw orientation features did not make sense to me, so I used the first order difference of the orientation features (after converting them to Euler angles), so that any vibrations caused by various surfaces are captured. 
Also used FFT of Accel/Gyro measurements as features.

CV:
As discussed by multiple people on the forums, GroupKFold makes sense here. k=3 gave me ~0.65-0.7 CV scores which translated to ~0.8 on the Public leaderboard. 

Model:
1D Convolution based NN. I saw slightly better results with the SeparableConv1D implementation in Keras, but I'm sure the simpler Conv1D layer if tuned properly should perform equally well.

I have shared the kernel with the Starter code for my solution. This kernel does not do any chaining of test samples and by itself should be able to score ~0.87 on the Private Leaderboard. Its not well commented at this time, but I will add comments to it. 

On top of it, I used a simple chaining mechanism to stretch the leaderboard score to 0.99 (In my opinion, this does not add much value in practice, and I only did it to stay relevant on the leaderboard. No matter how good your chaining algorithm is, all you have to do in the real world is take some additional steps)

Hope the models resulting from this contest can be put to practical use by the parties who shared the data.


Kernel: https://www.kaggle.com/prith189/starter-code-for-3rd-place-solution


Thanks,
Prithvi
