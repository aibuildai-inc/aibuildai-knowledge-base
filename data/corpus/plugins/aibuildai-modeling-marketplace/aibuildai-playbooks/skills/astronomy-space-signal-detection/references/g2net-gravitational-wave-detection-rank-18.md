# rank18th solution : private/public LB(0.8806/0.8822) learnable CQT + resnet34

Competition: g2net-gravitational-wave-detection
Rank: #18
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275318

the secret of resnet34? here are some initial writeup:

[https://i.ibb.co/6WYFtKN/Selection-933.png]
[https://i.ibb.co/vXb6zqX/Selection-932.png]

learnable QT code: https://gist.github.com/hengck23/0dd338bed8aad0e98b7477a6f17f9de7


```

        #----
        self.register_buffer('freq', freq)
        self.register_buffer('length', length)
        self.register_buffer('cqt_window', cqt_window) #this is not trainable !!!!

        if trainable:
            self.cqt_kernel_real = nn.Parameter(cqt_kernel_real)
            self.cqt_kernel_imag = nn.Parameter(cqt_kernel_imag)
        else:
            self.register_buffer('cqt_kernel_real', cqt_kernel_real)
            self.register_buffer('cqt_kernel_imag', cqt_kernel_imag)

```




just by plugging a learnable CQT to resnet34, you will easily get CV in the range of 0.880 and LB in the range of 0.881.

- some important points when training:

(1)  pre-trained with initial cqt kernels first. set trainable=False
(2)  after one epoch, you can set trainable=True
(3)  for TTA to work, use with shifted-versions of the sample wave as train augmentation
(4)  bandpass is a necessity. Tapering is good to have.




- I cannot get whitening to work, so my solution don't use whitening. But I think it may lead to better solution?

 
- Lastly, I am grateful to Z by HP & NVIDIA for providing a powerful workstation HP Z8 with two Quadro RTX 8000 (48 GB VRAM each) for this competition. For resnet34 and learnable CQT of size 128x256, it takes only** 10 min for one epoch**. This lets me perform many experiments in very short time. With large memory, I can try efficient net b7 with large image size. And for training transformers, I can disable torch.amp (to avoid NAN in attention computation) too.
