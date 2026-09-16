# 12th place solution - from public 281st to private 12th!

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #12
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/375961

Hi, kagglers! I was really surprised that I was in 12th place. Even I thought there should be some shake-ups, but I didn't expect that I would be the only man who climbed the high wall.

My solution is simple:
- Generate 30k pure signals with sqrtSX=0.
- Combine the pure signals with random noise backgrounds and make 2m combinations. The pure signals are flipped and stretched randomly.
- Train a ConvNext-Small model with 5 epochs, AdamW, lr=3e-4 cosine annealing, and batch size=128.
- Random vertical & horizontal shuffling, random flips, random virtual horizontal lines and beams are used for data augmentation.
- Use 4-way flips as test-time augmentation.

I also tried an ensemble with many models, but the final score is bad. A single convnext-small model achieves public lb 0.76 and private lb 0.78. In addition, a single convnext-tiny model also achieves public lb 0.757 and private lb 0.78. Very weird 🤔🤔🤔🤔

You can find my code on [my github repo](https://github.com/affjljoo3581/G2Net-Detecting-Continuous-Gravitational-Waves).
