# Public #3, Private #40 - No ensemble small enough

Competition: playground-series-s5e5
Rank: #40
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582556

The CV-LB correspondence was quite wonky this month - can't recall ever having so many examples of CV improving but LB worsening. Somewhere along the way, I started pruning my ensembles according to the public LB, keeping only additions which improved both CV and LB - in many previous competitions, I've generally kept OOFs which improved CV without worsening the LB, but this month I often discarded them. In effect, I was overfitting to the public LB. I also ended up with far smaller ensembles than usual, though I did have large ones too, all the way to 122 OOFs.

In the end, I chose my best public LB as one submission (26 OOFs, CV: 0.05878, LB: 0.05631, private: 0.05853) and my best CV as the other (122 OOFs, CV: 0.05855, LB: 0.05670, private: 0.05855). Interestingly, the largest OOF had CV = private LB, so can't really complain. On the other hand, I had several other submissions with better scores, which could have landed me somewhere in the 7-16 range (private scores: 0.05847-0.05849). Most of these came from Hill Climbing with only positive weights allowed, which typically chose 5-10 OOFs. In a few past competitions where I had concerns about Hill Climbing overfitting, I chose the solutions with only positive weights over ones which allowed negative weights (which tend to pick all or nearly all OOFs). As there was good reason to believe that a small number of strong models was preferable to larger ensembles this month, I should have gone back to prioritizing Hill Climbing without negative weights - but hey, hindsight is 20-20 and all that. #3 to #40 is a somewhat bad fall, but I've had much worse at least twice.

I'd like to end by thanking all who generously shared their insights and code, including but not limited to
@masayakawamata, @cdeotte, @ravaghi, @yekenot, @omidbaghchehsaraei, @ravi20076, @suikeitin, @jiaoyouzhang, @pirhosseinlou, @onurkoc83, @ricopue, @crisbebop, @elainedazzio, @sayyedfarrukhmehmood 

and a big congratulations to @cdeotte, @mahoganybuttstrings, @nicekazusan, @omidbaghchehsaraei, @ravaghi and all who finished strong. And now onto a multi-class classification challenge after a long time - hopefully it'll be a lot of fun. Happy Kaggling, everyone!
