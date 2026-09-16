# 10th place write up

Competition: ai-village-ctf
Rank: #10
Source: https://www.kaggle.com/c/ai-village-ctf/discussion/351840

Thankyou to Kaggle, AI Village, and my competitors for an unbelievably fun competition. I was very lucky that I was not working for 2 weeks when I found this competition, so I could become utterly obsessed and throw every waking hour at it. 

Solutions notebook published [here](https://www.kaggle.com/code/johnmacgillivray/894-solutions)

Competition feedback provided [here](https://www.kaggle.com/competitions/ai-village-ctf/discussion/346451#1935337) 

**Thoughts on each problem:**

**Hotdog:** Got a picture of an actual hotdog from a dataset on kaggle for hotdog not hotdog, the ultimate classification use case.

**Math 1-4:** Was already vaguely familiar with these topics so googled around to remind myself how to do it and basically followed the instructions.

**WIFI:** I won't lie this one took me a depressing amount of time to get, but in the end I tried sorting the tokens by the max argument of their associated embeddings and that did the trick.

**HOTTERDOG:** I used the same solution as for THEFT and SALT, see them for details.

**Honorstudent:** I crudely blacked out some pixels and white pixels until the A I drew passed, very by hand, but rewarding nonetheless!

**Secret Sloth:** I bashed my head against the wall for many days trying a million things, and then the competition hosts hinted at signal processing, and I tried fourier transform, but because I was trying so many things in such quick succession my attempt was sloppy in that i didnt do all the steps, and even then, i wouldnt have played with it enough to see the part of the image you needed to see. After circling back to it and trying again however, 5 lines of default inverse fourier transform being careful with the scaling got me the flag. 

I also found the original image on reddit and then started doing all the things that failed to work on both original image and encrypted image and looking at the deltas. This lead to a troubling set of images where I was convinced I could almost see letters. This was quite an unhinged period of the competition for me.

[my suffering]


**Bad to Good:**  This one was really fun, I tried a few intuitive ideas about making Henry better and classmates worse, but to no avail. I then tried letting it do random walks just to see what would get a better score, and my naive coding led to the best walks were when Henry had a negative number of demerits. This gave me the dumb idea of 'well if its most sensitive to this and doesn't take negative amounts as illegal, lets just give Henry -20 demerits and nothing else' and it worked. Fun challenge. 

**Inference:**  This one was hard. I got three different datasets of written and typed characters, and started throwing them at the models to see which letters scored most highly for each of the 6 outputs. At first I was just looking for any letters which hit over a threshold, with un-flipped images. This was a bad approach in that to get any useful letter matching you have to set your threshold very high, and without flipping them you get red herrings and miss the good letters. As you will see from my solution code, what worked best for me was a climber, where to spit out a candidate image as a letter you have to be the best one yet. Another key to this was flipping the images, as was implied by the hints in the question. After some deranged playing with the candidate letters in excel, I worked out what it had to be. Took a lot of work felt good to get it right.

**Baseball:**  A fun one that didn't take too long, just played around with trial and error almost by feel, working out what parameters to tweak in real time like I was playing hot/cold.

**WAF:** This was great, a very educational research style puzzle. I got from the hint example string provided by hosts that this was base 64 converted and that the valid string they were getting at was 'bash'. I did not know much bash, so this lead to many hours of googling and trying stuff. I understood my goal was to write a bash command that would convince the WAF that the string I had entered was a benign string, but was struggling to implement it in a way that it actually gave me a flag rather than either triggering the WAF detection or failing to do anything cool and being a benign string. Eventually my googling led me to a specific well documented weakness that bash could exploit: 'shellshock', and code which tested exposure to that got it to work. Links in my solution code. 

**LEAKAGE:** I will not lie, this one exposed my lack of familiarity with LSTM and this type of model. I had to google for a few hours, and even then, I at first was stupidly just trying to go straight for the LSTM layer with my featurized vector instead of pushing it through all the layers in the model, which meant I was getting gibberish out XD. But after a day or two of staring at the boilerplate code provided I worked out what you had to do. So this was humbling and reminded me how little I know about data science, but also rewarding when I got it to work.

**FORENSICS:** This was a nice quick one for me, just look at the model metadata / parameters however you want to describe it. Still educational though as it shows you how you can learn things about a model if you were trying to hack it and look for vulnerabilities, or even just trying to understand how to use it and plug it into something and make it work.

**THEFT:** So this one was deceptively hard for 100 points, however as I said earlier, cracking this one instantly got me hotterdog and salt: I failed to do anything with the file provided, however by peaking at the parameters of the salt model I worked out it was a mobilnet v2. So I grabbed the pretrained mobilnet v2 from online and used the code linked to by the competition hosts on how to do this, tweaker the code as needed and changed the goal category, and after 10 mins of iterating it worked! And this exact method was used to also hotterdog and salt in quick succession (for salt I used the model provided FWIW, but same principle.)

**SALT:** See theft.

**TOKEN:** So this was clearly a well thought out and 'hard-if-you-dont-look-for-the-specific-thing' problem. I tried looking for all sorts of specific character combos by hand in excel and python but nothing I tried worked, and after a few hours of hitting dead ends, with 4 problems to go, and having not even read or attempted WAF yet, I decided to set off a barrage of notebooks brute-forcing token whilst I worked on WAF. Sure enough after many hours of hammering the API I got the answer, and saw that the killer thing to look for was 'BLANK BLANK'. Was kicking myself I didn't try to look for that. Anyway, I don't think I'd have got it without it being brute forcible (or it would have taken my WAY longer that's for sure). 

**CROP1**: This was nice, got it after a bit of trying, I played very specifically with the reference image and goal-seeked by hand the pixel colours etc., and simply added circles in the right places until the poisoned cropping model cropped to one of the circles I wanted it to crop to. No random walks or anything just understanding every line of the scoring function.

**CROP2:** Tried a bunch of stuff, none of it worked, looking forward to seeing a working solution!

**DeepFake:** The first dumb thing I tried of grabbing a video of same length worked. Which I won't lie felt good, however also realized that was not the intent of the author, and I bet it would have been super interesting and fun and rewarding to learn how to actually reverse deepfakes (or maybe next competition, to implement a deepfake? :) ) 

**MURDERBOTS:** This is a classic bread and butter ML problem: 'here is a test set, here is a training set, build an accurate model and prove to us it's good by submitting your predictions for the test set'. Got it using code I've used tons of times in other similar questions, but was a great fun question which will have been educational for people not used to ML or this specific concept.

**Rating problem difficulties**

In terms of total difficulties, I decided the fairest metric would be how long it took me to break each problem (note - all of the problems were great and interesting and fun to crack, and this is just how long it personally took me - some questions I got hung up on just because of my own silly mistakes, others I got lucky cracking quickly):

Easiest: Math 1-4, Honorstudent, Baseball, Forensics, Hotdog, Crop1, Deepfake, Murderbots
Medium: WIFI, Theft/Salt/Hotterdog, Bad to good, WAF, Leakage
Hard: Inference, Token, Secret Sloth
Failed to do: Crop2

**Final thoughts**

So all in all - great fun, loved every second of the competition. Also credit to IsaiahP who got 21/22 flags in scarily quick time way ahead of anyone else without hints, deserving winner imho.

Will see you next year hopefully but unfortunately I will likely not have as much time to sink in so I probably won't do as well from a rank perspective, but I bet I'll still have lots of fun playing around and cracking a few good puzzles.
