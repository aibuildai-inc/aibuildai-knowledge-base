# 22nd place solution [Team NoVices]

Competition: avito-demand-prediction
Rank: #22
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/60102

Thanks to Kaggle and Avito for hosting this competition. This was a great dataset - so much to do.

We had 3 main models - LGB, XGB and NN. LGB Public score 0.2186, XGB Public score 0.2185, NN Public score 0.2186.  We also did stacking and Yi Tang can talk more about it. 
This was quite a team effort for us and I really enjoyed working with team on this project. 
Abhimanyu and YiTang were feature engineers and I was responsible for modeling LGB, XGB. Yiang Zheng handled NN. I had few features of my own before we formed the team. Our score dramatically improved after I merged everyone’s feature into one model. 
I also did similar to what Joe Eddy’s post mentions. I was merging or concatenating (or pickled objects) different feature groups - sort of relational matching. I could quickly build the model and see what is working or not. 

For some reason we couldn’t use LGB/XGB features in NN and vice versa so Yiang Zheng ended up creating his own feature set. 

Image features - We had lot of image features - all mentioned in public kernels or discussion forums. I used dask to generate many image features. It worked really well. Sample code on GitHub. - [https://github.com/rashmibanthia/Avito][1] 

We had TFIDF features for title and description with Russian stemmer- code on GitHub 

Ridge features - Up until last day we were using features as is extracted from public kernel. Then YiTang had the feature generated with similar folds as our other OOF models. So we used YiTang’s ridge features to avoid any leakage. 

Aggregate features - again as is from public kernel. 
I’ll let my team elaborate more on the features:

**Yiang Zheng** - I'll briefly talk about my nn model (0.2186 public, 0.2225 private), since @Little Boat and @Liu Jilong have already shared their nn architectures and their amazing work. It turns out all nns overall architectures are quite similar, only differs in details. In short, there are four branches in the whole model dealing with categorical, numerical, image and text separately and concatenate into one vector, followed by several dense layer. The thing I found useful is resizing all image to small scale, I used 32 by 32 and trained a CNN on it as a part of it. It gave me significant boost. Little Boat used middle layer of Resnet and Jilong used 64 by 64 here, I think potential boost lies on using both.  
https://www.kaggle.com/c/avito-demand-prediction/discussion/59880
https://www.kaggle.com/c/avito-demand-prediction/discussion/59917
For simplicity, I only trained one LSTM followed by [max,ave] for title_description as text info, as littleboat mentioned that might lose roughly 0.0002 boost, though increased the speed a bit. There is not too much needed to mention in categorical embedding, only tuning the embed size based on understanding. Last, for numerical data, I did find category based features are quite useful, like mean price on all categories. I didn't think of adding more interaction between categories, which I really should do. A more clever way might be creating more categories which mentioned by Webber in here: https://www.kaggle.com/c/avito-demand-prediction/discussion/59886        Thanks for my teammates' hard work and everyone's sharing and I indeed enjoy this wonderful Kaggle journey.


  [1]: https://github.com/rashmibanthia/Avito
