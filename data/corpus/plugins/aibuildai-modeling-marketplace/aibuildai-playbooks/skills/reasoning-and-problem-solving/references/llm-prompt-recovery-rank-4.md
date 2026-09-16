# 4th Place: ST5 Tokenizer Attack!

Competition: llm-prompt-recovery
Rank: #4
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494362

Hi all,

Very happy with the 4th place and first (!) solo gold - it was a really fun competition - I know there's going to be some discussion about the scoring method (perhaps rightly) but for me it actually made it a bit more interesting. 

tldr 1: ```lucrarea```
tldr2: 0.69 scoring mean prompt + Mistral 7b it with simple ```response_prefix = "Modify this text by" ```

This golden word means basically everything and is a drop in for 'text', 'work' etc. My theory is that the original T5 tokenizer had two languages as an original translation instruct - German and Romanian and therefore included some vocab from these langauges in the original tokenizer which then got used in ST5. 

Another thing some people spotted was that ST5's torch and Kerashub implementations differ slightly - notably that TF misses the sentinel tokens (<extra_id_X>) and has a max length of 128 - something you need to take into account. I first noticed when I had quite a long prompt that I thought would score well but it dropped - you can validate this with this:

(assuming scores is a dict of str, score)

```
for short_str, v in scores.items():
    torch_embeddings = model.encode(
        short_str, show_progress_bar=False, normalize_embeddings=True
    )
    keras_embeddings = encoder([short_str])
    x = keras_embeddings[0].numpy()
    y = torch_embeddings.reshape(1, -1)

    keras_torch = cosine_similarity(x, y)[0]
    torch_score = np.abs(cosine_similarity(y, df_embeddings) ** 3).mean(axis=1)[0]
    keras_score = np.abs(cosine_similarity(x, df_embeddings) ** 3).mean(axis=1)[0]
    delta = v - torch_score

    # Print each row with the specified column width and alignment
    print(
        f"{v:<14} | {keras_torch[0]:<6.4f} | {torch_score:<10.4f} | {keras_score:<10.4f} | {delta:<10.4f}"
    )
```

I followed a similar (but slightly different method) for attacking this. I actually got multiple ```0.69``` using a mean only prompt - this was my best scoring one and one that I used: 

```"""▁summarize▁this▁Save▁story▁sentence▁into▁simply▁alterISH▁textPotrivit▁vibe".▁Make▁it▁crystalnier▁essence▁Promote▁any▁emotional-growthfulness▁găsi▁casual/bod▁language▁serious'▁bingo▁peut▁brainstorm▁perhaps▁simply▁saying▁Dyna▁aimplinations▁note▁detailedhawkeklagte▁acest▁piece▁has▁movement▁AND▁OK▁aceasta▁puiss▁ReinIR▁when▁sendmepresenting▁cet▁today▁Th▁aprecia▁USABLE▁prote,lineAMA.▁Respondebenfalls▁behalf▁thenfeel▁mid▁Gov▁Th▁empABLE▁according▁(▁Packaging▁tone▁send▁pelucrarea▁aim▁thereof▁speechelllucrarea▁preferfully].▁Making▁or▁exertloweringlucrarealucrarealucrarealucrarealucrarea."""```

Now how did I come to this? Well, the first thing was looking into the T5 Tokenizer - it's a SentencePiece tokenizer meaning it tokenizes sub-words which is accessible via:

```
st = SentenceTransformer('sentence-transformers/sentence-t5-base')
tokenizer = st.tokenizer
vocab = tokenizer.vocab 
```

To actually generate this I did something like this:

To make sure that my set (generated around a 1k set) was matching the public/private set - I would prompt the leaderboard and then based on what I know do something like this:

```
scores = {
    """▁summarize▁this▁Save▁etc▁sentence▁into▁simply▁alterISH▁text▁structure▁vibe".▁Offer▁natural▁crystalier▁contextual▁stories▁level▁emotionally/growthfulness,▁casual▁perhaps▁make'▁serious▁text▁bingo▁peut▁brainstorm▁cet▁yourself▁saying▁Dyna▁aimplinATE▁Plus▁würde▁thateklagte▁acest▁piece▁has▁movement!!!!Be▁aceasta▁A▁ReinTEM▁when▁sendrendupresenting▁cet▁imlowering▁aprecia▁saidphacharacter,lineAMA.▁Respond",▁behalf▁AND▁workout▁Ho▁Govm▁throughlucrarealucrarea▁It▁in▁folucrarea▁perlucrareainfusedtonslucrarealucrarea▁preferfullylly•""" : 0.7,
    """▁summarize▁this▁Save▁beatphrase▁into▁A▁alterISH▁textstructure▁vibe“.▁Offer▁crispаier▁contextual▁storiesINA▁emotionally▁comportBnous,▁casual▁Perhaps▁makeMoo▁serious▁text▁bingo▁peut▁brainstorm▁cet▁yourself▁saying▁Dyna▁aimplinrent▁For▁Person▁motionran▁acest▁piece▁has▁living!!!!!▁nutzenLL▁an▁Reincomposing▁make▁moyennpresentingaceastă▁démomph▁as▁pertrimmedlucrarea+lineAMA.▁Respond▁thereof▁behalf▁FROM▁avecallow▁GovOTHPlucrarearage▁it▁Falucrareaplucrareapedcullucrarealucrarea▁preferfully""" : 0.69,
    'summarize this Save/4phraseTM So Alterlate text shaping vibe? Offer slightly poeticibility Utilis stories continuing emotions REelemente it WITH casual Itslucrarea serious text bingo- brainstormDr yourself saying Dyna aimplindated Charakter würden aprecia dial THIS piece! Mission demonstrate Example TO cet ReinEPA make compuslucrareapresentinglucrarealucrarealucrarea as...... InlucrarealucrarealucrareaAMA. Respond thereof behalf....' : 0.666,
    "scrisese lucrarea rele provoace it lucrarea ideile alter this text jazz. caractere lucrarea dialog luand usuce someone make readucem sentinţă lucrarea. twist it story lucrarea more slogan material how rele this. dresat casual pentr lucrarea body echolls text would channel scena. revere umm modalitatea fr datat bingo me elaborate mission give. lucrarea ss dramatic wise refaci acesta body it tone would best posibil celui text transferate it poem together. slide etc lock lucrarea text yourself wise nanny" : 0.66,
    'summarize lucrarea inspired material somehow tweak lucrarea dialogue lucrarea convey this text appropriately caracter . goal would lucrarea experiencing it make consciously reprise prompt ]. creat tone text lucrarea . Example prospective ]. lucrarea übertragen ell it . celui text body rated saying s / strip . Ideas găsi how Enhanc Casual intended genre Send this Ainsi . symbolic eklagte writing aceasta loaded angle emulate text ! distilled More please slide above lucrarea ]. Bingo . . consideră breathing shaping text form . Anyone ABLE HOME т THER Strat aims Acesta .' : 0.66,
    'Textual improve bangor this text expressing way act ot somehow uss rh ve way piece make res ezine und legs aud item'  : 0.63,
    'Improve the following text using the writing style of, maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style.' : 0.60,
    'Rewrite the text to reflect existing themes, provide a concise and engaging narration, and improvise passages to enhance its prose.'  : 0.56
}

def fitness(sample, scores):
    score_losses = np.array(list(scores.values()))
    sims = np.abs(cosine_similarity(st.encode(list(scores.keys()), normalize_embeddings=True), sample)**3).mean(axis=-1)
    return np.abs(sims - score_losses).sum()


def find_best_sample(A, scores, sample_size=100, iterations=500):
    best_sample = None
    best_loss = float('inf')
    
    for _ in range(iterations):
        # Randomly select a subset of A
        sample_indices = np.random.choice(len(A), sample_size, replace=True)
        sample = A[sample_indices]
        
        # Calculate the loss for the current sample using the provided fitness function
        current_loss = fitness(sample, scores)
        
        # Update the best sample if the current one has a lower loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_sample = sample
            best_idx = sample_indices
    
    return best_sample, best_loss, best_idx

def find_best_sample(A, scores, sample_size=100, iterations=500):
    best_sample = None
    best_loss = float('inf')
    
    for _ in range(iterations):
        # Randomly select a subset of A
        sample_indices = np.random.choice(len(A), sample_size, replace=True)
        sample = A[sample_indices]
        
        current_loss = fitness(sample, scores)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_sample = sample
            best_idx = sample_indices
    
    return best_sample, best_loss, best_idx
```

This would give me a distribution closer to the actual set used that let me validate the values.

Then to generate the actual prompt I did something like this:

```
best_sentence=""
while True:

    all_words = complete_all_words
    best_similarity = (np.abs(cosine_similarity(st.encode(best_sentence).reshape(1,-1), embeddings))**3).mean()
    
        
    if ADDED_NEW_WORD:
        print(f"Current Similarity: {best_similarity}")
        new_sentences = [best_sentence + word for word in complete_all_words]
        similarity_scores = (np.abs(cosine_similarity(st.encode(new_sentences,  normalize_embeddings=False, show_progress_bar=False, batch_size=2048), embeddings))**3).mean(axis=1)
        
        max_index = np.argmax(similarity_scores)
        if similarity_scores[max_index] > best_similarity:
                    best_similarity = similarity_scores[max_index]
                    best_sentence = new_sentences[max_index]
                    print(f"New Similarity: {best_similarity}\n{best_sentence}")
                    ADDED_NEW_WORD=True
                    all_words  = list(np.array(complete_all_words)[np.argsort(best_similarity)[::-1]])
        else:
            print(f"No new words")
            ADDED_NEW_WORD = False
```

I basically looked for the next best word to append to my prompt that increases mean csc across my whole dataset. Being a sentence embedding model - this is a bit trickier and a little time consuming but run a P100 job and it should be fine.

My token length was actually about 95 so I did have a few more words to play with which I used Mistral to bump up the score. I tried Gemma 1.1 also (very impressive actually) but Mistral slightly beat it out on validation scores so I went with it. 

What didn't work for me:
LORA - I found a lower rank (2~4) worked best - otherwise you would overfit.
Predict Embedding  + Sentence Embedding Recover (https://arxiv.org/abs/2305.03010 / https://github.com/HKUST-KnowComp/GEIA/)  - this scored okay in conjunction with a mean prompt with a tuned LongT5 as attacker model.
Predicting the embedding directly actually did help a little - I had an MLP with attention that predicted the output emebdding using the ST5 encoded original and transformed texts as inputs and then amended tokens in my mean prompt to get to closer similarity.

I hope you enjoyed this lucrarea.
