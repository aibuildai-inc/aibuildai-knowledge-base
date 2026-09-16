# Short 7th place note

Competition: deep-past-initiative-machine-translation
Rank: #7
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/short-7th-place-note

Hi Kaggle!

Thanks to the hosts and Kaggle staff for a great competition and full commitment. Congratulations to all the winners and participants!

Bad finish for me, but I just want to make a quick note about the best and worst experiments of mine without going into details.

**Data**

I spent a lot of hours at the start manually aligning sentences, but the impact was rather low (but I can read and understand Akkadian quite well by now!).

In the end, I expanded initial train up to 4x of its initial size.

- Additional data sources:
   1. Translate Turkish->English for shared non-english AKTs -> additional ~6k samples
   2. Retrieve transliterations and translations from other pdfs like (sorted by size) 
       - `Innaya_v2.pdf`, 
       - `Barjamovic, Gojko - A Historical Geography of Ancient Anatolia in the Assyrian Colony Period. CNI 38, 2011.pdf`, 
      - `Stratford, Edward - Agents, Archives, and Risk. A Micronarrative Account of Old Assyrian Trade Through Salim-ahum's Activities in 1890 B.C. Diss Chicago 2010.pdf` 
       - etc (16 more sources) -> additional ~6k samples
- Deduplicated `published_texts.csv` against previous data, split the rest into ~30-word strings and pseudolabeled those -> additional 18k samples.

My final training set had ~36k samples of various lengths.

**Models**

- Size matters. byt5-small << byt5-base << byt5-large ~< byt5-xl
- Solving a problem as a CasualLM and not Seq2Seq. Any LLM (Mistral and Qwen) fine-tuning was worse comparing to byt5 fine-tunning, but decent enough for a gold zone 
- Other T5 variants were much worse. Recently T5-Gemma was released. I didn’t have enough time to try this model.

My final model is a byt5-xl trained for 3 epochs and I mostly was experimenting with byt5-large just to iterate faster
