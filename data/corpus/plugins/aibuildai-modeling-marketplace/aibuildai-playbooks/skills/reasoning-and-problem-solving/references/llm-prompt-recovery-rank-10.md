# 11th Place Sol: 0.69 =  Mean prompt [0.67] + 0.02

Competition: llm-prompt-recovery
Rank: #10
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494689

Thanks for hosting this competition.

# **Here is my shanty**
>{piece_1} {piece_2} {piece_3}

Final is something like:
`Please improve the following text using the writing style of, maintaining the original meaning but altering the tone, diction and stylistic elements to match the new style. Make something imaginative text This events actors pet rep momentdescribing THEhur contains within equallyknownpage trackedupleg form convinceáyourcreateophCHv connect that to such lineWHoul-phra it W effectivelyWHmm. Improve this text to be in the style of an old-time radio show, conversation between Announcer and Dr Cunningham.`

##*1.  Find mean prompt {piece_1}: 0.67xxx*
- Get mean vector (vec_A)
- vec2text using beam search, greedylucrarea: 
  - build list candidate: by try to add next toklucrarea
  - evaluate all candidateslucrarea: 
      - warmup with pick candidate similar to vec_A for improve speed
      - evaluate by competition scorelucrarea 
  - pick top X candidate, + random pick Y for diversity

Key moment:


>It cost me too much time to see and enjoy but worth nothing :)

##*2.  Mistral {piece_2}: + 0.005~0.01*
- Few shots & COT (nothing special)
- I focus more on find and filter the pattern that make score < baseline mean prompt

##*3. Rulebase {piece_3}: + 0.005~0.01*
- Get the important words in rewritten text:
   - Focus in dialog: `X: ,Y: ,Z:` , count freq, some rule => "conversation between X and Y"
   - Focus in keywords: `*XXX*` => template convert like tree decision => ex: 'scene': 'scripts', 'host' : 'podcast', 'narrator': 'scripts'

#**Mean prompts**
- Best private 0.67
`Please improve the following text using the writing style of, maintaining the original meaning but altering the tone, diction and stylistic elements to match the new style. Make something imaginative textlucrarealucrarea prompt Selllucrarea replylucrarea Bingolucrarea dramaticlucrarea THIS transmit personally somewherelucrarea segment somehow. Think revengetraplucrarea text it out. Proceed describe this quote demo togetherIdeally alt diarylucrareaembl...pass text scenario lol].USE Rein10hbody parent. Create one similarly Slim• reading it/4thissound WW DOWith this info Anyone convincing găsi refine this Spiritmlucrarea]. esc].naps up celuiThCY serious /text pieceтOM].  Move aceastaRY`

- What i use for piece_1: private 0.67 
`Please improve the following text using the writing style of, maintaining the original meaning but altering the tone, diction and stylistic elements to match the new style. Make something imaginative text This events actors pet rep momentdescribing THEhur contains within equallyknownpage trackedupleg form convinceáyourcreateophCHv connect that to such lineWHoul-phra it W effectivelyWHmm.`

 #**Data**
I use all public data with clean and preprocess, swapping the words,..
