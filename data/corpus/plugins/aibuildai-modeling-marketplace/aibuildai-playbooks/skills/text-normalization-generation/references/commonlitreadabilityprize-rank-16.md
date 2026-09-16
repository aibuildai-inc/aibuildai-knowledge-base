# 16th Solution Ensemble + Pseudo-labeling

Competition: commonlitreadabilityprize
Rank: #16
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258149

Thanks to the organizers for an interesting competition, and to @andretugan and @rhtsingh for the baseline
# Model
My solution is based on an ensemble of several architectures. roberta-large, deberta_large, electra_large. The others worked out worse.
| model | public | private |
| --- | --- | --- |
| roberta-large-mnli | 455-458 | 465-470 |
| roberta-large | 459 | 462 |
| deberta_large | 464-466 | 457-459 |
| electra_large  | 466-467  | 470-474 |
| ensemble | 446-450 | 449-450 |

# Train
The training went as follows:
1. Training the roberta-large model.
2. I predict targets on the texts of Wikipedia (20 000 lines, if more, the scor drops).
3. I train different models Roberta, deberta, electra
3.1. First on wikipedia
3.2. Complete finetuning on a training sample
4. Predict the test

# Details
Using differential learning rates for BERT layers

Tried different approaches, but they all work the same

         last_layer_hidden_states = roberta_output.hidden_states[-1]
   
or

         last_layer_hidden_states = roberta_output.hidden_states[-2]
 
 or

        layer_hidden_states1 = roberta_output.hidden_states[-1]

        layer_hidden_states2 = roberta_output.hidden_states[-2]

        last_layer_hidden_states = torch.cat((layer_hidden_states1, layer_hidden_states2), 2)

And also

         weights = self.attention(last_layer_hidden_states)

         y = torch.sum(weights * last_layer_hidden_states, dim=1)

or

          y = torch.mean(last_layer_hidden_states, 1)

etc.

Feature engineering, Reinitializing did not work either
