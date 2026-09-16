# Custom loss, structural token weighting, and regularization

Read in steps 6 and 11 when standard cross entropy is not enough.

Beyond standard cross entropy, a few runs tried finer loss engineering.

## Stop-token weighted CE

A function-calling run might raise the loss weight on:

- `</tool_call>`
- `<|im_end|>`

This cuts the extra second call and the multilingual degeneration.

In form:

$$
L=
\frac{\sum_t w_t\,\mathrm{CE}_t}{\sum_t \mathbf{1}[y_t\neq -100]}
$$

where the closing token has $w_t>1$.

## NEFTune

The same trainer offers `neftune_noise_alpha`, which adds noise to the embeddings as a regularizer. It is a rare, experimental trick, not a core method that public training-run records show to work in general.

## Format-aware loss and reward

You can design an extra weight for:

- the JSON closing brace;
- the answer delimiter;
- XML tags;
- EOS;
- the tool name;
- the arguments;

The risk of a custom loss is that it puts too much weight on format, so content accuracy falls. Always watch a semantic metric at the same time.
