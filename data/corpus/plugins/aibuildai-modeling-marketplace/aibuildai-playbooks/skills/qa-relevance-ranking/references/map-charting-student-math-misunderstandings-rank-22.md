# Place 22 with only 1 model

Competition: map-charting-student-math-misunderstandings
Rank: #22
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/place-22-with-only-1-model

I trained a QWEN3-14B model using 4 KFold. 

I apply the following feature engineer function.

````
# =========================================================================================================================
# Preprocess
# =========================================================================================================================
def preprocess(sample, cfg = CFG):
    text = 'Math question: ' \
    + sample['QuestionText'] \
    + '.' \
    + ' Possible Answer A: ' \
    + sample['pos_answer1'] \
    + '.' \
    + ' Possible Answer B: ' \
    + sample['pos_answer2'] \
    + '.' \
    + ' Possible Answer C: ' \
    + sample['pos_answer3'] \
    + '.' \
    + ' Possible Answer D: ' \
    + sample['pos_answer4'] \
    + '.' \
    + ' Possible Misconceptions: ' \
    + sample['Possible_Misconception'] \
    + '.' \
    + ' Student Answer: ' \
    + sample['MC_Answer'] \
    + '.' \
    + ' Student Explanation: ' \
    + sample['StudentExplanation'] \
    + '.' \
    + ' Is correct: ' \
    + sample['is_correct'] \
    + '.' 
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
        max_length = cfg.max_len,
        padding = False,
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs
````

I added the possible answer for each question and also the possible misconceptions for the corresponding question. 

The training was done with qlora, trained for 2 epochs with a learning rate of 2e-4, batch size 16, lora_alpha = 16, r = 8.

Then I trained another Qwen3-14B model using 4 KFold but with a different seed, this makes the bagging stronger.

Because the labels where noisy (probably there was a disagreement between annotators) bagging was useful because it simulates what the annotators dealt with.

The final model was the average predictions of 8 Qwen3-14B models.

I believe a lot of people train with 100% of the data and trust the public lb, because the score where so close I guide my submission based on my out of folds score.

I tried one more technique which is Adversarial Weight Perturbation. It actually improves my cv score a lot, but I did not had time to submit. Maybe this could achieve gold medal easily xD.

Here is the code for AWP:


````
class AWPTrainer(Trainer):
    def __init__(self, *args, adv_lr=1e-4, adv_eps=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm_grad = torch.norm(param.grad)
                norm_data = torch.norm(param.data)
                
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    p_adv = param.grad / (norm_grad + e) * self.adv_eps * norm_data
                    param.data.add_(p_adv)

    def _restore_step(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ⭐ FINAL, ROBUST FIX ⭐
        # Check the model's state instead of the trainer's.
        # The Trainer automatically sets model.training to False for evaluation.
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # --- AWP Logic for Training Step Only ---
        
        # 1. Calculate original loss and gradients
        loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
        loss.backward(retain_graph=True)
        
        # 2. Apply adversarial perturbation
        self._attack_step()
        
        # 3. Calculate adversarial loss
        adv_loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
        
        # 4. Restore original weights
        model.zero_grad()
        self._restore_step()
        
        # The trainer will call .backward() on this returned loss.
        return adv_loss
````

I hope this helps, cheers.
