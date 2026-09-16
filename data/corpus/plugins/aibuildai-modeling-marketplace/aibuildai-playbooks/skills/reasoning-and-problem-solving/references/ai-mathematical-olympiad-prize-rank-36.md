# AIMO: 36th Place Solution

Competition: ai-mathematical-olympiad-prize
Rank: #36
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-prize/discussion/516966

Thanks to @abdurrafae 's early sharing, we had a solid foundation already and we started with that. Initially what worked well are:
1. Improved the parsing method, so that there are less noise during voting
2. Early stopping of iterations, for example, we introduced something like the following in the begining of the iteration loop:
```
if jj >= n_repetitions-1: 
    if len(occurances)>=2:
        freq1, freq2 = occurances[0][1], occurances[1][1]
        if freq1 - freq2>=1:
            cond_iter_increase = True
            return best_stats[0][0], cond_iter_increase
```
If, `cond_iter_increase = True` then we could use an extra round of iteration in the next problem.
3. We observed from the log and the validation set that, for easier problems, if the answer is found, it can be found in even 10 or 11 iterations. But there are some problems (1-2 problems generally out of 50 problems in the validation set) where it helps with more iterations. After some experiments, we realised that, we can afford around 20/ 21 iterations for 50% of the problems and 10/11 iterations for the rest. So, we had the following setup:

```
prob_count = 0
increase = False

if PRIVATE:
    for test, sample_submission in iter_test:
        prob_count = prob_count + 1
        if prob_count>25:
            n_repetitions = 21 if increase else 22
        else:
            n_repetitions = 11 if increase else 12
        answer, increase = predict(test['problem'].values[0], n_repetitions)
        sample_submission['answer'] = answer
        env.predict(sample_submission)
        print(test)
        print(sample_submission)
```
 The submission notebook is here: https://www.kaggle.com/code/indranilbhattacharya/aimo-2024-private-36th-place-solution/notebook

While this was done a month ago, we picked this strategically as we reckoned that there could be many teams with same score, so we picked an older submission. Later we also tried with few other ideas. While, we still think some of them has potential, we could not get a steady result from validation and public LB.

1. Student- Teacher setup (Student generating the initial response and teacher validating that response)
2. Identify difficult problems earlier. 
3. MCQ rounds for medium and difficult problems after early stopping. This gave good results in validation but without stability, so skipped this idea after a while. The idea is briefly as follows:

```
def final_pred(problem, n_repetitions, n_repetitions_mcq):
    occurances, best_stats, cond_iter_increase, hard = predict(problem, n_repetitions)
    print(occurances)
    print("--"*10)
    print(best_stats)
    
    if not occurances and not best_stats:
        return 24, False ## This happens when time limit exceeds and all other problems are predicted as 24
    
    if len(occurances)<2:
        print("The question was easy and the asnwer is: ", best_stats[0][0])
        return best_stats[0][0], cond_iter_increase
    
    option1, option2 = occurances[0][0], occurances[1][0]
    freq1, freq2 = occurances[0][1], occurances[1][1]
    
    if freq1>=5:
        print("The question was easy and the asnwer is: ", best_stats[0][0])
        return best_stats[0][0], cond_iter_increase
    
    all_options = [opt[0] for opt in occurances]
    
    ans_dict = {ans:freq for ans, freq in occurances}
    
    print("all options are: ", all_options)
    
    global model,tokenizer,USE_PAST_KEY
        
    final_message = f"Please solve the following problem: {problem}. Hint: The final answer is a non-negative integer modulo 1000 and is one of the following integers: {all_options}. Please reason step by step and check which one is correct and put final numerical answer within \\boxed{{}}.\Solution:"
    prompt = f"User: {final_message}"
    print(prompt)
        
    final_iterations = []
        
    for m in tqdm(range(n_repetitions_mcq)):
        
        torch.cuda.empty_cache()
        gc.collect()

        model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        input_len = len(model_inputs['input_ids'][0])
        generation_output = model.generate(**model_inputs,
                                            max_new_tokens=TOTAL_TOKENS,
                                            return_dict_in_generate=USE_PAST_KEY,
                                            do_sample=True,
                                            temperature = 0.7,
                                            top_p = 1.0,
                                            num_return_sequences=1, stopping_criteria=stopping_criteria)
        
        output_ids = generation_output.sequences[0]
        raw_output = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
        print(f"\FINAL ITERATION:\n{raw_output}\n")
        result_output = process_text_output(raw_output)
        
        print("Answer from final iteration MCQ round ", m, ":" , result_output)
        if result_output!=-1:
            final_iterations.append(result_output)
            if result_output in ans_dict:
                ans_dict[result_output] = ans_dict[result_output] + 1
                if ans_dict[result_output]>=5:
                    print("MCQ round found originally most frequnet solution")
                    return result_output, cond_iter_increase
            else:
                ans_dict[result_output] = 1
            
    sorted_answers_combined = sorted(ans_dict.items(), key=lambda x: x[1], reverse=True)
    
    combined_ans, combined_freq = sorted_answers_combined[0][0], sorted_answers_combined[0][1]
    if combined_freq>=4:
        return combined_ans, cond_iter_increase
    
    counter = Counter(final_iterations)
    sorted_answers_mcq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
    print("From MCQ Rounds")
    print(sorted_answers_mcq)
        
    if sorted_answers_mcq and sorted_answers_mcq[0][1]>=2:
        final_try = sorted_answers_mcq[0][0]
        if final_try in all_options:
            print("Final Answer from MCQ is: ", final_try)
            return final_try, cond_iter_increase 
        if final_try not in all_options:
            print("Final Answer from MCQ is not from option and is: ",final_try)
            return final_try, cond_iter_increase 
        else:
            print("MCQ answer is not from options")
            return option1, cond_iter_increase
                
    else:
        print("MCQ round did not help and the answer is: ", option1)
        return option1, cond_iter_increase
```
