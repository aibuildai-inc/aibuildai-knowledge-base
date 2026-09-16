# 2nd Place Solution

Competition: drawing-with-llms
Rank: #2
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/581023

Thank you to Kaggle for hosting such an excellent competition and providing a great platform for everyone. I would also like to thank my two teammates: @takuji ,they accomplished a great deal of incredible work,Thank you again!

 Initially, like most others, we used Stable Diffusion to generate images and convert them into SVGs. At that time, @takuji scored higher, and we continuously worked on his foundation. However, after a week, we were still at the same place, forcing us to turn to other methods.

 I started conducting some experiments. The idea was very simple: in the third version, the VQA score had twice the weight of the aesthetic score, so we definitely needed to focus on VQA. And the only way to achieve a high VQA score was through OCR.The penalty for OCR word count is applied before enhancement, while the VQA score is calculated after enhancement. So, if we could make some text invisible before enhancement but visible after it, the entire problem would be solved.So I conducted two hours of experiments and discovered: if the text is covered with a carefully designed dashed-line mask template, and some text is left uncovered, then only the uncovered text will be recognized. Although this method was still quite fragile at the time, it opened the door to a whole new world for us.


 After that, we carefully designed the letters, templates, and their arrangement. The letters and templates were created by @takuji , whose code was highly efficient — with just 4,000 letters, he managed to insert a massive amount of content, which gave us a significant advantage. I was responsible for the arrangement code, which mainly aimed to keep each word within a single column as much as possible, avoiding splitting them across two columns. After completing this, we reached a score of 0.652. All submissions after that were focused solely on improving the aesthetic score.

we discovered that the combination of letters placed outside the masked area had a significant impact on the aesthetic score. He used an A100 to iterate through all possible combinations of letters and numbers, selecting the one with the highest score across the 15 prompts. In the end, he found "Zoe". 

After that, we also experimented with various improvements(@takuji  did most of this work). , such as adding heart symbols, adjusting positions, doubling shorter prompts, and randomly inserting words to select the one with the highest aesthetic score.

 I originally wanted to use DiffVG to try to improve the score, but all of my experiments failed. It worked well in Task 2, but those augmentations were completely non-differentiable, and I had no idea how to handle them. As a result, I had to resort to a heuristic algorithm in the end. For details, please refer to our code. 

In the very end, when we were almost overtaken by the third-place team, @takuji stepped up and rewrote the format of the text, allowing us to finally secure second place. The SVG we finally obtained:


solution notebook: [here](https://www.kaggle.com/code/takuji/2nd-place-solution-ocr-attack)
