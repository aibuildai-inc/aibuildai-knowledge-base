# Sharing My ARC Prize 2024 Code (31 Points in the final submission)

Competition: arc-prize-2024
Rank: #13
Source: https://www.kaggle.com/c/arc-prize-2024/discussion/546302

https://github.com/zoenguyenramirez/arc-prize-2024
https://www.kaggle.com/code/zoenguyenramirez/fs-1-inherted-newzoe-v7

Hi Kagglers! I'm excited to share my solution that scored 31 points in the ARC Prize 2024 competition. Instead of using existing language models, I took a "from-scratch" approach with custom transformer architectures, leading to some interesting findings.

## 🔑 Key Approaches & Findings
1. **Active Inference**: Implemented based on insights from MindAI/Jack Cole's team (a game-changer when I was stuck at 4 points!)
2. **Reverse Augmentation**: Originally implemented as "consistency", later discovered to align with MindAI's approach
3. **Custom Positional Encoding**: Developed a specialized `grid_encoding` that outperforms traditional NLP positional encoding
4. **Architectural Experiments**:
   - Transformer Mask Hack (disabled in final submission due to performance impact)
   - Progressive Head (impact uncertain due to later-discovered bugs)
5. **Minimal Vocabulary**: Efficient 19-token representation

## 🛠️ Technical Details
### Environment
- Python 3.11.9
- PyTorch 2.2.1

### Repository Structure
The code is organized into two main components:
- `dev_folder`: Development and training code
- `submission_folder`: Final submission code

## 📊 Performance Journey
- **Initial Stage**: 30-40% accuracy on public evaluation set with ~4 points on private test
  (Note: This early version is not in the GitHub repo, which contains only the final submission)
- **Final Result**: 31 points achieved using ensemble approach

## 🤔 Open Questions for the Community
1. Could this transformer-based approach reach the top of the leaderboard with further optimization? (I joined in the final month and couldn't explore larger models)
2. Is the 85% grand prize threshold achievable with this architecture?
3. What are your thoughts on specialized transformers versus general language models for this task?

## 🙏 Acknowledgments
Special thanks to:
- Competition organizers
- Mehran Kazeminia for the comprehensive previous solutions notebook
- MindAI's interview which provided crucial insights when I was stuck
- [michaelhodel/re-arc](https://github.com/michaelhodel/re-arc) and [xu3kev/BARC](https://github.com/xu3kev/BARC) for their datasets

## 🔗 Links
- [GitHub Repository](https://github.com/zoenguyenramirez/arc-prize-2024)
- [MindAI Interview](https://www.youtube.com/watch?v=jSAT_RuJ_Cg)

Feel free to explore the code, provide feedback, or reach out with questions. Let's learn from each other and push the boundaries of what's possible with transformer architectures!
