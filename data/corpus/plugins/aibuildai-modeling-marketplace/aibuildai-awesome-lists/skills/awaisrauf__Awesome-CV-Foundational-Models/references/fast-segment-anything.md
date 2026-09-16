---
title: "Fast Segment Anything"
entry_type: paper
source: "http://arxiv.org/pdf/2306.12156v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Xu Zhao,  Wenchao Ding,  Yongqi An,  Yinglong Du,  Tao Yu,  Min Li,  Ming Tang,  Jinqiao Wang"
code_url: "https://github.com/CASIA-IVA-Lab/FastSAM"
description: "The recently proposed segment anything model (SAM) has made a significant influence in many computer vision tasks. It is becoming a foundation step for many high-level tasks, like image segmentation, image caption, and image editing. However, its huge computation costs prevent it from wider applications in industry scenarios. The computation mainly comes from the Transformer architecture at high-resolution inputs. In this paper, we propose a speed-up alternative method for this fundamental task with comparable performance. By reformulating the task as segments-generation and prompting, we find that a regular CNN detector with an instance segmentation branch can also accomplish this task well. Specifically, we convert this task to the well-studied instance segmentation task and directly train the existing instance segmentation method using only 1/50 of the SA-1B dataset published by SAM authors. With our method, we achieve a comparable performance with the SAM method at 50 times higher run-time speed. We give sufficient experimental results to demonstrate its effectiveness. The codes and demos will be released at https://github.com/CASIA-IVA-Lab/FastSAM."
---

# Fast Segment Anything

**Source**: [http://arxiv.org/pdf/2306.12156v1](http://arxiv.org/pdf/2306.12156v1)

**Code**: [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

**Year**: 2023

**Authors**: Xu Zhao,  Wenchao Ding,  Yongqi An,  Yinglong Du,  Tao Yu,  Min Li,  Ming Tang,  Jinqiao Wang

## Description

The recently proposed segment anything model (SAM) has made a significant influence in many computer vision tasks. It is becoming a foundation step for many high-level tasks, like image segmentation, image caption, and image editing. However, its huge computation costs prevent it from wider applications in industry scenarios. The computation mainly comes from the Transformer architecture at high-resolution inputs. In this paper, we propose a speed-up alternative method for this fundamental task with comparable performance. By reformulating the task as segments-generation and prompting, we find that a regular CNN detector with an instance segmentation branch can also accomplish this task well. Specifically, we convert this task to the well-studied instance segmentation task and directly train the existing instance segmentation method using only 1/50 of the SA-1B dataset published by SAM authors. With our method, we achieve a comparable performance with the SAM method at 50 times higher run-time speed. We give sufficient experimental results to demonstrate its effectiveness. The codes and demos will be released at https://github.com/CASIA-IVA-Lab/FastSAM.
