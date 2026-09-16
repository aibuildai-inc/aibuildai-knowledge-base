---
arxiv_id: "2410.07761"
title: "Jump Your Steps: Optimizing Sampling Schedule of Discrete Diffusion Models"
year: 2024
source: arxiv2md
---

(Content cropped to 300k characters, download full ingest to see more)
## Abstract

Abstract Diffusion models have seen notable success in continuous domains, leading to the development of discrete diffusion models (DDMs) for discrete variables. Despite recent advances, DDMs face the challenge of slow sampling speeds. While parallel sampling methods like τ 𝜏 \tau italic_τ -leaping accelerate this process, they introduce Compounding Decoding Error (CDE), where discrepancies arise between the true distribution and the approximation from parallel token generation, leading to degraded sample quality. In this work, we present Jump Your Steps (JYS), a novel approach that optimizes the allocation of discrete sampling timesteps by minimizing CDE without extra computational cost. More precisely, we derive a practical upper bound on CDE and propose an efficient algorithm for searching for the optimal sampling schedule. Extensive experiments across image, music, and text generation show that JYS significantly improves sampling quality, establishing it as a versatile framework for enhancing DDM performance for fast sampling.

### 1 Introduction

Diffusion models  have achieved remarkable success in generation tasks within the continuous domain. However, certain modalities, such as text and music, inherently possess discrete features. Recently, discrete diffusion models (DDMs)  have demonstrated performance comparable to state-of-the-art methods in various areas, including text  and image  generation. Nevertheless, like their continuous counterparts, DDMs encounter a significant bottleneck in sampling speed due to their progressive refinement process.

In contrast to continuous-domain diffusion models, where sampling dynamics are driven by sample-wise differential equations , allowing for the direct application of well-established numerical methods to accelerate generation, enhancing speed in DDMs poses a significant challenge. To address this, researchers have proposed fast and efficient samplers, including notable methods such as the $\tau$-leaping  and $k$-Gillespie algorithms , which facilitate parallel sampling of multiple tokens in a single step. However, this parallel but independent sampling introduces Compounding Decoding Error (CDE) , which arises from a mismatch between the training and inference distributions of intermediate latents during parallel sampling. Specifically, while each token is generated according to its marginal distribution, the joint distribution deviates from the learned distribution. To mitigate this issue, the predictor-corrector (PC) sampler  has been proposed. This sampler slightly perturbs the generated data to correct incorrectly generated tokens. However, these methods have limitations, including impracticality under low computational budgets , the need for an additional corrector , or reliance on specialized architectures and loss functions .

Figure: Figure 1: (Top) Comparison of sampling trajectories: ground truth vs. parallel sampling using a uniform schedule and the Jump Your Steps (JYS) schedule. (Bottom) Uniform schedule exhibits compounding decoding errors during parallel sampling, while JYS reduces them by using fewer steps in deterministic phases and reallocating skipped steps to other timesteps.
Refer to caption: x1.png

To reduce CDE and enable fast sampling in DDM fundamentally, we first introduce a rigorous quantity to measure CDE (see Figure 1 Top, and Section 3.1) and propose a novel approach called Jump Your Steps (JYS), which optimizes the allocation of discrete sampling timesteps $\{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode
\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower
-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\dots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{N-1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0\}$ under a fixed total sampling budget $N$ to minimize CDE. Our core idea is to derive efficiently computable bounds for CDE (see Section 3.2 and Section 3.3) and strategically select sampling timesteps by solving minimization problems to reduce these bounds (Section 3.4 and Section 3.5), theoretically ensuring a decrease in the gap between the ground truth distribution and the approximated distribution through parallel sampling (see Figure 2).

Unlike previous methods such as the PC sampler, our approach requires no additional computational resources or modifications to the model architecture or loss function. We empirically validate the effectiveness of our sampling schedule across various datasets, including synthetic sequential data, CIFAR-10 (image), Lakh Pianoroll (music), and text modeling. Our approach accelerates DDM sampling across models using different forward corruption transition kernels, such as uniform, Gaussian, and absorbing transition matrices. Our comprehensive experiments cover both unconditional and conditional generation tasks, consistently showing that optimizing the sampling schedule significantly enhances sampling quality. These results indicate that our method serves as a general framework for speeding up discrete diffusion model sampling.

### 2 Background

#### 2.1 Continuous Time Framework for Discrete Diffusion Models.

Discrete diffusion models (DDMs) define the generative process as the reverse of the data-corrupting forward process, expressed as a Continuous Time Markov Chain (CTMC) on a finite state space $\mathcal{S}$ . For the data-corrupting process $(X_{t})_{t\in[0,T]}$, the density evolution is described as:

$$ $q_{t+dt|t}(y\mid x)=\delta_{xy}+R_{t}(x,y)dt+o(dt)$ (1) $$

Here, $\delta_{xy}$ is the Dirac delta function, $R_{t}\in\mathbb{R}^{S\times S}$ is the transition rate matrix of the forward CTMC, with $S=|\mathcal{S}|$, and $dt>0$. Rate matrices ensure the marginal distribution $q_{t}(x_{t})=\int q_{t}(x_{t}|x_{0})q_{0}(x_{0})dx_{0}$, where $q_{0}=p_{\mathrm{data}}$ and $q_{T}\approx\pi$, the stationary distribution of the forward CTMC. Various transition matrices have been proposed, allowing $\pi$ to follow a uniform or Gaussian distribution, or converting samples into masked tokens.

For generation, we reverse the forward process, moving from the marginal $q_{T}$ back to $p_{\mathrm{data}}$. This time-reversal CTMC is also a CTMC :

$$ $q_{t-dt|t}(y\mid x)=\delta_{xy}+\tilde{R}(x,y)dt+o(dt),$ (2) $$

where the backward transition rate $\tilde{R}$ is defined as:

$$ $\tilde{R}(x,y)=R(y,x)\underbrace{\frac{q_{t}(y)}{q_{t}(x)}}_{\mathclap{\textrm {Score Parametrization}}}=R(y,x)\sum_{x_{0}}\frac{q_{t}(y\mid x_{0})}{q_{t}(x \mid x_{0})}\underbrace{q_{0|t}(x_{0}\mid x)}_{\mathclap{\textrm{Denoising Parametrization}}}$ $$

The literature primarily falls into two parameterizations: Denoising parameterization  approximates a parameterized denoising model as $p^{\theta}_{0|t}(x_{0}|x)\approx q_{0|t}(x_{0}|x)$. Conversely, score parameterization  estimates the ratio of the data distribution as $s_{t}^{\theta}(y|x)={q_{t}(y)}/{q_{t}(x)}$.

#### 2.2 Sampling from the backward CTMC

##### Gillespie’s Algorithm

was proposed as a simulation algorithm for a given CTMC . Gillespie’s algorithm simulates the CTMC by calculating the rate matrix at each state transition. If the rate matrix of the CTMC depends only on the state, Gillespie’s algorithm serves as an exact simulation method. However, since it allows for only one token transition each time the rate matrix is calculated, it is computationally inefficient.

##### k 𝑘 k italic_k -Gillespie’s Algorithm

Instead of updating only one token for each rate matrix calculation, the $k$-Gillespie’s algorithm updates $k$ tokens in parallel. This reduces the computation by a factor of $1/k$ compared to the original Gillespie algorithm.

##### τ 𝜏 \tau italic_τ -Leaping

On the other hand, proposes sampling through $\tau$-leaping. Unlike the $k$-Gillespie algorithm, which update $k$ tokens in parallel, $\tau$-leaping simultaneously updates all tokens according to the given fixed rate matrix within the specified time interval $[t,t+\tau)$. Recently, Tweedie $\tau$-leaping, which considers changes in the rate matrix according to the noise schedule, has been proposed .

### 3 Optimizing the sampling schedule of discrete diffusion models

In this section, we aim to optimize sampling schedule $\{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode
\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower
-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{2}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\dots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{N-1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0\}$ to minimize the Compounding Decoding Error (CDE) introduced by parallel sampling. First, we define and analyze the CDE, examining its relationship to both the sampling schedule (Section 3.1) and sampling quality (Section 3.2). In Section 3.3, we derive an upper bound on the CDE, which serves as the objective for the sampling schedule optimization. Finally, we introduce a hierarchical breakdown strategy (Section 3.4) and computational techniques (Section 3.5) to make the optimization tractable.
Figure 2 summarizes the relationships between the theoretical analyses discussed in this section.

Although this section focuses on samplers based on $\tau$-leaping, all methods are also applicable to the $k$-Gillespie algorithm. For extensions to $k$-Gillespie, please refer to the Algorithm 1.

##### Notations

To begin, we introduce some essential mathematical notation. $X:$ a random variable, $\mathbf{x}:$ its observation, $\mathbb{P},\mathbb{Q}:$ distributions, $\{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode
\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower
-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\dots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{N-1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0\}:$ sampling schedule, and $\mathbb{Q}^{a\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{
\leavevmode\hbox to4.47pt{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0
.0pt\lower-0.58333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }
\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}
\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}
\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto
{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath
\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}b\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt
{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333
pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\cdots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt
{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333
pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}c}:$ the distribution generated by the sampling schedule $\{a\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode
\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower
-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}b\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\cdots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}c\}$. For clarity, when working with backward CTMCs, we slightly abuse notation and express intervals as $[s,t]\triangleq\{u\mid s\geq u\geq t\}$;
the same applies to open and half-open intervals.

Figure: Figure 2: An illustration of the relationship between the KL divergence of the distribution, the compounding error $\mathcal{E}_{\mathrm{CDE}}$ (Section 3.1), and KLUB (Section 3.3). The sampling schedule $\{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode \hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower -0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{ pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333 pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt} \pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{ \set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt }\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{ \set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt }\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt {\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{ rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333 pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt} \pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{ \set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt }\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{ \set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt }\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\dots\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt {\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{ rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333 pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt} \pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{ \set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt }\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{ \set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt }\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{N-1}\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt {\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{ rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333 pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt} \pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{ \set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt }\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{ \set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt }\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0\}$ is optimized to minimize KLUB using the efficient algorithms detailed in Section 3.4, and 3.5.

#### 3.1 Time-Dependent Nature of Compounding Decoding Errors

We introduce a measure for the *Compounding Decoding Error* (CDE), $\mathcal{E}_{\mathrm{CDE}}$, which quantifies the discrepancy between the true joint distribution and the distribution from parallel token generation. For illustration, we consider a discrete process $X_{t}=(X^{1}_{t},X^{2}_{t})$ with sequence length 2, consisting of tokens $X^{1}_{t}$ and $X^{2}_{t}$. The general case is provided in Appendix A.1.

We propose measuring the CDE for a single parallel sampling step $\{s\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode
\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower
-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t\}$ start from $\mathbf{x}_{s}$ by using the KL divergence between the joint distribution $P_{X^{1}_{t},X^{2}_{t}|\mathbf{x}_{s}}$ and the product of marginal distributions $P_{X^{1}_{t}|\mathbf{x}_{s}}\otimes P_{X^{2}_{t}|\mathbf{x}_{s}}$:

$$ $\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t|\mathbf{x}_{s})\triangleq \mathcal{D}_{\mathrm{KL}}(\underbrace{P_{X^{1}_{t},X^{2}_{t}|\mathbf{x}_{s}}}_ {\textrm{True distribution}}\|\underbrace{P_{X^{1}_{t}|\mathbf{x}_{s}}\otimes P _{X^{2}_{t}|\mathbf{x}_{s}}}_{\textrm{Approx.\ distribution from parallel sampling}}).$ (3) $$

We note that the defined CDE is equivalent to the conditional mutual information $\mathcal{I}(X^{1}_{t};X^{2}_{t}|\mathbf{x}_{s})$ of tokens $X^{1}_{t},X^{2}_{t}$:

$$ $\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t|\mathbf{x}_{s})=\mathcal{I}(X^{ 1}_{t};X^{2}_{t}|\mathbf{x}_{s}).$ (4) $$

This expression links the compounding error to the mutual information between tokens; lower mutual information reduces parallel sampling errors. For example, as shown in Figure 1 (Bottom), as generation progresses, the uncertainty of each token decreases over time due to the tokens already generated, reducing mutual information and preventing CDE. In general, CDE depends on the timesteps, and its behavior varies with the data distribution, corruption kernel (see Fig. 9 for illustration), and DDM sampling methods. Motivated by this observation, we hypothesize that we can reduce the CDEs during generation process by optimizing the sampling schedule.

#### 3.2 Relation between Compounding Decoding errors and generation quality

While the Eq. (3) allows us to estimate the CDE starting from a specific state $\mathbf{x}_{s}$, in practice, we are interested in the average compounding error over all possible starting states at time $s$. To assess the overall impact of the CDE when transitioning over the timesteps $s\shortrightarrow t$, we consider the expected value of $\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t|\mathbf{x}_{s})$ with respect to $\mathbf{x}_{s}\sim\mathbb{P}_{s}$. This leads us to consider:

$$ $\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t)\triangleq\mathbb{E}_{\mathbf{x }_{s}}\left[\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t|\mathbf{x}_{s}) \right].$ (5) $$

Consider a sampling schedule $\{T=t_{0}\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{
\leavevmode\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0
.0pt\lower-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }
\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}
\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}
\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\dots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{N-1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0=t_{N}\}$, which will be specified later. Our goal is to minimize the cumulative CDE that arises from each parallel sampling step within the given schedule. If we ignore the accumulated error from the previous steps that affects the consecutive steps, our objective is as follows:

$$ $\min_{t_{1},t_{2},\dots,t_{N-1}}\sum_{i=0}^{N-1}\mathcal{E}_{\mathrm{CDE}}(t_{ i}\shortrightarrow t_{i+1}).$ (6) $$

Interestingly, we find in the following theorem that cumulative CDEs over the sampling schedule can upper bound the KL divergence between the true distribution at time $t=0$, denoted $\mathbb{P}_{0}$, and the distribution $\mathbb{Q}_{0}^{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{
\leavevmode\hbox to4.47pt{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip
0
.0pt\lower-0.58333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }
\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}
\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}
\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto
{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath
\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt
{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333
pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\cdots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt
{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333
pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0}$ obtained from parallel sampling along the sampling schedule (see Appendix A.1 for proof):

###### Theorem 3.1 .

We have the following bound on the KL divergence between $\mathbb{P}_{0}$ and $\mathbb{Q}_{0}^{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{
\leavevmode\hbox to4.47pt{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip
0
.0pt\lower-0.58333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }
\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}
\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}
\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto
{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath
\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt
{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333
pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\cdots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt
{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333
pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0}$ in terms of cumulative CDEs:

$$ $\mathcal{D}_{\mathrm{KL}}(\mathbb{P}_{0}\|\mathbb{Q}_{0}^{T\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{ rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{ rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\cdots\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{ rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0})\leq\sum_{i=0}^{N -1}\mathcal{E}_{\mathrm{CDE}}(t_{i}\shortrightarrow t_{i+1}).$ (7) $$

This theorem suggests that effectively allocating the time schedule to minimize the CDEs can implicitly reduce the discrepancy between the true distribution and the approximate distribution obtained from parallel sampling. Motivated by this, in the next section, we derive a tractable upper bound for
$\sum_{i=0}^{N-1}\mathcal{E}_{\mathrm{CDE}}(t_{i}\shortrightarrow t_{i+1})$ that depends on the sampling schedule $\{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode
\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower
-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\dots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{N-1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0\}$ to facilitate its optimization.

###### Theorem 3.1 .

#### 3.3 Estimating the Compounding Decoding Error Using Girsanov’s Theorem

As shown in Eq. (3), computing $\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t|\mathbf{x}_{s})$ involves determining the KL divergence between the true distribution and the approximated distribution from DDM’s parallel sampler, which is often intractable. To address this, we treat the ground truth reverse process and the sampling process from DDM’s parallel samplers (introduced in Section 2.2) as two CTMCs, starting from the same initial distribution. By applying Girsanov’s theorem , we derive a tractable formula to compare the KL divergence between the distributions of these stochastic processes at any time interval $[s,t]$. We summarize this as the following general theorem applicable to any two backward CTMCs with $R^{1}_{t}$ and $R^{2}_{t}$ as their respective transition rate matrices:

$$ $\begin{cases}\textrm{CTMC 1}:&q^{1}_{u-du|u}(y\mid x)=\delta_{xy}+R^{1}_{t}(x, y)du+o(du),\\ \textrm{CTMC 2}:&q^{2}_{u-du|u}(y\mid x)=\delta_{xy}+R^{2}_{t}(x,y)du+o(du). \end{cases}$ $$

We defer its proof to Appendix A.2.

###### Theorem 3.2 .

(KL-Divergence Upper Bound, KLUB)
Consider an interval $[s,t]$ ($s>t$). If both CTMCs start from the same initial distribution, $\pi_{s}=\mathbb{P}_{s}=\mathbb{Q}_{s}$, then we have:

$$ $\mathcal{D}_{\mathrm{KL}}(\mathbb{P}_{t}\|\mathbb{Q}_{t})\leq\mathcal{D}_{ \mathrm{KL}}(\mathbb{P}_{\mathrm{paths}}\|\mathbb{Q}_{\mathrm{paths}})= \underbrace{\mathbb{E}_{\mathbb{P}_{\mathrm{paths}}}\left[\sum_{i\neq j}\sum_{ \begin{subarray}{c}t<u\leq s\end{subarray}}H_{u}^{ij}\log\frac{R^{1}_{u}(i,j)} {R^{2}_{u}(i,j)}\right]}_{\triangleq~{}\textrm{{KLUB}}_{\pi_{s}}(\mathbb{P}_{t }\|\mathbb{Q}_{t})}.$ (8) $$

Here, $\mathbb{P}_{t}$ and $\mathbb{Q}_{t}$ are the probability distributions at time $t$ resulting from CTMC 1 and CTMC 2, respectively. $\mathbb{P}_{\mathrm{paths}}$ and $\mathbb{Q}_{\mathrm{paths}}$ denote the distributions over their path spaces $(X_{u})_{u\in[t,s]}$, generated by CTMC 1 and CTMC 2, respectively. The indicator function $H_{u}^{ij}$ is defined as $H_{u}^{ij}=1$ if a transition from state $i$ to $j$ occurs at time $u$, and $H_{u}^{ij}=0$ otherwise.

We denote the rightmost term in Eq. (8) as the *Kullback-Leibler Divergence Upper Bound (KLUB)*, which quantifies the mismatch between distributions generated by different CTMCs based on their rate matrices. Theorem 3.2 demonstrates that the difference in trajectories sampled from different CTMCs depends on the ratio of their rate matrices.

Consider CTMC 1 as the ground truth reverse CTMC and CTMC 2 as the reverse CTMC obtained via parallel sampling by substituting the forward transition kernel with the corresponding reverse-in-time kernel. Thus, $\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t|\mathbf{x}_{s})$ can be expressed as the KL divergence between the output distributions of the two CTMCs at time $t$, starting from the initial point $\mathbf{x}_{s}$. This leads to the upper bound (Proof in Appendix A.3):

$$ $\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t)\leq\textrm{{KLUB}}_{\mathbb{P} _{s}}(\mathbb{P}_{t}\|\mathbb{Q}^{s\mathrel{\mathchoice{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t}_{t}),$ (9) $$

where $\mathbb{Q}^{s\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{
\leavevmode\hbox to4.47pt{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0
.0pt\lower-0.58333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }
\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}
\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}
\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto
{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath
\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t}_{t}$ is given by the process discretized at time $s$ and $t$.
This expectation considers the distribution of states at time $s$ and offers a more comprehensive measure of the CDE over the interval $[s,t]$.
Moreover, from the additivity of KLUB, we can bound the sum of CDEs as

$$ $\mathcal{E}_{\mathrm{CDE}}(s\shortrightarrow t)+\mathcal{E}_{\mathrm{CDE}}(t \shortrightarrow u)\leq\textrm{{KLUB}}_{\mathbb{P}_{s}}(\mathbb{P}_{t}\| \mathbb{Q}^{s\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0 .0pt\lower-0.58333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ } \definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0} \pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ } \pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{} \pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto {4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath \pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t}_{t})+\textrm{{ KLUB}}_{\mathbb{P}_{t}}(\mathbb{P}_{u}\|\mathbb{Q}^{t\mathrel{\mathchoice{ \mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to 5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt {\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}u}_{u})=\textrm{{ KLUB}}_{\mathbb{P}_{s}}(\mathbb{P}_{u}\|\mathbb{Q}^{s\mathrel{\mathchoice{ \mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to 5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt {\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{ rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}u}_{u}),$ (10) $$

which shows that the KLUB can be useful for comparing the quality of
discretization of the interval $[s,u]$ with different break point $t$.
We can easily extend this result for the sum of CDEs over the entire sampling schedule $\{T=t_{0}\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{
\leavevmode\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0
.0pt\lower-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }
\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}
\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}\pgfsys@invoke{ }
\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}
\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }
\pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\dots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }
\pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{N-1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }
\pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0=t_{N}\}$.

$$ $\mathcal{D}_{\mathrm{KL}}(\mathbb{P}_{0}\|\mathbb{Q}_{0}^{T\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{ pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ } \pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{ pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ } \pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\cdots\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{ pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ } \pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0})\leq\sum_{i=0}^{N -1}\mathcal{E}_{\mathrm{CDE}}(t_{i}\shortrightarrow t_{i+1})\leq\textrm{{KLUB} }_{\mathbb{P}_{T}}(\mathbb{P}_{0}\|\mathbb{Q}^{T\mathrel{\mathchoice{\mkern 2. 0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{ pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ } \pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\cdots\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{ pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ } \pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{ 0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0} \pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0}_{0}),$ (11) $$

where inequality on the left-hand side comes from Theorem 3.1.

To summarize, optimizing the sampling schedule involves finding a set of timesteps $\{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode
\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower
-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]
{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }
\pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }
\pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}\dots\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }
\pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t_{N-1}\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }
\pgfsys@color@gray@fill{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{
0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@invoke{ }\pgfsys@color@gray@fill{0}
\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0\}$ that minimizes the KLUB on the right-hand side. This approach approximately reduces the cumulative CDE (middle) and provides an upper bound on the KL divergence between the true distribution and the sampled distribution for the given schedule (left-hand side).

###### Theorem 3.2 .

#### 3.4 Feasible Computation with Hierarchical Breakdown Strategy

Using the derived KLUB, we can formulate the timestep search as a minimization problem over KLUB. Here, we employ a hierarchical breakdown strategy, dividing a coarser sampling schedule into a finer one, as shown in Figure 3. Suppose our sampling schedule is given by $\{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode
\hbox to6.39pt{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower
-0.83333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{
pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt
{\vbox to7.98pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333
pt}\pgfsys@lineto{0.0pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{7.14444pt}
\pgfsys@lineto{6.38892pt}{-0.83333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to6.39pt{\vbox to7.98pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.83333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.83333pt}\pgfsys@lineto{0.0pt}{7.14444pt
}\pgfsys@lineto{6.38892pt}{7.14444pt}\pgfsys@lineto{6.38892pt}{-0.83333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-6.3889pt\hbox{
\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.19pt{\vbox to3.99pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.41666pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.41666pt}\pgfsys@lineto{0.0pt}{3.57222pt
}\pgfsys@lineto{3.19446pt}{3.57222pt}\pgfsys@lineto{3.19446pt}{-0.41666pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.19444pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0\}$. Let $\mathbb{Q}_{0}^{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{
\leavevmode\hbox to4.47pt{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip
0
.0pt\lower-0.58333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }
\definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}
\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}
\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto
{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath
\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t\mathrel{
\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt
{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt
\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{
rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }
\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}
\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333
pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext
\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}
\pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox
{\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt
}\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox
{\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{
\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu
\leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{
\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{
\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0}
\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}
{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox
to
0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt
}\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt}
\pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}
}{}
{{}{{}}}{{}{}}{}{{}{}}
{{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.
0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox
{\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}}
\pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}}
{}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope }
\pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}0}$ represent the distribution generated by this schedule. Our goal is to find the optimal $t$ that minimizes cumulative CDE, i.e.,
$\mathcal{E}_{\mathrm{CDE}}(T\shortrightarrow t)+\mathcal{E}_{\mathrm{CDE}}(t
\shortrightarrow 0)$.
This is approximately achievable by minimizing its KLUB upper bound:

$$ $t_{1}=\operatorname*{arg\,min}_{t\in(T,0)}\textrm{{KLUB}}(\mathbb{P}_{0}\| \mathbb{Q}_{0}^{T\mathrel{\mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0 .0pt\lower-0.58333pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ } \definecolor{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0} \pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ } \pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{} \pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto {4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath \pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\displaystyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt{\vbox to5.58pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333pt}\pgfsys@lineto{0.0pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{-0.58333pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@color{$\textstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to3.13pt{\vbox to3.91pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.40833pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.40833pt}\pgfsys@lineto{0.0pt}{3.50076pt }\pgfsys@lineto{3.13055pt}{3.50076pt}\pgfsys@lineto{3.13055pt}{-0.40833pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-3.13055pt\hbox {\set@color{$\scriptstyle\vphantom{+}{\shortrightarrow}$}}}}}\pgfsys@invoke{ \lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}{\mkern 2.0mu \leavevmode\leavevmode\hbox{ \leavevmode\hbox to2.24pt{\vbox to2.79pt{ \pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.29166pt\hbox to0.0pt{ \pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{rgb}{0,0,0} \pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0} {0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to 0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.29166pt}\pgfsys@lineto{0.0pt}{2.50055pt }\pgfsys@lineto{2.23611pt}{2.50055pt}\pgfsys@lineto{2.23611pt}{-0.29166pt} \pgfsys@closepath\pgfsys@clipnext\pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{} }{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-2.23611pt\hbox {\set@color{$\scriptscriptstyle\vphantom{+}{\shortrightarrow}$}}}}} \pgfsys@invoke{\lxSVG@closescope }\pgfsys@endscope}}}} {}{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{\lxSVG@closescope } \pgfsys@endscope\hss}}\lxSVG@closescope\endpgfpicture}}}}}t\mathrel{ \mathchoice{\mkern 2.0mu\leavevmode\leavevmode\hbox{ \leavevmode\hbox to4.47pt {\vbox to5.58pt{\pgfpicture\makeatletter\hbox{\hskip 0.0pt\lower-0.58333pt \hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor{pgfstrokecolor}{ rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ } \pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt} \pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@moveto{0.0pt}{-0.58333 pt}\pgfsys@lineto{0.0pt}{5.0011pt}\pgfsys@lineto{4.47223pt}{5.0011pt} \pgfsys@lineto{4.47223pt}{-0.58333pt}\pgfsys@closepath\pgfsys@clipnext \pgfsys@discardpath\pgfsys@invoke{ }{{{}{}{{}}{} {{}{{}}}{{}{}}{}{{}{}} {{{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1. 0}{0.0pt}{0.0pt}\pgfsys@invoke{ }\hbox{{\leavevmode\hbox{\hskip-4.47223pt\hbox {\set@colo