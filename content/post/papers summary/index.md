---
title: Summary of research papers
subtitle: A brief summary of interesting papers I have come across
summary: A brief summary of interesting papers I have come across
date: '2024-04-28'
lastmod: '2024-04-28'
---

I come across a lot of interesting papers that may or may not directly related to my research. I will be updating this page with summary of such papers so that I can look back and utilize the findings if necessary. 

### Is Your Neural Network at Risk? The Pitfall of Adaptive Gradient Optimizers, 2024

Paper: [Link](https://openreview.net/pdf?id=ed8SkMdYFT)
Authors: Avery Ma, Yangchen Pan, and Amir-massoud Farahmand, Vector Institute 

Summary: ochastic gradient descent (SGD) and adaptive gradient methods, such as Adam and
RMSProp, have been widely used in training deep neural networks. We empirically show
that while the difference between the standard generalization performance of models trained
using these methods is small, those trained using SGD exhibit far greater robustness under input perturbations. Notably, our investigation demonstrates the presence of irrelevant frequencies in natural datasets, where alterations do not affect models’ generalization
performance. However, models trained with adaptive methods show sensitivity to these
changes, suggesting that their use of irrelevant frequencies can lead to solutions sensitive
to perturbations. To better understand this difference, we study the learning dynamics
of gradient descent (GD) and sign gradient descent (signGD) on a synthetic dataset that
mirrors natural signals. With a three-dimensional input space, the models optimized with
GD and signGD