---
title: Summary of research papers
subtitle: A brief summary of interesting papers I have come across
summary: A brief summary of interesting papers I have come across
date: '2024-03-20'
lastmod: '2024-03-20'
---

I come across a lot of interesting papers that may or may not be directly related to my research. I will be updating this page with summary of such papers so that I can look back and utilize the findings if necessary.

### Is Your Neural Network at Risk? The Pitfall of Adaptive Gradient Optimizers, 2024
**Authors:** Avery Ma, Yangchen Pan, and Amir-massoud Farahmand, Vector Institute
**Summary:** This paper has an interesting observation on the model robustness based on the type of gradient optimizer used during training a model. Their extensive empirical and theoretical analysis shows that models trained with SGD optimizer have high robustness against adversarial perturbatioins than adaptive optimizers like RMSProp and ADAM. 

Why does this happen? The paper performs a frequency analysis to provide the reasoning behind this behavior. It shows that that natural datasets contain some frequencies that do not significantly impact the standard generalization performance of models. However, these irrevalent information will make a model vulnerable to adversarial perturbation depending on the type of optimizer used. 


### Feature Purification: How Adversarial Training Performs Robust Deep Learning, 2021

**Authors:** Zeyuan Allen-Zhu, Yuanzhi Li,  Microsoft Research Redmond, Carnegie Mellon University
**Summary:** The paper discusses the impacts of adversarial training on a model that makes it robust against adversarial perturbations. The adversarial perturbation arises because of the accumulation of dense mixtures in the hidden weights during model training. The goal of adversarial training is to remove such mixtures and putrify hidden weights. THe paper also proves that training a model only over natural data will produce a modl that is non-robust to adversarial perturbations, and adversarial training even with a weaker attack like FGSM can increase provable robustness against such adversarial perturbations. 

### Interpretable machine learning: Fundamental principles and 10 grand challenges, 2022 

**Authors:** 
**Summary:** 