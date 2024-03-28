---
title: Summary of research papers
subtitle: A brief summary of interesting papers I have come across
summary: A brief summary of interesting papers I have come across
date: '2024-03-20'
lastmod: '2024-03-20'
---

I come across a lot of interesting papers that may or may not directly be related to my research. I will be updating this page with summary of such papers so that I can look back and utilize the findings if necessary. 

### Is Your Neural Network at Risk? The Pitfall of Adaptive Gradient Optimizers, 2024
**Authors:** Avery Ma, Yangchen Pan, and Amir-massoud Farahmand, Vector Institute
**Summary:** This paper has an interesting observation on the model robustness based on the type of gradient optimizer used during training a model. Their extensive empirical and theoretical analysis shows that models trained with SGD optimizer have high robustness against adversarial perturbatioins than adaptive optimizers like RMSProp and ADAM. 

Why does this happen? THe paper performs a frequency analysis to provide the reasoning behind this behavior. It shows that that natural datasets contain some frequencies that do not significantly impact the standard generalization performance of models. However, these irrevalent information will make a model vulnerable to adversarial perturbation depending on the type of optimizer used. 


