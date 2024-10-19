---
title: Summary of research papers
subtitle: A brief summary of interesting papers I have come across
summary: A brief summary of interesting papers I have come across
date: '2024-03-20'
lastmod: '2024-03-20'
---

I come across a lot of interesting papers that may or may not be directly related to my research. I will be updating this page with summary of such papers so that I can look back and utilize the findings if necessary.

### Is Your Neural Network at Risk? The Pitfall of Adaptive Gradient Optimizers, 2024
**Summary:** This paper has an interesting observation on the model robustness based on the type of gradient optimizer used during training a model. Their extensive empirical and theoretical analysis shows that models trained with SGD optimizer have high robustness against adversarial perturbatioins than adaptive optimizers like RMSProp and ADAM. 

Why does this happen? The paper performs a frequency analysis to provide the reasoning behind this behavior. It shows that that natural datasets contain some frequencies that do not significantly impact the standard generalization performance of models. However, these irrevalent information will make a model vulnerable to adversarial perturbation depending on the type of optimizer used. 


### Feature Purification: How Adversarial Training Performs Robust Deep Learning, 2021
**Summary:** The paper discusses the impacts of adversarial training on a model that makes it robust against adversarial perturbations. The adversarial perturbation arises because of the accumulation of dense mixtures in the hidden weights during model training. The goal of adversarial training is to remove such mixtures and putrify hidden weights. THe paper also proves that training a model only over natural data will produce a modl that is non-robust to adversarial perturbations, and adversarial training even with a weaker attack like FGSM can increase provable robustness against such adversarial perturbations. 

### Position: Explain to Question not to Justify, ICML 2024 
**Summary:** This position paper divides the XAI research into human/value-oriented explanations (BLUE XAI) and model/validation-oriented explanations (RED XAI) and argues that the area of RED XAI is currently under-explored. We need more methods that question models, spotting and fixing bugs in faulty models. The authors also argue that it is unrealistic and harmful to design explanations for end-users understanding. Instead of serving to end users, “it is wiser to design explanations to empower model developers. If we want to make progress toward safe AI, then we need new techniques for exploring and debugging models to be used by AI professionals.”

The authors also deconstruct some fallacies associated with XAI: 

Fallacy 1: “Interpretability is a binary concept and models can be divided into interpretable vs. black boxes.” While in the literature, models such as linear regression or decision trees are often called transparent, the authors argue that such division is untrue as both tree or linear models can be difficult to analyze if they are based on a very large number of variables, especially on real world applications where they are based on thousands of 
of variables. 

Fallacy 2: “Single XAI silver bullet exists and we just need to find this single best XAI method”:  Different methods explain different components in a model making different assumptions so there is no almighty explanation method. 

Fallacy 3: “The illusion of a “true explanations”. This is something I have been thinking a lot lately. The quality of explanations, cannot be judged, just by seeing what the expert answer should be with that problem domain. What if the mismatch between explanation and the ground truth may not be due to a bad method of explanation, but, for example, to a bad model itself? 


