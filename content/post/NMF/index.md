---
title: Understanding Non-Negative Matrix Factorization
subtitle: A brief overview of matrix factorization method called NMF
date: '2024-02-27'
math: true
---

Matrix factorization is a powerful technique used in various fields such as machine learning, data analysis, and recommender systems. At its core, matrix factorization involves decomposing a matrix into multiple matrices, typically of lower rank, in order to extract meaningful patterns and latent features. By representing the original matrix in terms of its constituent parts, matrix factorization enables dimensionality reduction, noise reduction, and capturing underlying structure within the data.

**What is NMF?**

Non-negative Matrix Factorization (NMF) is a specialized form of matrix factorization that imposes additional constraints on the decomposition process. Unlike traditional matrix factorization methods, NMF requires that all elements in the original matrix and the resulting factor matrices be non-negative. This constraint makes NMF particularly useful for data that naturally occurs in non-negative spaces, such as images, text documents, and audio signals.

In NMF, the original matrix $X$ is factorized into two matrices, $W$ and $H$, such that

$$
X \approx WH
$$

Where:
- $X \in R^{M\times N}_+$
- $W \in R^{M\times r}_+$
- $H \in R^{r \times N}_+$


The $r$ is called the approximation rank. The key characteristic of NMF is that all elements of $W$ and $H$ are constrained to be non-negative, which allows for intuitive interpretation of the resulting factors as additive parts.

***What does this approximation signify?*** 

The above equation for NMF can be rewritten column wise as $x \approx Wh$, where $x$ and $h$ are corresponding columns of the matrix $X$ and $H$. This indicates that, with this factorization, we are approximating the data vector $x$ by a linear combination of the columns in $W$ weighted by the components in $h$. This means that the matrix $W$ contains of new basis vector which is optimized for the linear approximation of the original data in $X$. Good approximation can be achieved if the basis vectors are able to discover underlying structure from the data. 

**Solving the optimization**

To sole this optimization, we need to define a cost function that we can minimize using optimization technique like gradient descent. One of the most common cost function is Frobenius norm, where the distance between the original data vector and the approximation is computed as the Frobenius norm between the matrices. 

$$
D(X \mid WH) = \| X - WH \|^2_F
$$

where, $\| A - B\|^2_F = \sum_{ij} (A_{ij} - B_{ij})^2$. 

Hence, our optimization problem for NMF can be defined as: 

$$
\{W^*, H^* \} = \argmin_{W\geq 0,H \geq 0} \frac{1}{2} \| X - WH \|^2_F
$$


We cannot minimize this cost function jointly with respect to both $W$ and $H$, hence, an alternating technique is used. Minimization is performed for each varirable separately at each iteration, keeping the other one fixed.


 Lets compute $\nabla_W \frac{1}{2} \| X - WH \|^2_F$ first and then $\nabla_H \frac{1}{2} \| X - WH \|^2_F$. We keep $H$ fixed to compute the update rule for $W$. 
 

$$
\nabla_W \frac{1}{2} \| X - WH \|^2_F = \frac{1}{2} \nabla_W Tr \left[ (X^T - H^T W^T) (X - WH) \right]
$$

Here Tr is the trace operator. We can use trace operator because  $\| A \|_F^2 = Tr A^T A$.

$$
\nabla_W \frac{1}{2} \| X - WH \|^2_F = \frac{1}{2} \nabla_W Tr \left[ X^T X - X^T WH - H^T W^T X + H^T W^T W H\right] \\
$$

$$
\nabla_W \frac{1}{2} \| X - WH \|^2_F = \frac{1}{2} \nabla_W Tr \left[ - X^T WH - H^T W^T X + H^T W^T W H\right] \\
$$
$$
\nabla_W \frac{1}{2} \| X - WH \|^2_F = - X H^T + W H H^T
$$

The update rule for this matrix is given by
$$
W_{ij} \leftarrow W_{ij} + \eta_{ij} (X H^T - W H H^T)_{ij}
$$


The NMF multiplicative rule by Lee and Seung uses following $\eta$

$$
\eta_{ij} = \frac{W_{ij}}{(W H H^T)_{ij}}
$$

This modifies the update rule as: 
$$
W_{ij} \leftarrow W_{ij} \frac{(X H^T)_{ij}}{(W H H^T)_{ij}}
$$


Similar to $H$, we can derive the update rule by keeping $W$ fixed as: 

$$
\nabla_H \frac{1}{2} \| X - WH \|^2_F = \frac{1}{2} \nabla_H Tr \left[ (X^T - H^T W^T) (X - WH) \right]
$$
$$
\nabla_H \frac{1}{2} \| X - WH \|^2_F = \frac{1}{2} \nabla_H Tr \left[ X^T X - X^T WH - H^T W^T X + H^T W^T W H\right]
$$

$$
\nabla_H \frac{1}{2} \| X - WH \|^2_F = \frac{1}{2} \nabla_H Tr \left[ - X^T WH - H^T W^T X + H^T W^T W H\right]
$$

$$
\nabla_H \frac{1}{2} \| X - WH \|^2_F= -W^T X +  W^T W H
$$

The gradient descent update is given by:

$$
\mu_{ij} (W^T X - W^T W H)_{ij}
$$

with $\mu$ picked as: 
$$
\mu_{ij} = \frac{H_{ij}}{(W^T W H)_{ij}}
$$

which gives the update rule as:

$$
H_{ij} \leftarrow H_{ij} \frac{(W^T X)_{ij}}{(W^T W H)_{ij}}
$$


<b>Reference papers:</b>
- [Algorithms for NMF](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf)
- [Learning the parts of objects by NMF](https://www.nature.com/articles/44565)
- [Solving NMF](https://www.almoststochastic.com/2013/06/nonnegative-matrix-factorization.html)

