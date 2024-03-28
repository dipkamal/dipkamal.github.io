---
title: MLE and MAP
date: '2022-02013'
summary: Notes on MLE and MAP
---

Maximum Likelihood Estimation is a method to find such parameters of a data distribution that maximizes the likelihood of observing that data. Formally, 

$$
\theta_{MLE}=arg max_\theta~P(D;\theta) = arg max_\theta ~ log P(D;\theta)
$$

So, there are two steps in obtaining parameters from MLE:
1. We explicitly assume a data distribution to represent the data. 
2. Obtain the parameters of that distribution by maximizing the likelihood of observing the data in hand. 

Let us take a coin-toss example. Let us consider we have following data outcomes: 

$$ 
D=\{H,T,T,H,H,H,T,T,T,T\}
$$

A coin toss event that has two outut states can be represented by a Bernoull distribution. 

$$ 
Ber(x|\theta)=\theta^{Nh}~(1-\theta)^{Nt}
$$

where, $Nh$ and $Nt$ represents number of head and number of tail. In our case, 

$$ 
Nh=4, Nt=6
$$

Using our MLE estimation formula, 

$$ 
\theta_{MLE}=argmax_\theta~log P(D;\theta)
$$

$$
= arg max_\theta~log~\theta^{Nh}~(1-\theta)^{Nt}
$$

$$ 
= arg max_\theta~log\theta^{Nh}+log(1-\theta)^{Nt}
$$

$$ 
= arg max_\theta~Nh~log\theta+Nt~log(1-\theta)\\
$$

Differentiate above equation with respect to $\theta$ and equate it to zero, we obtain,

$$ 
\theta= \frac{Nh}{(Nh+Nt)}
$$

Substituting the values from our experiment, we obtain, 
$$
\theta=4/(4+6)=0.4
$$

However, MLE is not suitable when number of samples is small. For example: again consider the coin toss experiment, lets assume we toss it four times and obtain the sample $\{H,H,H,H\}$. In this example, we obtain $\theta=1$. Is this the correct likelihood of getting head in a fair coin?

That's where MAP or Maximum a Posteriori Probability Estimation comes into play. Here, we find such parameters $\theta$ that maximizes the probability distribution $P(\theta | D)$. Formally, 

$$
\theta_{MAP}=arg max_\theta~P(\theta|D) \\ 
= arg max_\theta ~ log P(D|\theta) + log P(\theta)
$$

Here, 
$P(\theta)$ expressed our prior knowledge on the outcome. This acts a regularizer to the model. If $P(\theta)$ is a Gaussian distribution, it is L2 regularization. If $P(\theta)$ is a Laplace distribution, it is L1 regularization. If $P(\theta)$ is a Bernoulli distribution, it is L0 regularization. 

Let us take the coin toss example again.

Probability of an event is how likely that event is going to happen. If we toss a coin, if it's a fair coin, the probability of heads is 0.5, $\theta=0.5$. Prior knowledge tries to encode this knowledge. Most common choice for priori of Bernoulli distribution is a Beta distribution. So, Beta distribution is the probability distribution of the probability. That means, it will explain how likely it is for theta to take value between $0$ and $1$. Formally, beta distribution is a continuous probability distribution of probabilities of random variables that have a finite set of values. 

Beta distribution has two parameters, $\alpha$ and $\beta$. If the value that the random variable of a beta distribution $\theta$ can take is between $a$ and $b$ instead of $(0,1)$, we call it a general beta distribution or, $4$ parameters distribution.  

$$
P(\theta | \alpha, \beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)*\Gamma(\beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

Beta distribution is used to model the prior belief. $\alpha$ and $\beta$ models that belief.

So,ignoring the normalizing constant of the Beta distribution, we obtain our MAP estimate as:
$$
\theta_{MAP}=arg max_\theta~P(\theta|D) 
$$

$$
= arg max_\theta ~ log P(D|\theta) + log P(\theta)
$$

$$ = arg max_\theta ~ Nh~log(\theta) + Nt~log(1-\theta)+(\alpha-1)~log(\theta)+(\beta-1)~log(1-\theta)
$$

$$
= arg max_\theta ~ (Nh+\alpha-1)~log(\theta)+(Nt+\beta-1)~log(1-\theta)
$$

Differentiating wrt $ \theta$, we obtain: 
$$
\theta_{MAP}=\frac{Nh+\alpha-1}{Nt+Nh+\beta+\alpha-2}
$$


Let us take the same $\{H,H,H,H \}$ coin-toss example. Now, we encode our prior as $\alpha=100$ and $\beta=100$ Here, 

$$ 
\theta=50.99
$$

This value makes some sense now for a coin toss. Doesnot it? 

However, as the size of sample tends to grow to infinity, the parameter estimation of MLE and MAP will be similar. And remember, if our prior belief is wrong, MAP will still give a wrong answer. Consider $\alpha=100$ and $\beta=10$ in the same experiment. We have a different estimation. 

There's another approach for parameter estimation. It is called a Bayesiam approach which is a general formulation for parameter estimation. MLE and MAP are just special cases of Bayesian inference. In Bayesian distribution, we are interested in the distribution of the parameter $\theta$ itself over all hypothesis space. In Bayesian approach, we use the posterior predictive distribution to directly make prediction about the output label $Y$ of given sample $X$. 

$$
P(Y|D,X)=\int_{\theta}^{} P(Y,\theta | D,X)d\theta = \int_{\theta}^{} P (Y|\theta, D,X)P(\theta|D) d\theta
$$



<b>References</b>
- [Machine Learning: A Probabilistic Perspective](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf) 
- [Cornell Class CS4780 by Prof. Kilian Weinberger](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/.html)

