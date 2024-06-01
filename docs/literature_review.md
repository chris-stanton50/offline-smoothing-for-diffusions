# Literature Review

**Gradually add to this document as you read stuff. Try to make the explanations reasonably compact!**

We review the literature on Monte Carlo methods for inference of discretely observed Stochastic Differential equations. These observations may be either with or without noise.

## Maximum Likelihood Estimation of Diffusions 

We assume that we have the (in general, time inhomogeneous) SDE:

$$dX_t = b_\theta(t, X_t)dt + \sigma_\theta(t, X_t) dW_t$$ 

Where we choose $b, \sigma, x_0, T$ to satisfy the required Lipschitz and linear growth conditons for any value of $\theta$. We observe the SDE at times $0=t_0<t_1<\dots<t_n$. There exists an (up to a.s unique) solution to the SDE: $(X_t)_{t\in [0,T]}$. This solution is an Ito Diffusion. The observations are thus $X_{t_1}, \dots, X_{t_n}$. 

The log-likelihood is given by:

$$l_n(\theta) = \sum_{i=1}^n log[p_{\theta, t_{i-1}, t_{i}}(x_{t_i}| x_{t_{i-1}})]$$

However, the transition densities $p_{\theta, t_{i-1}, t_{i}}(x_{t_i}| x_{t_{i-1}})$ are intractable. It is proposed to approximate the likelihood function for a given value of $\theta$ via Monte Carlo methods:

For any $N$, one can contstruct an approximation of $p_{\theta, t_{i-1}, t_{i}}(x_{t_i}| x_{t_{i-1}})$, as:

$p_{\theta, t_{i-1}, t_{i}}(x_{t_i}| x_{t_{i-1}}) \approx p_{\theta, t_{i-1}, t_{i}}^N(x_{t_i}| x_{t_{i-1}})$

The approximate density converges to the true density as $N \rightarrow \infty$. The approximate density is given by the integral:

$$p_{\theta, t_{i-1}, t_{i}}^N(x_{t_i}| x_{t_{i-1}}) = \int_{\mathcal{X}} \prod_{j=1}^N p^{EM}_{t_{i-1} + (j-1)\frac{\Delta t_i}{N}, \frac{\Delta t_i}{N}}(x_j | x_{j-1})dx_{1:j-1}$$

Where $x_0 = x_{t_{i-1}}, x_N = x_{t_i}$, and $p^{EM}_{t_{i-1} + (j-1)\frac{\Delta t_i}{N}, \frac{\Delta t_i}{N}}(x_j | x_{j-1})$ is the analytically tractable implied transition density of the Euler-Maruyama scheme, with step size $\frac{\Delta t_i}{N}$ at time $t_{i-1} + (j-1)\frac{\Delta t_i}{N}$.

The above integral is intractable. Monte Carlo methods are the most feasible method of evaluating this high-dimensional integral. [Pedersen (1995)](https://www.jstor.org/stable/4616340?searchText=&searchUri=&ab_segments=&searchKey=&refreqid=fastly-default%3A98284e9365e4bbcf8964130271fdf984) proposes the following standard Monte Carlo estimator:

$$\hat{p}_{\theta, t_{i-1}, t_{i}}^{N, K}(x_{t_i}| x_{t_{i-1}}) = \frac{1}{K}\sum_{k=1}^K p^{EM}_{t_{i} - \frac{\Delta t_i}{N}, \frac{\Delta t_i}{N}}(x_{t_i} | \tau_k)$$

Where the random variable $\tau_k$ are generated i.i.d from applying $N-1$ Euler-Maruyama transitions. These proposals are in a sense 'blind', in that they do not assume knowledge of $x_{t_i}$. Knowledge of this observation is important as the function that one is attempting to integrate via Monte Carlo methods is large at values that are close to $x_{t_i}$. Bearing this in mind, [Durham and Gallant (2002)](https://www.tandfonline.com/doi/abs/10.1198/073500102288618397) propose instead using importance sampling with various bridge proposals to estimate $p_{\theta, t_{i-1}, t_{i}}^N(x_{t_i}| x_{t_{i-1}})$. This has the effect of significantly reducing the variance of the Monte Carlo estimators, for fixed $N$. The effect of using the bridge proposals is exemplified by implemention on a CIR model, for which the transition densities are known. 


## Note:

In the introductory paper on ancestral sampling, Linsden and Schon suggest that PGAS is favourable relative to PGBS, in a framework where the models are non-Markovian: that is that we have a latent process $x_{1:t}$ with distribution defined by the kernels:

$$P[dx_t, x_{1:t-1}] = p(x_t | x_{1:t-1})dx_t$$

And we have observation densities:

$$P[dy_t, x_t] = p(y_t| x_t)dy_t$$

To formulate such a model as a Feynman-Kac model, this would require that the state space on which the measures are defined is extended to higher dimensions. Such an extension will result in there being Dirac measures in the transition kernels of the resulting Feynman-Kac model. According to the implications in the SMC book, backward sampling/ancestral sampling both become degenerate when the transition kernels $M[dx_t, x_{t-1}]$ contain dirac measures. This should be investigated further.

An exercise to try:

1. Understand in greater detail the PGBS/PGAS algorithms, along with the invariance proofs for targeting a general parametric Feynman-Kac measure.

2. Have a go at understanding the exact construction of a Feynman-Kac model for the type of model given above.   

The utility of this to the research is it would allow us to understand exactly why it is that when the observations are not equidistant in time, it is still feasible to implement e.g the backward step.  