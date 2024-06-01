# The Function Space Model: Deriving the Posterior Distribution

## Introduction - Model Specification

We have the probability space $(\Omega, \mathcal{F}, \mathbb{P})$, that is equipped with a $d_w$-dimensional Brownian motion, denoted $(W_t)_{t\geq0}$. We specify:

- Functions $b(t, x)$, $\sigma(t, x)$ that satisfy the regularity conditions (assuming a particular dimension $d_x$).
- Initial point, $X_0=x_0: \quad x_0 \in \mathbb{R}^{d_x}$
- An end point in time: $T$

We then immediately obtain an Ito Diffusion $(X_t)_{t\in[0,T]}$ on the probability space that is the unique (up to a.s) solution of the implied Stochastic Differential Equation:

$$dX_t = b(t, X_t)dt + \sigma(t, X_t) dW_t$$

 This Ito Diffusion has unique distribution $\mathbb{X}_{x_0, T}$ on the space of continuous functions $C([0,T], \mathbb{R}^{d_x})$, and is adapted to $(\mathcal{F}_t)_{t\geq0}$: the natural filtration of the Brownian motion.

We have $n$ noisy observations at (wlog) equidistant discrete points in time. The common distance is $\Delta t = \frac{T}{n}$, thus we have observations at times $t_1, \dots, t_n$, where $t_i=i \Delta t$.

When deriving the posterior distribution of the latent SDE, one can immediately discretise and derive a posterior distribution on finite-dimensional space. Alternatively, one can consider the latent SDE as a random variable on function space, and obtain a posterior distribution based on this interpretation. Designing an algorithm to approximate integrals w.r.t a distribution on function space will necessarily involve finite-dimensional approximations. We focus here solely on the derivation of the posterior distribution of the model, and not on the algorithms that could be implemented to conduct inference on the models.

## Posterior Distribution derivation

Instead of attempting to directly infer the posterior distribution of the Ito process $X_{[0, T]}$, we apply a specifically chosen invertible (measureable) function to the Ito process: $\Phi$.

$$\Phi^{{(DH)}}: C([0,T], \mathbb{R}^{d_x}) \rightarrow \Bigl[ C([0,\Delta t], \mathbb{R}^{d_x}) \times \mathbb{R}^{d_x}\Bigr]^n$$


$$\Phi^{{(DH)}}(X_{[0, T]}) = X'_{1:n}$$

Each $X'_i = (Z_i, X_i), i \in [n]$ is a random variable in $C([0,\Delta t], \mathbb{R}^{d_x}) \times \mathbb{R}^{d_x}$. 

$$Z_i = F^{(DH)}_{t_{i-1}, t_i}(X_{i-1}, X_i, X_{[t_{i-1}, t_i]})$$
$$X_i = X_{t_i}$$


Under the standard assumptions on the distributions of the observations $Y_{1:n}$, we obtain the following posterior distribution for $X'_{1:n}|Y_{1:n}=y_{1:n}$:

$$\mathbb{P}[X'_{1:n} \in dx'_{1:n}|Y_{1:n} = y_{1:n}] = \frac{1}{p(y_{1:n})} \prod_{i=1}^n p_{i}(y_i|x'_{i-1}, x'_i, y_{i-1})P[dx'_{1:n}]$$


Where $p_i(y_i|x_{i-1}, x_i, y_{i-1})$ are the conditional densities each with a given $\sigma$-finite dominating measure.

$P[dx'_{1:n}]=\mathbb{P}[X'_{1:n} \in dx'_{1:n}]$ is the Markov measure:

$$P[dx'_{1:n}] = P_n[dx'_n, x'_{n-1}]\dots P_{x_0}[dx'_1]$$

Under **Assumption 1**, it is possible to generate samples from the transition kernel $P$. Further, by considering the kernels componentwise, one can further deduce that:

$$P_i[dx'_i, x'_{i-1}] = p_i(x'_i|x'_{i-1})\mu(dx'_i)$$

Where $\mu = \mathbb{W} \otimes Leb^{\otimes d_x}$. The densities $p_i$ can be evaluated pointwise, and are given by:

$$p_i(x'_i | x'_{i-1}) = \frac{|\Sigma(t_i, x_i)|^{1/2}}{|\Sigma(t_{i-1}, x_{i-1})|^{1/2}}\mathcal{N}_{d_x}(x_2; x_1, \Delta t \Sigma(t_{i-1}, x_{i-1})) \psi_{t_{i-1}, t_i}(F_{t_{i-1}, t_i}(z_i, x_{i-1}, x_i), x_{i-1}, x_i)$$


- (**To do: verify that the above expression is the correct generalisation of the density to the time-inhomogeneous case**)
- (**To do: when looking at the implementation of the density, can add more details here**)

We can therefore express the posterior distribution in the following form:

$$\mathbb{P}[X'_{1:n} \in dx'_{1:n}|Y_{1:n} = y_{1:n}] = \frac{1}{p(y_{1:n})} p_1(x'_1)p_1(y_1 | x'_1)\prod_{i=2}^n p_i(x_i | x_{i-1} )p_i(y_i|x'_{i-1}, x'_i, y_{i-1}) [\otimes_{i=1}^n\mu(dx'_{i})]$$

## Extensions

### Using different transforms to obtain different posterior distributions

In the derivation given above, we have obtained a posterior distribution for $X'_{1:n}|Y_{1:n}=y_{1:n}$, where $X'_{1:n}$ = $\Phi^{(DH)}(\tilde{X}_{[0, T]})$. This (invertible, measureable) transform from the Ito process to the particles $X'_{1:n}$ is based on the Delyon and Hu auxiliary bridge. This auxiliary bridge is chosen because it results in transition kernels between the particles $P_i[dx_i, x_{i-1}]$ that are absolutely integrable with respect to common dominating measure. 

In a seperate document, we will outline:

- What exactly the transforms $\Phi^{(DH)}$ and $F_{s, t}^{(DH)}(x_s, x_t, x_{[s,t]})$ are, and why it is the case that this transform results in random variables with transition kernels that have densities with respect to a common dominating measure.
- How to construct alternative transforms using different auxiliary bridges, that will result in different posterior distributions with respect to a common dominating measure.

### Using different Feynman-Kac formalisms to implement different SMC algorithms

Given a particular choice of transform of the Ito process based on a particular auxiliary bridge, say $\Phi^{(DH)}$, $F_{s, t}^{(DH)}$ respectively, we obtain a particular posterior distribution for which we could like to approximate integrals. The natural way to express this posterior distribution as a Feynman-Kac measure is to use the bootstrap formalism: this is to generate particles from the unconditional distribution of the signal, and weight the particles using the observation densities. 

However, since the transition kernels $P_i[dx_i, x_{i-1}]$ have densities that can be evaluated with respect to a common dominating measure, it is possible to use any proposal distribution to generate the particles, as long as it is possible to simulate from this proposal, and the proposal is absolutely integrable with respect to a common dominating measure $\mu$, with known density. One can signficantly improve performance of inference algorithms (by reducing the variance of the weights) by using proposals that use the information in the observed data. How to choose these proposals is a topic for further study.




