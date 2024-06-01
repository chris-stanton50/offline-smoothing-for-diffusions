# The Discretised Model: Deriving the Posterior Distribution

## Latent SDE, without jumps

We have the probability space $(\Omega, \mathcal{F}, \mathbb{P})$, that is equipped with a $d_w$-dimensional Brownian motion, denoted $(W_t)_{t\geq0}$. 

- Functions $b$, $\sigma$ that satisfy the regularity conditions (assuming a particular dimension $d_x$).
- Initial point, $X_0=x_0$
- An end point in time: $T$

We then immediately obtain an Ito Diffusion $(\tilde{X}_t)_{t\in[0,T]}$ on the probability space that the is the unique (up to a.s) solution of the implied SDE.

We will bring in say $n$ noisy observations at (wlog) equidistant discrete points in time. The common distance is $\Delta t = \frac{T}{n}$.

Bearing this in mind, we now pick the number of points on the SDE in between observations to approximate using a numerical scheme, as $m$. This implies a step size: $\delta = \frac{\Delta t}{m} = \frac{T}{nm}$ in between observations.

We then construct a numerical approximation to the SDE solution as the following $nm$ random variables:

$$X_i = f(X_{i-1}, \Delta W_i) \qquad i \in [nm], \qquad \Delta W_i = W_{i\delta} - W_{(i-1)\delta}$$

Each random variable $X_i$ is of dimension $d_x$. The function $f$ is chosen specifically so that it is a valid numerical scheme (e.g Euler-Maruyama, Millstein, Higher order Taylor etc).

This also implicitly defines $n$ random variables, each of dimension $md_x$:

$$X'_i = (X_{m(i-1)+1},  \dots X_{mi}) \qquad i \in [n] \qquad$$

This construction implies the following joint distribution:

$$\mathbb{P}[X'_{1:n} \in dx_{1:n}] = P[dx'_n, x'_{n-1}]\dots P[dx'_2, x'_1]P_1[dx_1]$$

Where the transition kernel $P[dx', x]$ is given by:

$$P[dx'_2, x'_1] = Q[dx_{2m}, x_{2m-1}]\dots Q[dx_{m+1}, x_m]$$

It is possible to sample from the transition kernel $Q$, which is uniquely determined by choice of numerical approximation scheme $f$. In the special cases of the Euler-Maruyama and the Millstein scheme, the kernel has a density w.r.t the Lebesgue measure of appropriate dimension. Since it is possible to generate samples from $Q$, one can also generate samples from $P$.

We now introduce the observations $(Y_{t_1}, \dots, Y_{t_n})$ each of dimension $d_y$ and let $Y_i = Y_{t_i}$, $y_i = y_{t_i}$. We then have that:

$$P[Y_{t_i} \in dy_{i}|Y_{t_j}= y_j: j<i, X'_{1:n}= x'_{1:n}] \overset{a.s}{=}  P[Y_{t_i} \in dy_{i} | X'_i, Y_{t_{i-1}}] \overset{a.s}{=} g_\theta(y_i| x_{i-1}, y_{i-1})dy_i$$

A standard example of how to construct said random variables $Y_i$ in practice is by:

$$Y_i = g(X_{mi}) + \epsilon_i \quad \epsilon_i \sim \mathcal{N}_{d_y}(0, \Sigma_Y)$$

We would then have that $Y_i | X'_i=x'_i \sim \mathcal{N}(g(x_{mi}), \Sigma_Y)$, which has a density provided that $\Sigma_Y$ is positive definite. Note that we assume here that the dominating measure is Lebesgue, however one could also consider discrete observations where the dominating measure is the counting measure.

This is also an example of a standard case that is most common in practical applications, where $Y_i$ depends only on $X_{mi}$, which is the numerical approximation of $\tilde{X}_{t_i}$.

Under these assumptions, the joint distribution of $(X'_{1:n}, Y_{1:n})$ is given by:

$$\mathbb{P}[X'_{1:n} \in dx_{1:n}, Y_{1:n} \in dy_{1:n}] = g(y_1|x'_1)\prod_{i=2}^n g(y_i | y_{i-1}, x'_i) dy_{1:n} P[dx_{1:n}]$$

Thus, the posterior distribution of $X'_{1:n} | Y_{1:n}=y_{1:n}$ is given by:

$$\mathbb{P}[X'_{1:n} \in dx_{1:n} | Y_{1:n} = y_{1:n}] = \frac{1}{p(y_{1:n})}g(y_1|x'_1)\prod_{i=2}^n g(y_i | y_{i-1}, x'_i) P[dx_{1:n}]$$

This is a probability distribution on the $nmd_x$ dimensional space. Note the following:

- For the simplest numerical approximation scheme (Euler-Maruyama, Millstein), we have that $Q[dx_2, x_1]$ has known density w.r.t the $d_x$ dimensional Lebesgue measure. Thus, the transition kernel $P[dx'_2, x'_1]$ will also have a known density with respect to the $md_x$ dimensional Lebesgue measure. Thus the posterior distribution has unnormalised density that can be evaluated pointwise. Provided that the observation densities $g$ are differentiable, this enables MALA and HMC, and guided proposals for PMCMC methods.

- One can use a non-centred parameterisation, and conduct inference on the increments $\Delta W_i$ s as opposed to the $X_i$ s. This would improve the performance of an HMC/MALA scheme by reducing the correlations between the components. A particle filtering scheme is not possible in this case (without suffering from the curse of dimensionality).

This scheme introduces the following sources of bias:

- Bias from using a numerical scheme instead of either sampling (in the case of SMC) or using the intractable true transition density (in the case of HMC)
- Possible bias from evaluating the observation density $p(y_i | x'_i)$ by interpolating in between the points on $x'_i$ to obtain a function. In practice this will not happen often, as $Y_i$ will usually only depend on $X_{mi}$. 

## Latent SDE, with jumps

We have the probability space $(\Omega, \mathcal{F}, \mathbb{P})$, that is equipped with a $d_w$-dimensional Brownian motion, denoted $(W_t)_{t\geq0}$, and compound Poisson Process $(J_t)_{t\geq 0}$. The compound Poisson process is given by $J_t = \sum_{i=1}^{N_t} \zeta_i$, where $(N_t)_{t>0}$ is a Poisson process with intensity function $\lambda(.)$, $\zeta_i$ are i.i.d random variables with Lebesgue density $h(.)$. We specify, in a particular dimension $d_x$:

- Functions $b$, $\sigma$ that satisfy the regularity conditions (assuming a particular dimension $d_x$).
- Initial point, $X_0=x_0$
- An end point in time: $T$

We then obtain an Ito jump diffusion that solves the jump SDE (**note: more understanding of this is needed**): $(\tilde{X}_t)_{t\geq0}$.

Note that the Poisson process is univariate - therefore the current exposition is limited to the case where there the are simultaneous jumps across dimensions. In terms of model formulation, there is great scope for generalisation.

As in the case of the SDE without jumps, we will bring in say $n$ noisy observations at (wlog) equidistant discrete points in time. The common distance is $\Delta t = \frac{T}{n}$.

We again pick the number of points on the SDE with jumps in between observations to approximate using a numerical scheme, as $m$. This implies a step size: $\delta = \frac{\Delta t}{m} = \frac{T}{nm}$ in between observations.

We then construct a numerical approximation to the SDE solution as the following $nm$ random variables:

$$X_i = f(X_{i-1}, \Delta W_i)+ \Delta N_i\zeta_i \qquad i \in [nm]$$

Where we have that:
$$\Delta W_i = W_{i\delta} - W_{(i-1)\delta}, \quad \Delta N_i \overset{i.i.d}{\sim} Bernoulli(\lambda((i-1)\delta)\cdot \delta)$$

This is one possibility for a numerical scheme that approximates the solution to an Ito jump diffusion, whereas in practice there are multiple possible options. See [Platen and Bruti-Liberati (2011)](https://link.springer.com/book/10.1007/978-3-642-13694-8) for details.

We ultimately aim to evaluate the posterior distribution of $X_{1:nm}$. However, it may be easier to design an algorithm for different parameterisations. With this in mind, we define the random variables:

$$T_i = f(\tilde{X}_{i-1}, \Delta W_i) \quad U_i = \Delta N_i \zeta_i, \quad i \in [nm]$$

We then obtain the following recursion:

$$T_i = f(T_{i-1}+ \Delta N_{i-1}\zeta_{i-1},\Delta W_i)$$

We could then, instead of considering the posterior distribution of $X_{1:nm}$, instead consider the posterior of:

- $T_{1:mn}, U_{1:nm}$

We can only use PMMH.

For this choice, the distributions of the $U_is$ are spike-slabs: this could motivate the use of PDMPs?

- $T_{1:mn}, \Delta N_{1:nm}, \zeta_{1:nm}$

For this choice, the transitions will have a tractable transition density with respect to the appropriate product of the counting measure and the Lebesgue measure.

The addition of the discrete variables mean that it is not possible to use Hamiltonian Monte Carlo.

Standard PMMH is possible, as is standard PG. PGBS would, in practice, reduce to standard PG.





