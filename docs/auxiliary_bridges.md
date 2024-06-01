# Auxiliary Bridges

We recall the modelling problem of interest: to infer a latent SDE given noisy observations at discrete points in time. 

Instead of attempting to directly infer the posterior distribution of the Ito process $X_{[0, T]}$, we apply a specifically chosen invertible (measureable) function to the Ito process: $\Phi$.

$$\Phi^{{(DH)}}: C([0,T], \mathbb{R}^{d_x}) \rightarrow \Bigl[ C([0,\Delta t], \mathbb{R}^{d_x}) \times \mathbb{R}^{d_x}\Bigr]^n$$


$$\Phi^{{(DH)}}(X_{[0, T]}) = X'_{1:n}$$

Each $X'_i = (Z_i, X_i), i \in [n]$ is a random variable in $C([0,\Delta t], \mathbb{R}^{d_x}) \times \mathbb{R}^{d_x}$. 

$$Z_i = F^{(DH)}_{t_{i-1}, t_i}(X_{i-1}, X_i, X_{[t_{i-1}, t_i]})$$
$$X_i = X_{t_i}$$

The posterior distribution of the transformed random variables: $X'_{1:n}|Y_{1:n}=y_{1:n}$ is expressible (in multiple ways) as a Feynman-Kac measure. For more details, including an exposition of the posterior distribution, see the [function space model](./function_space_model.md) document.

The transition kernels of $X'_i | X'_{i-1}$ are given by $P_i[dx'_i, x'_{i-1}]$

In this document, we focus in greater detail on:

- What exactly the kernel is.
- How to generate simulations from the kernel.
    - This will include a discussion on what the function $F_{s, t}^{(DH)}$ is, and how to evaluate it.
- How to evaluate the density of the kernel.

What is the kernel?

The joint kernel $P_i[dx'_i, x'_{i-1}]$ can be expressed as the product of the conditional kernels $P_i[dx'_i, x'_{i-1}] = P_{i, 2}[dz_{[t_{i-1}, t_i]} , x_{i-1:i}]P_{i, 1}[x_i, x_{i-1}]$

Where we have:
$$P_{i, 1}[dx_i, x_{i-1}] = p_{t_{i-1}, t_i}(x_i | x_{i-1})dx_i$$

The density $p_{t_{i-1}, t_i}(x_i | x_{i-1})$ is the intractable transition density between 2 points on an SDE. The density is with respect to the Lebesgue measure.

The probability measure $P_{i, 2}[dz_{[t_{i-1}, t_i]}, x_{i-1:i}]$ for fixed $x_{i-1}, x_i$ is the image measure of $\mathbb{P}_{x_{i-1}, x_i}^{t_{i-1}, t_i}$ with the transform $F_{t_{i-1}, t_i}^{(DH)}(x_{i-1}, x_i,\quad  \cdot  \quad )$ applied.

The probability measure $\mathbb{P}_{x_{i-1}, x_i}^{t_{i-1}, t_i}$ is given by the conditional distribution of $X_{[t_{i-1}, t_i]}| X_{t_{i-1}} = x_{t_{i-1}}, X_{t_i} = x_{t_i}$. By the Markov property for SDEs, one can show that the solution of the SDE:

$$d\tilde{X}_t = b(t+t_{i-1}, \tilde{X}_t)dt + \sigma(t+t_{i-1}, \tilde{X}_t)dW_t$$

Assuming that $\tilde{X}_0 = x_{t_{i-1}}$, conditioned on $\tilde{X}_{\Delta_t} = x_{t_i}$ has the same distribution. We define a probability measure $\mathbb{Q}_{x_{i-1}, x_i}^{t_{i-1}, t_i}$ to be the distribution of the solution of the SDE:

$$d\tilde{X}_t = \bigg[b(t+t_{i-1}, \tilde{X}_t) + \frac{x_{t_i} - \tilde{X}_t}{\Delta t-t}\bigg]dt + \sigma(t+t_{i-1}, \tilde{X}_t)dW_t$$

The transform $F_{t_{i-1}, t_i}^{(DH)}(x_{i-1}, x_i,\quad  \cdot  \quad )$ is the invertible map from the solution of the above SDE to the driving noise. Delyon and Hu show that the probability measures $\mathbb{P}_{x_{i-1}, x_i}^{t_{i-1}, t_i}$ and $\mathbb{Q}_{x_{i-1}, x_i}^{t_{i-1}, t_i}$ are absolutely continuous with respect to each other, with a known Radon-Nikodym derivative. By definition, the image measure of $\mathbb{Q}_{x_{i-1}, x_i}^{t_{i-1}, t_i}$ with the transform $F_{t_{i-1}, t_i}^{(DH)}(x_{i-1}, x_i,\quad  \cdot  \quad )$ applied is the parameter free Weiner measure $\mathbb{W}$. Therefore, the image measure of $\mathbb{P}_{x_{i-1}, x_i}^{t_{i-1}, t_i}$ with the transform $F_{t_{i-1}, t_i}^{(DH)}(x_{i-1}, x_i,\quad  \cdot  \quad )$ applied has a known density with respect to the parameter free Weiner measure that can be evaluated pointwise.

We now consider how to simulate from the kernel, and how to evaluate the density of the kernel.

### Kernel Simulation

To simulate from the joint kernel $P_i[dx'_i, x'_{i-1}] = P_{i, 2}[dz_{[t_{i-1}, t_i]} , x_{i-1:i}]P_{i, 1}[x_i, x_{i-1}]$, it is sufficient to be able to simulate from each of the conditional kernels $P_{i, 1}[x_i, x_{i-1}]$ and $P_{i, 2}[dz_{[t_{i-1}, t_i]} , x_{i-1:i}]$.

If one can simulate from $X_{[t_{i-1}, t_i]} | X_{t_{i-1}} = x_{i-1}$ exactly, then by taking the end point of the simulated function, we have generated a simulation from $P_{i, 1}[dx_i, x_{i-1}]$. By then applying the transform $F_{t_{i-1}, t_i}^{(DH)}(x_{i-1}, x_i,\quad  \cdot  \quad )$ to the simulated path, we have simulated from the kernel $P_{i, 2}[dz_{[t_{i-1}, t_i]} , x_{i-1:i}]$. In practice, it is not possible to generate a simulation of a path on (uncountably) infinite dimensional space, so, necessarily, the simulation of $X_{[t_{i-1}, t_i]} | X_{t_{i-1}} = x_{i-1}$ is done approximately by simulating from a finite-dimensional marginal distribution. The application of the function $F_{t_{i-1}, t_i}^{(DH)}(x_{i-1}, x_i,\quad  \cdot  \quad )$ is also done approximately. In practice, both of these approximations are based on the Euler-Maruyama numerical approximation scheme for the solution of an SDE. 

### Kernel Density Evaluation

To do:

 
- Conclude what the form of the transition kernel is, and hence how to simulate from it, and evaluate its density.
- Outline where the function $F_{s, t}^{(DH)}$ would be used in the practical algorithm, and its inverse.
- Outline where the density would be used. 


Results:


The conditional distribution $X'_i | X_{i-1}=x_{i-1}$ is given by the

### Scrap Notes

Say that we have the Stochastic Differential Equation:

$$dX_t = b(t, X_t)dt + \sigma(t, X_t) dW_t$$

Where $b, \sigma$ satify the linear growth and Lipschitz continuity conditions.
Given some $T>0$ and $X_0=x_0$, one can define an a.s unique strong solution to the SDE, given by: $X_{[0,T]}$.

If we pick two points in time: $t_0 < t_1$, $\Delta t = t_1 - t_0$ the function-valued random variable $X_{[t_0, t_1]}$, if conditioned on $X_{t_0} = x_0$ and $X_{t_1} = x_1$, will have a distribution (say $\mathbb{P}_{x_0, x_1}^{t_0, t_1}$) that is absolutely continuous with respect to the distribution of the random variable $\tilde{X}_{[0, \Delta t]}$, where $\tilde{X}$ is the solution of the auxiliary SDE:

$$d\tilde{X}_t = [b(t_0 + t, \tilde{X}_t) + \frac{x_1 - \tilde{X}_t}{\Delta t - t}dt] + \sigma(t_0 + t, \tilde{X}_t) dW_t$$

Where we set $\tilde{X}_0 = x_0$. We denote the distribution of $\tilde{X}_{[0, \Delta t]}$ as $\mathbb{Q}_{x_0, x_1}^{t_0, t_1}$. In fact, $\mathbb{P}_{x_0, x_1}^{t_0, t_1}$ and $\mathbb{Q}_{x_0, x_1}^{t_0, t_1}$ are equivalent. Delyon and Hu (2006) derive an analytical expression for the Radon-Nikodym derivative between $\mathbb{P}_{x_0, x_1}^{t_0, t_1}$ and $\mathbb{Q}_{x_0, x_1}^{t_0, t_1}$.

Both the $X_t$ and the $\tilde{X}_t$ processes are adapted to $(\mathcal{F}_t)_{t>0}$: the natural filtration of the underlying Weiner process $(W_t)_{t>0}$. Therefore (by Doob-Dynkin), there exist (measureable) functions $F_{t_0, t_1}(x_0, x_1, \tilde{X}_{[0, \Delta t]}) = W_{[0, \Delta t]}$ and $F^{-1}_{t_0, t_1}(x_0, x_1, W_{[0, \Delta t]}) = \tilde{X}_{[0, \Delta t]}$

These functions $F_{t_0, t_1}$ and $F^{-1}_{t_0, t_1}$ are the auxiliary bridge transform and the inverse auxiliary bridge transforms respectively. We test the implementation of these functions for univariate SDEs.