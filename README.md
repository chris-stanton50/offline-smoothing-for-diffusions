# Offline Smoothing for Diffusion Processes

The purpose of this repo is to contain the implementation of PMCMC methods applied to conduct joint Bayesian inference of latent state and parameters of Bayesian stochastic differential equations, that possibly also have jumps, that have been observed with noise.

We begin by considering the case of a standard Ito SDE without jumps, and then extend to the case where the SDE has jumps.

## Implementation

The trick of using the result shown by Delyon and Hu (2006) results in a Feynman-Kac model for the posterior distribution of the latent process. Thus, it is possible to apply any SMC filtering and/or smoothing method to conduct inference in this class of models. For joint smoothing, the state of the art offline approaches are those that combine SMC and MCMC: namely Particle MCMC methods. The state of the art online approach is SMC^2. For implementing these methods, it makes sense to extend the `particles` package that accompanies the book [An Introduction to Sequential Monte Carlo](https://nchopin.github.io/books/). The package contains an abstraction of State Space models and Feynman Kac models. Once the models of interest (i.e SDES observed with noise)have been expressed in this form, through the package a whole host of MCMC/PMCMC methods can be implemented. It will be necessary to either ask N Chopin to extend the package to contain PGBS/PGAS sampling methods, or to do this yourself.

- Extend the package in the SDEs without jumps case, using the naive discretisation approach. This will need to be extended to the case with jumps.
- Extend the package in the SDEs without jumps case, to implement the infinite dimensional filtering algorithm on function space.
- Look into the case with jumps and think about how these extensions would work, in the context of the ones that you are already building.

## Inference in SDEs without jumps

For the case of SDEs without jumps, upon resorting to a discretisation, the posterior distribution has a density that is smooth (i.e continuously differentiable), and known up to a normalising constant. Thus, gradient based MCMC methods (such as MALA/HMC) are possible. However, these methods do not adhere to the Roberts-Stramer critique, namely that the performance of any inference algorithm is stable as the step size decreases. 

One alternative approach to inference on latent SDEs in continous time is given  by [Manifold Markov chain Monte Carlo methods for Bayesian inference in a wide class of diffusion models](https://arxiv.org/abs/1912.02982). This method does adhere to the Roberts Stramer critique, and is the state-of-the-art approach to inference in Diffusions observed with noise, for the continous time case. It also performs well when the observations are highly informative of the data, or when the process is observed directly.

Matt Graham has implemented this method in the following [repo](https://github.com/thiery-lab/manifold-mcmc-for-diffusions) as an extension of the [mici](https://github.com/matt-graham/mici) package. The mici package uses [autograd](https://github.com/HIPS/autograd), which is the predecessor to [JAX](https://jax.readthedocs.io/en/latest/), to compute automatic derivatives. The mici package implements HMC methods, so it would not be appropriate to extend this package for the implementation of the proposed filtering approaches. 

One can also use Particle MCMC methods for inference in diffusion models without jumps. If one were to implement the PMMH with the bootstrap filter, then in most practical models where the noisy observations only depends on the process at a single point in time, the variance of the weights would remain constant as the step size decreases, thus also adhering to the Roberts-Stramer critique. *Guided proposals would need to be chosen very carefullly to avoid a curse of dimensionality issue.* (To do: Think about how guided proposals would be chosen when discretising, to mitigate the impact of a curse of dimensionality). As such proposals are 'blind' in the sense that they do not see the data, their performance can degenerate in the case where the observations are highly informative. It is possible to implement Particle Gibbs, along with backward and ancestral sampling extensions. However, the backward sampling approach becomes degenerate, and in practice would reduce to standard PG. (To do: check what happens to the ancestral sampling case).

The PMMH case has been considered and extended to the cPMMH and augmented cPMMH schemes in [Augmented pseudo-marginal Metropolis–Hastings for partially observed diffusion processes](https://arxiv.org/abs/1912.02982).


## Inference in SDEs with Jumps

Have more of a think about the case where there are jumps involved. The most immediate competitor would be the standard PMMH. 
