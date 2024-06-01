# Building Function Space Feynman-Kac formulae

The ultimate programming goal is to be able to define a `FeynmanKac` object that specifies the Feynman Kac measure given by the posterior distribution given in the [function space model](./function_space_model.md). We need to define:

- `M0`: Function that generates random variables from the initial proposal distribution. Output must be in the form of random variables that have been generated by a `StructDist` object.
- `M`: Function that generates random variables from the proposal kernel. Output must be in the form of random variables that have been generated by a `StructDist` object.
- For the first implementation, this will require the construction of a simulator that generates an approximate simulation of an SDE skeleton. The end point is retained and the stored as a particle. The history of the simulation is tranformed using $F(x_0, x_T, X)$ as implied by the Delyon and Hu auxiliary bridge.

- `logG`: Logarithm of the potential at each time point t. 

- `logpt`: Logarithm of the conditional density from transition kernel $M_t[dx_t, x_{t-1}]$

There are multiple ways in which valid Feynman-Kac measures can be defined that (implicitly, through a transform) target the posterior distribution of a partially observed SDE:

- Different auxiliary bridge processes can be used to transform the solution of the SDE into a form that is expressible as a Feynman-Kac measure. The standard approach is to use the Delyon and Hu auxiliary bridge. However, other auxiliary bridges are possible.
- Given a choice of auxiliary bridge process, we have now defined a target posterior distribution. This posterior distribution can be expressed as a Feynman-Kac measure in multiple ways. The simplest would be a bootstrap formalism, however one can also use guided or 'non-blind' proposals that are informed by the data. Use of the auxiliary formalism may also be possible.

### How should the sample paths be stored when implemented in SMC?

When the SMC algorithm is running, there are two possibilities for how the generated particles (from the `M` method of the Feynman-Kac object) can be stored:

1. As $(Z, X_i)$ particles
2. As $(X, X_i)$ particles

Storage method 1 would be consistent with how the Feynman-Kac model is formally defined. Storage method 2 would be used for numerical convenience, particularly for the particle filtering step.

We assume that the bootstrap filter is used. One could argue that it would be better to use storage method 2., as this would reduce the compute cost of generating the particles (within the `M` method). Further, evaluating the potentials `G` to obtain the weights may involve the entire sample path from the SDE. If the particles are stored via storage method 1, then the inverse transform would need to be applied to obtain the $X$ process from the stored $Z$ processes. This would result in unnecessary compute cost.

However, by not transforming the paths into the form in which the Feynman-Kac model is formally defined, problems will arise when trying to use the SMC algorithms as built within the package, when implementing smoothing methods. To resolve these issues when using storage method 2, it would be necessary to either:

- Change the particle history object by applying the transform to the $(X, X_i)$ paths to obtain the $(Z, X_i)$ paths after the particle filter has been run.
- Change the Particle Gibbs implementation, so that 

This is highly undesirable, as we would like the constructed Feynman-Kac models to 'plug and play' into all of the SMC methods that are available within the `particles` package. 

Further, in the majority of practical cases, the noisy observations are only dependent on the value of the SDE at the same point in time, thus there is no need to transform back to the $X$ processes from the $Z$ processes. 

Conclusion: We will focus on attempting to do the implementation using this method.


Although the underlying Feynman-Kac model consists of paths that have been transformed, for the practical algorithm, when the bootstrap filter is being used, it is not necessary to actually transform the paths. The particles can be simply be stored as simulations from the latent SDE. This will be more efficient when calculating the weights. When doing backward smoothing, using the transform function will then become necessary. (Paths should be stored in $X$ form)

If the proposal is guided, then the proposal part should generate the sample paths and then transform them into the implied latent process simulation. This form can then be used for each particle to evaluate the weights. (Paths should be stored in $X$ form)

So, whether guided or standard, our implementation should store the particles as the implied latent SDE paths. This is more convenient for the evaluation of the potentials, in the general case of dependence on the path.

If we have a guided case where the observation density does not depend on the path (only on the end point), and the auxiliary bridge has a density that is not path dependent, then in this case only it would not be necessary to transform the paths. the entire algorithm, including the smoothing, could be run without transfroming the paths. (Paths should be stored in $Z$ form)

To do: For now, just build the `PartiallyObservedSDE` object.

Have a think about how 

## Objects to Build

Given these possible approaches to constructing an appropriate Feynman-Kac measure, we will define the following objects

- `PartiallyObservedSDE`: This will be a subclass of `SDE`, and will include the observation density of the noisy data. The observation density will take the SDE path between the the observation times as input. The simulate method will generate data from the latent SDE, and observations. One may also need to define proposals as methods for this class. This would be consistent with the approach taken for standard state space models.
- `AuxiliaryBridge`: The auxiliary bridge object will be used to detail possible transforms to the SDE solution to create a posterior distribution that is expressible as a Feynman-Kac model. The most standard example is the Delyon and Hu auxiliary bridge. An auxiliary bridge is characterised by:
    - The density of the bridge transformed SDE path and end point, which is absolutely integrable w.r.t $\mathbb{W} \otimes dt$. If a guided proposal is used, then this will be used in the particle filter. It is used anyway for particle smoothing methods. (**Think about how this density depends on $t$ for a time inhomogeneous case**)
    - The transform that is applied to simulations from the SDE to obtain simulations from the bridge transformed path. The inverse of this transform is also useful for obtaining the 


When given a choice of auxiliary bridge, partially observed SDE and proposal distribution (that is absolutely integrable w.r.t $\mathbb{W} \otimes dt$), one can define uniquely a Feynman-Kac model to simulate from. This can be built by subclassing `FeynmanKac`, and having an `__init__` method which takes an instance of a `PartiallyObservedSDE` as input. Within this `__init__` method, there should be an option as to whether the simulations are transformed (using the proposal bridge transform) or not. When the bootstrap filter is being used, there is no need for the paths to be transforms in a forward pass of the data. When a guided proposal is being used, 

For the construction of proposal distributions, it may also be useful to construct special path-point type distribution objects, representing distributions that are absolutely integrable with respect to $\mathbb{W} \otimes dt$. We could split out these objects into two types: 

- One of product measures of a path and an end point, where both are independent. These examples can be constructed by using the standard Girsanov Theorem to build path measures that are absolutely integrable w.r.t the Weiner measure, then taking a particular choice of path.
- One in which there are dependendencies between the r.v respresenting the path, and the r.v representing the end point. These are generally constructed through transforms from diffusion bridges.

## Notes



Have a think about:

- How the framework can be extended to time-inhomogeneous SDEs.
- Whether one can use backward sampling methods in the case where $\Delta t_i = t_i - t_{i-1}$ is not constant. This will be important for the write-up of the thesis, but not for the practical implementation because of the overhead of effort.

We need to consider simulators that will perform well in high dimensions.

*Note* that the cost of the algorithm will scale linearly with the number of points using to simulate the approximate SDE solution. Shouto's paper uses only 10 points to make the algorithm feasible!

The objects that will need to be created are:

- SDEs
    - Has the methods `b` and `sigma` which define the drift and diffusion coefficient respectively. 
    - Connects with Numerical Schemes through **simulate** method.
    - Connects with BridgeTransforms.  
- Numerical Methods (Euler-Maruyama, Milstein, Taylor 1.5 etc)
    - Takes an SDE as input.
    - Has a `distribution` method, which constructs a `StructDist` object. This object 
- SDESSMs
    - Initialised with an SDE, is an abstract base class.
    - Build observation densities on top of the ABC.
- AuxiliaryBridges
    - d

## Details of Particle Gibbs implementation:

`MCMC` --> `RWMH` --> `PMMH`

`MCMC` --> `GenericGibbs` --> `ParticleGibbs`

The existing implementation of the `ParticleGibbs` class within the `mcmc.py` module takes as input a state space model class, and a Feynman-Kac formalism (e.g Bootstrap/Guided/Auxiliary) to convert the state space model class into a Feynman-Kac model. 

The method `fk_mod` takes as input a parameter value `theta`, and converts it into the required Feynman-Kac model, by instantiating the ssm class using the given value of `theta`, then creating the Feynman-Kac model using the FK formalism class (e.g `ssm.Bootstrap`).  

The method `update_theta` needs to be written by the user, and in practice this will be a Metropolis-within-Gibbs step to update the value of theta given $x$. If we wanted to be fancy, a gradient-based sampler using automatic derivatives could be implemented here.

The method `update_states` creates a Feynman-Kac model  for the input value of $\theta$ using the method `fk_mod`, and runs a CSMC particle filter given $\theta$ and the previous starred trajectory. The backward step is implemented to generate a new value of $x$. This backward step requires that the transition kernel: $M_t^\theta[dx_t, x_{t-1}]$ has a density that can be evaluated pointwise up to a normalising constant. This density will need to be implemented in the `logpt` function of the Feynman Kac model.

To implement Particle Gibbs on the SDEs, the approach will be to subclass the `ParticleGibbs` class. This subclass will need to take as input the newly created `sdessm` classes, and the formalisms that convert the `sdessms` into Feynman-Kac models. The `fk_mod` function will likely need to be re-written so that it accounts for the new classes.

Possible implementation issues:

Note possible mistake in line 421 of `mcmc.py` of `particles` package:

`self.x = self.update_states(self.chain.theta[n-1], self.x)`

from context should be `theta[n]` not `theta[n-1]`.

Also, on line 489, we have the following: 

`new_x = cpf.hist.backward_sampling(1)`

Many types of backward sampling methods are implemented for the `ParticleHistory` class, however `backward_sampling` is not one of them, and it is not clear how a specific method is chosen from the code.

For the implementation, we should start with taking the existing SDE class that you've built, and adding implementations of the Delyon and Hu density and the Delyon and Hu transforms applied to skeletons. Next, create a SDEssm class, and a Feynman Kac boostrap formalism class to convert it all into the correct form. You will have then provided everything necessary to run the algorithm.