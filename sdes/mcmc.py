from particles.mcmc import MCMC, GenericRWHM, PMMH, CSMC
from particles.mcmc import ParticleGibbs, GenericGibbs
from particles.kalman import Kalman
import particles.state_space_models as ssms
import particles.smc_samplers as ssp
import particles.distributions as dists
from particles.variance_mcmc import MCMC_variance
from sdes.continuous_discrete_ssms import CDSSM
from sdes.feynman_kac import CDSSM_SMC, CDSSM_FeynmanKac, BootstrapReparameterisedDA_DH
from sdes.tools import init_kwargs_dict
import numpy as np

class GenericGibbs(GenericGibbs):

    def step(self, n):
        """ Added to correct mistake in particles package."""
        self.chain.theta[n] = self.update_theta(self.chain.theta[n - 1], self.x)
        self.x = self.update_states(self.chain.theta[n], self.x)
        if self.store_x:
            self.chain.x[n] = self.x

class ParticleGibbs(ParticleGibbs):
        
    def step(self, n):
        """ Added to correct mistake in particles package."""
        self.chain.theta[n] = self.update_theta(self.chain.theta[n - 1], self.x)
        self.x = self.update_states(self.chain.theta[n], self.x)
        if self.store_x:
            self.chain.x[n] = self.x

class IMMH(GenericRWHM):
    """
    Implementation of IMMH: Ideal Marginal Metropolis-Hastings.
    Marginal MCMC algorithm that targets theta for Linear Gaussian 
    State space models.Likelihood is computed using Kalman filter.
    """    
    def __init__(
        self,
        niter=10,
        verbose=0,
        lgssm_cls=None,
        prior=None,
        data=None,
        theta0=None,
        adaptive=True,
        scale=1.0,
        rw_cov=None,
    ):    
        """
        Parameters
        ----------
        niter: int
            number of iterations
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        lgssm_cls: MVLinearGauss class
            the considered parametric class of linear, gaussian state-space models.
            Must be a subclass of MVLinearGauss, so that one can implement the Kalman
            filter.
        prior: StructDist
            the prior
        data: list-like
            the data
        theta0: structured array of length=1
            starting point (generated from prior if =None)
        adaptive: True/False
            If true, random walk covariance matrix is adapted recursively
            based on past samples; see also scale and rw_cov for extra info.
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38^2 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array (defaults to Identity matrix if not provided)
            covariance matrix of the random walk proposal if adaptive=False;
            if adaptive=True, rw_cov is used as a preliminary guess for the
            covariance matrix of the target.
        """
        self.lgssm_cls = lgssm_cls
        self.prior = prior
        self.data = data
        generic_rwhm_kwargs_dict = init_kwargs_dict(GenericRWHM, locals())
        GenericRWHM.__init__(self, **generic_rwhm_kwargs_dict)

    def alg_instance(self, theta):
        return Kalman(ssm=self.lgssm_cls(**theta), data=self.data)
    
    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            kf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            kf.filter()
            self.prop.lpost[0] += np.sum(kf.logpyt)

class MetropoliswithinGibbs(GenericRWHM):
    """
    Use within Gibbs samplers for automated parameter updates. Not to be used alone.
    
    Note: Could make this class more efficient by not storing all of the theta particles,
    as the ones we are interested in will be stored in the underlying Gibbs sampler.
    """
    def __init__(
        self, niter=10, verbose=0, ssm_cls=None, prior=None, data=None, theta0=None, adaptive=True, scale=1.0, rw_cov=None
    ):
        """
        Parameters
        ----------
        niter: int
            number of MCMC iterations
        verbose: int (default=0)
            progress report printed every (niter/verbose) iterations (never if 0)
        ssm_cls: StateSpaceModel class
            the considered parametric class of state-space models
        prior: StructDist
            the prior
        data: list-like
            the data
        theta0: structured array of size=1 or None
            starting point, simulated from the prior if set to None
        adaptive: True/False
            If true, random walk covariance matrix is adapted recursively
            based on past samples; see also scale and rw_cov for extra info.
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38^2 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array (defaults to Identity matrix if not provided)
            covariance matrix of the random walk proposal if adaptive=False;
            if adaptive=True, rw_cov is used as a preliminary guess for the
            covariance matrix of the target.
        """
        self.ssm_cls = ssm_cls
        self.prior = prior
        self.data = data
        generic_rwhm_kwargs_dict = init_kwargs_dict(GenericRWHM, locals())
        GenericRWHM.__init__(self, **generic_rwhm_kwargs_dict)

    def loglik(self, theta, x):
        ssm = self.ssm_cls(**{k: theta[k] for k in theta.dtype.names})
        loglik = ssm.PX0().logpdf(x[0]) + ssm.PY(0, None, x[0]).logpdf(self.data[0][0])
        for t in range(1, len(self.data)):
            loglik += ssm.PX(t, x[t-1]).logpdf(x[t]) + ssm.PY(t, x[t-1], x[t]).logpdf(self.data[t][0])
        return loglik
    
    def compute_post(self):
        self.prop.lpost = self.prior.logpdf(self.prop.theta) + self.loglik(self.prop.theta, self.x)


class AutoGibbs(GenericGibbs):
    """
    Gibbs sampler for state space models. Automatically updates the 
    parameter by using a Metropolis-within-Gibbs step.
    
    Must be subclassed with the method `update_states` defined.    
    """
    def __init__(
        self,
        niter=10,
        verbose=10,
        theta0=None,
        ssm_cls=None,
        prior=None,
        data=None,
        store_x=False, 
        adaptive=True,
        scale=1.0,
        rw_cov=None,
        N_steps=1,
    ):  
        generic_gibbs_kwargs_dict = init_kwargs_dict(GenericGibbs, locals())
        GenericGibbs.__init__(self, **generic_gibbs_kwargs_dict)
        self.N_steps = N_steps
        self.theta0 = self.prior.rvs(size=1) if theta0 is None else theta0 # Assign theta0 early so it can be passed to MWG
        mwg_kwargs_dict = {**init_kwargs_dict(MetropoliswithinGibbs, locals()), **{'niter': self.N_steps*niter, 'theta0': self.theta0}}
        self.mwgibbs = MetropoliswithinGibbs(**mwg_kwargs_dict)
        self.n_mwg = 0

    def update_theta(self, theta, x):
        self.mwgibbs.x = x # Pass the current value of x to the MWG
        for _ in range(self.N_steps):
            if self.n_mwg == 0:
                self.mwgibbs.step0()
            else:
                self.mwgibbs.step(self.n_mwg)
            self.n_mwg += 1
        theta = self.mwgibbs.chain.theta[self.n_mwg - 1]
        return theta

class AutoParticleGibbs(ParticleGibbs, AutoGibbs):
    """
    Implementation of Particle Gibbs that automatically updates theta
    using a Metropolis-within-Gibbs step. Can be used without subclassing.
    """
    def __init__(
        self,
        niter=10,
        verbose=0,
        ssm_cls=None,
        prior=None,
        data=None,
        theta0=None,
        Nx=100,
        fk_cls=None,
        regenerate_data=False,
        backward_step=False,
        store_x=False,
        adaptive=True,
        scale=1.0,
        rw_cov=None,
        N_steps=1
        ):
            autogibbs_kwargs = init_kwargs_dict(AutoGibbs, locals())
            AutoGibbs.__init__(self, **autogibbs_kwargs)
            self.Nx = Nx
            self.fk_cls = ssms.Bootstrap if fk_cls is None else fk_cls
            self.regenerate_data = regenerate_data
            self.backward_step = backward_step
        
class CDSSM_MCMC(MCMC):
    
    def _check_ssm_and_fk(self, cdssm_cls, fk_cls):
        if not issubclass(cdssm_cls, CDSSM):
            raise TypeError('cdssm class must be a subclass of CDSSM.')
        if not issubclass(fk_cls, CDSSM_FeynmanKac):
                raise TypeError('fk_cls must be a subclass of CDSSM_FeynmanKac.')
            
    def _build_cdssm(self, cdssm_options, theta):
        if cdssm_options is not None:
            return self.cdssm_cls(**{**cdssm_options, **ssp.rec_to_dict(theta)})
        else:
            return self.cdssm_cls(**ssp.rec_to_dict(theta))

class CDSSM_PMMH(PMMH, CDSSM_MCMC):

    def __init__(self, num=10., cdssm_options=None, **kwargs): # This code doesn't assign any default values. Could do this the other way around.
        PMMH.__init__(self, **kwargs)
        self.cdssm_options = cdssm_options
        self.num = num

    def alg_instance(self, theta):
        if self.cdssm_options is not None:
            cdssm = self.ssm_cls(**{**self.cdssm_options, **theta})
        else:
            cdssm = self.ssm_cls(**theta)
        return self.smc_cls(
                            fk=self.fk_cls(cdssm=cdssm, data=self.data), 
                            N=self.Nx, 
                            num=self.num,
                            **self.smc_options
                            )
        
class CDSSM_CSMC(CSMC, CDSSM_SMC):
    """Conditional SMC for CDSSMs"""

    def __init__(self, fk=None, N=100, ESSrmin=0.5, num=10, xstar=None):
        cdssm_smc_kwargs = {**init_kwargs_dict(CDSSM_SMC, locals()), **{'resampling': 'multinomial', 'store_history': True, 'collect': 'off'}}
        CDSSM_SMC.__init__(self, **cdssm_smc_kwargs)
        self.xstar = xstar

class CDSSM_MetropoliswithinGibbs(MetropoliswithinGibbs, CDSSM_MCMC):
    """
    Metropolis within Gibbs for Continuous-Discrete State Space Models. Not to be used alone.
    """
    def __init__(
        self, niter=10, verbose=0, cdssm_cls=None, cdssm_options=None, fk_cls=None, prior=None, data=None, theta0=None, adaptive=True, scale=1.0, rw_cov=None
    ):
        """
        Parameters
        ----------
        niter: int
            number of MCMC iterations
        verbose: int (default=0)
            progress report printed every (niter/verbose) iterations (never if 0)
        cdssm_cls: CDSSM class
            the considered parametric class of state-space models
        cdssm_options: dict
            Additional options for CDSSMs that are not parameters. 
            Possible keys include starting point 'x0' and time step size 'delta_s'.
        fk_cls: CDSSM_FeynmanKac class
            The Feynman-Kac model for the CDSSM.
        prior: StructDist
            the prior
        data: list-like
            the data
        theta0: structured array of size=1 or None
            starting point, simulated from the prior if set to None
        adaptive: True/False
            If true, random walk covariance matrix is adapted recursively
            based on past samples; see also scale and rw_cov for extra info.
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38^2 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array (defaults to Identity matrix if not provided)
            covariance matrix of the random walk proposal if adaptive=False;
            if adaptive=True, rw_cov is used as a preliminary guess for the
            covariance matrix of the target.
        """
        self._check_ssm_and_fk(cdssm_cls, fk_cls)
        mwg_kwargs_dict = init_kwargs_dict(MetropoliswithinGibbs, locals())        
        MetropoliswithinGibbs.__init__(self, **mwg_kwargs_dict)
        self.fk_cls = fk_cls
        self.cdssm_cls = cdssm_cls

    def loglik(self, theta, x):
        theta = {k: theta[k] for k in theta.dtype.names}
        cdssm = self.cdssm_cls(**theta)
        fk_mod = self.fk_cls(cdssm=cdssm, data=self.data)
        loglik = fk_mod.logpt(0, None, x[0]) + cdssm.PY(0, None, x[0]).logpdf(self.data[0][0])
        for t in range(1, len(self.data)):
            loglik += fk_mod.logpt(t, x[t-1], x[t]) + cdssm.PY(t, x[t-1], x[t]).logpdf(self.data[t][0])
        return loglik    
    
class CDSSM_ParticleGibbs(AutoParticleGibbs, CDSSM_MCMC):
    """"""    
    def __init__(
        self,
        niter=10,
        verbose=0,
        cdssm_cls=None,
        cdssm_options=None,
        prior=None,
        data=None,
        theta0=None,
        Nx=100,
        fk_cls=BootstrapReparameterisedDA_DH,
        num=10,
        regenerate_data=False,
        backward_step=False,
        store_x=False,
        adaptive=True,
        scale=1.0,
        rw_cov=None,
        N_steps=1
        ):  
            self._check_ssm_and_fk(cdssm_cls, fk_cls)
            local_vars = locals()
            for k in ["cdssm_cls", "prior", "data", "theta0", "niter", "store_x", "verbose", "N_steps", "Nx", "fk_cls", "regenerate_data", "backward_step", "num", "cdssm_options"]:
                setattr(self, k, local_vars[k])
            self.theta0 = self.prior.rvs(size=1) if theta0 is None else theta0 # Assign theta0 early so it can be passed to MWG
            mwg_kwargs = {**init_kwargs_dict(CDSSM_MetropoliswithinGibbs, local_vars), **{'theta0': self.theta0, 'niter': N_steps*niter}}
            self.mwgibbs = CDSSM_MetropoliswithinGibbs(**mwg_kwargs) # Do this a few times to make the code look nicer
            self.n_mwg = 0
            self.delta_s = cdssm_options['delta_s'] if (cdssm_options and 'delta_s' in cdssm_options) else 1.
            self.build_chain_container()

    def build_chain_container(self):
        theta = np.empty(shape=self.niter, dtype=self.prior.dtype)
        if self.store_x:
            # Remember: when changing this code, 'state_container' 
            # the keyword argument 'dimX'
            x = self.cdssm_cls.state_container(self.niter, len(self.data), self.num, self.delta_s)
            self.chain = ssp.ThetaParticles(theta=theta, x=x)
        else:
            self.chain = ssp.ThetaParticles(theta=theta)

    def fk_mod(self, theta):
        cdssm = self._build_cdssm(self.cdssm_options, theta)
        return self.fk_cls(cdssm=cdssm, data=self.data)

    def update_states(self, theta, x):
        fk = self.fk_mod(theta)
        if x is None:
            cpf = CDSSM_SMC(fk=fk, N=self.Nx, store_history=True, num=self.num)
        else:
            cpf = CDSSM_CSMC(fk=fk, N=self.Nx, xstar=x, num=self.num)
        cpf.run()
        if self.backward_step:
            new_x = cpf.hist.backward_sampling_ON2(1)
        else:
            new_x = cpf.hist.extract_one_trajectory()
        if self.regenerate_data:
            self.data = fk.ssm.simulate_given_x(new_x)
        return new_x

    def samples_transform_W_to_X(self):
        if hasattr(self, 'transformed'):
            raise ValueError('The samples have already been transformed.')
        for i in range(self.niter):
            theta = self.chain.theta[i]
            fk = self.fk_mod(theta)
            trans_x = fk.sample_transform_W_to_X(self.chain.x[i])
            self.chain.x[i] = trans_x
        self.transformed = True
        
# Below are particular classes that run algorithms on LGSSMs.
class SingleSiteLGGibbs(GenericGibbs):
    
    def update_states(self, theta, x):
        """
        Input: theta: structured array containing single theta.
                x: (T, ) numpy arrray containing states.
        Output: x: (T, ) numpy array containing states.
        """
        T = len(self.data)
        if x is None:
            # Initialise x by simulating from the model given initial parameter
            ssm = self.ssm_cls(**ssp.rec_to_dict(theta))
            x_list, _ = ssm.simulate(T)
            x_new = np.array([x_t[0] for x_t in x_list])
            return x_new
        x_new = x.copy()
        x_new[0] = self.single_state_cond_dist(theta, 0., x[1], self.data[0][0]).rvs()
        for t in range(1, T-1):
            x_new[t] = self.single_state_cond_dist(theta, x[t-1], x[t+1], self.data[t][0]).rvs()
        x_new[-1] = self.final_state_cond_dist(theta, x[-2], self.data[-1][0]).rvs()
        return x_new

    def single_state_cond_dist(self, theta, xp, xf, yt):
        sigmaY_2 = 0.01 ** 2
        # A  = (1 + theta['rho'] ** 2)/theta['sigmaX_2'] + 1./theta['sigmaY_2']
        # B = (theta['rho']*(xp + xf))/theta['sigmaX_2'] + yt/theta['sigmaY_2'] 
        A  = (1 + theta['rho'] ** 2)/theta['sigmaX_2'] + 1./sigmaY_2
        B = (theta['rho']*(xp + xf))/theta['sigmaX_2'] + yt/sigmaY_2 
        
        loc = B/A; scale=np.sqrt(1/A)
        return dists.Normal(loc=loc, scale=scale)
    
    def final_state_cond_dist(self, theta, xp, yt):
        sigmaY_2 = 0.01 ** 2
        # A  = 1./theta['sigmaX_2'] + 1./theta['sigmaY_2']
        # B = (theta['rho']*(xp))/theta['sigmaX_2'] + yt/theta['sigmaY_2'] 
        A  = 1./theta['sigmaX_2'] + 1./sigmaY_2
        B = (theta['rho']*(xp))/theta['sigmaX_2'] + yt/sigmaY_2 

        loc = B/A; scale=np.sqrt(1/A)
        return dists.Normal(loc=loc, scale=scale)
    
    def update_theta(self, theta, x):
        """
        Input: theta: structured array containing single theta.
                x: (T, ) numpy arrray containing states.
        Output: new_theta: (T, ) structured array containing single theta.
        """
        posterior_dist = self.ssm_cls.posterior(x, self.data, **self.prior.hyperparams)        
        new_theta = posterior_dist.rvs()
        return new_theta
    
class SingleSiteAutoLGGibbs(AutoGibbs, SingleSiteLGGibbs):
    
    def update_states(self, theta, x):
        return SingleSiteLGGibbs.update_states(self, theta, x)

class LGPGibbs(ParticleGibbs):
    
    def update_theta(self, theta, x):
        """
        Input: theta: structured array containing single theta.
                x: (T, ) numpy arrray containing states.
        Output: new_theta: list containing single theta.
        """
        posterior_dist = self.ssm_cls.posterior(np.array(x), self.data, **self.prior.hyperparams)        
        new_theta = posterior_dist.rvs() 
        return new_theta