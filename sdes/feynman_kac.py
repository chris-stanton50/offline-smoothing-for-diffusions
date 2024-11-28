"""
Feynman Kac module: 

In this module, we construct the Feynman Kac models for continuous-discrete state space models, which include reparameterisations.

How to interact with the API:

We have implemented the CDSSM_SMC class: 

This is a subclass of SMC. We use this to run particle filters and smoothers for continuous-discrete state space models.
The only difference between this class and the standard SMC class, is that we need to pass a CDSSM_FeynmanKac object to the __init__ method,
and we need to provide the number of imputed points for simulation of SDEs.

We have the following Feynman-Kac formalisms of continuous-discrete state space models: These are instances of CDSSM_FeynmanKac:

- Bootstrap DA
- BootstrapReparameterisedDA - provide an auxiliary_bridge_cls
- ForwardGuidedDA - provide a proposal_sde_cls
- BackwardGuidedDA - provide an end_pt_proposal_sde_cls, auxiliary_bridge_cls
- ForwardGuidedReparameterisedDA - provide a proposal_sde_cls, auxiliary_bridge_cls
- BackwardGuidedReparameterisedDA - provide a end_pt_proposal_sde_cls, auxiliary_bridge_cls # Only option for hypoelliptic case.

We provide to the __init__ method of these classes:

- cdssm: an instance of CDSSM
- data: the observed data

And any proposal/auxiliary bridge classes that are required. Some default choices are specified if 
they are not provided, but the user can specify one manually.

Each of these classes have the class methods:

'auxiliary_bridge_cls_options'
'univ_auxiliary_bridge_cls_options'
'mv_auxiliary_bridge_cls_options'

'proposal_sde_cls_options'
'univ_proposal_sde_cls_options'
'mv_proposal_sde_cls_options'

'end_pt_proposal_sde_cls_options'
'univ_end_pt_proposal_sde_cls_options'
'mv_end_pt_proposal_sde_cls_options'

These methods return dictionaries of the available classes that can be used for the respective roles.

Currently, the default choices are:

- Forward Proposals: LocalLinearOUProp/ MvOUProposal
- end_pt_proposal_sde_cls: LocalLinearOUProp/ MvOUProposal
- auxiliary_bridge_cls: DelyonHuAuxBridge/ MvDelyonHuAuxBridge
"""

import numpy as np

from particles import SMC, FeynmanKac
from particles.utils import multiplexer
from particles.resampling import wmean_and_var
from particles.collectors import default_collector_cls, Moments
 
from sdes.continuous_discrete_ssms import CDSSM
from sdes.sdes import BrownianMotion, MvIndepBrownianMotion, MvSDE
from sdes.auxiliary_bridges import univ_forward_proposals, univ_auxiliary_bridges, mv_forward_proposals, mv_auxiliary_bridges
from sdes.auxiliary_bridges import LocalLinearOUProp, DelyonHuAuxBridge
from sdes.auxiliary_bridges import MvDelyonHuAuxBridge, MvOUProposal
from sdes.collectors import default_add_funcs

class CDSSM_FeynmanKac(FeynmanKac):
    
    num = 10 # Defualt number of steps for the numerical scheme if not passed new value by SMC.

    def __init__(self, cdssm=None, data=None):
        self.cdssm = cdssm
        self.data = data
        self.cdssm.T = self.T
        self.model_sde = self.cdssm.model_sde
        self.dimX = self.cdssm.model_sde.dimX
        self._check_CDSSM_FK(cdssm)
        
    def _check_CDSSM_FK(self, cdssm):
        if not isinstance(cdssm, CDSSM):
            raise ValueError(f'cdssm {cdssm} must be an instance of CDSSM')

    @property
    def t_end(self):
        state_cont = self.cdssm.state_container(1, 1, self.num, self.cdssm.delta_s, dimX=self.dimX)
        return state_cont.dtype.names[-1]

    def upper_bound_trans(self, t):
        """
        Method needs to be defined on the underlying cdssm to use the 
        collector `Paris` and the offline smoothing algorithm 
        `backward_sampling_reject`.
        
        For CDSSMs, logpt is the product of the potential G_t and the transition density of proposal kernel M_t.
        It may not be possible to find an upper bound on the pathspace.
        """
        return self.cdssm.upper_bound_log_pt(t)

    def add_func(self, t, xp, x):
        """
        Method needs to be defined on the underlying cdssm to use the collectors
        'Online_smooth_naive'/'Online_smooth_ON2'/'PaRIS'. 
        """
        return self.cdssm.add_func(t, xp, x)

    @property
    def add_funcs(self):
        return default_add_funcs # Defined in sdes/collectors.py

class CDSSM_SMC(SMC):
    def __init__(
        self,
        fk=None,
        N=100,
        resampling="systematic",
        ESSrmin=0.5,
        store_history=False,
        verbose=False,
        collect=None,
        num=10
    ):
        super().__init__(fk=fk,
                        N=N,
                        qmc=False,
                        resampling=resampling,
                        ESSrmin=ESSrmin,
                        store_history=store_history,
                        verbose=verbose,
                        collect=collect,
                        )
        self.fk.num = num # Pass the number of imputed points to the fk object.
        if not isinstance(fk, CDSSM_FeynmanKac):
            raise ValueError('fk must be an instance of CDSSM_FeynmanKac')

class BootstrapDA(CDSSM_FeynmanKac):
    """
    Basically the same as the standard Bootstrap PF. Only difference is that instead of 
    simulating from the transition density of the SDE, an approximate sample is generated by using a 
    numerical scheme.
    
    Observation density is univariate Gaussian: Y_t |E_t \sim N(e_t, \eta^2).

    Subclass this object and specifiy the 'ModelSDECls' to fully define a valid fk_model.

    Example:
    -----------

    class BootstrapDA_OU(BootstrapDA):
        ModelSDECls = OrnsteinUhlenbeck

    """    
    
    cls_sname = 'BootstrapDA'
    
    @property
    def sname(self):
        return self.cls_sname

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)
    
    def M0(self, N):
        return self.model_sde.simulate(N, self.cdssm.x0, t_start=self.cdssm.s(0), t_end=self.cdssm.s(1), num=self.num)

    def M(self, t, xp):
        return self.model_sde.simulate(xp.shape[0], xp[self.t_end], t_start=self.cdssm.s(t), t_end=self.cdssm.s(t+1), num=self.num)

    def logG(self, t, xp, x):
        return self.cdssm.PY(t, xp, x[self.t_end]).logpdf(self.data[t])
    
    def default_moments(self, W, X):
        """
        Defines the default moments function for the collector 'Moments' 
        (see the particles module 'collectors').

        In the future, we could use the function resampling.wmean_and_var_str_array to store 
        summaries of the entire SDE path. We will need to think about how to implement this a 
        bit, so that the paths at each time step have a common ending point.
        """
        end_pts = X[self.t_end]
        return wmean_and_var(W, end_pts)

class BootstrapReparameterisedDA(BootstrapDA):

    cls_sname = 'BsR'
    
    def __init__(self, cdssm=None, data=None, auxiliary_bridge_cls = None):
        super().__init__(cdssm=cdssm, data=data)
        self.auxiliary_bridge_cls = self.default_auxiliary_bridge_cls if auxiliary_bridge_cls is None else auxiliary_bridge_cls

    @property
    def sname(self):
        name = self.cls_sname
        bridge_ext = self.auxiliary_bridge_cls.sname if self.dimX == 1 else self.auxiliary_bridge_cls.sname[2:]
        return name + '_' + bridge_ext

    @classmethod
    def auxiliary_bridge_cls_options(cls):
        return {**cls.univ_auxiliary_bridge_cls_options(), **cls.mv_auxiliary_bridge_cls_options()}

    @classmethod
    def univ_auxiliary_bridge_cls_options(cls):
        return {cls.__name__: cls for cls in univ_auxiliary_bridges}

    @classmethod
    def mv_auxiliary_bridge_cls_options(cls):
        return {cls.__name__: cls for cls in mv_auxiliary_bridges}

    @property
    def default_auxiliary_bridge_cls(self):
        return MvDelyonHuAuxBridge if self.dimX > 1 else DelyonHuAuxBridge
    
    def M0(self, N):                            
        X = super().M0(N)
        end_points = X[self.t_end]
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(0), self.cdssm.s(1), end_points)
        x_start = np.ones(N)*self.cdssm.x0 if self.dimX == 1 else np.concatenate([self.cdssm.x0]*N)
        return self.auxiliary_bridge.transform_X_to_W(X, x_start)

    def M(self, t, xp):
        X = super().M(t, xp)
        x_start = xp[self.t_end]; end_points = X[self.t_end]
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), end_points)
        return self.auxiliary_bridge.transform_X_to_W(X, x_start)

    def logG(self, t, xp, x):
        """
        If the observation density depends only on the end point e_t, then transforming the paths is not necessary here.
        """
        
        if t == 0:
            N = x[self.t_end].shape[0]
            x_start = np.ones(N)*self.cdssm.x0 if self.dimX == 1 else np.concatenate([self.cdssm.x0]*N)
        else:
            x_start = xp[self.t_end]
        end_points = x[self.t_end]
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), end_points)
        x = self.auxiliary_bridge.transform_W_to_X(x, x_start)
        log_obs_density = super().logG(t, xp, x)
        return log_obs_density
    
    def logpt(self, t, xp, x):
        """
        The role of logpt in the smoothing algorithms. Test whether changing the transform improves performance.
        """
        x_start = self.cdssm.x0 if xp is None else xp[self.t_end]
        N = x_start.shape[0] if type(x_start) is np.ndarray else 1
        x = np.stack([x]*N) if self.dimX > 1 and type(x) is np.void else x
        if type(x_start) is not np.ndarray:
            x_start = np.array([x_start])
        x_end = x[self.t_end] if self.dimX == 1 else x[self.t_end].reshape((N, self.dimX))
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), x_end)
        x = self.auxiliary_bridge.transform_W_to_X(x, x_start)
        bridge_log_likelihood = self.auxiliary_bridge.bridge_log_likelihood(x_start, x)
        return bridge_log_likelihood
    
    def samples_transform_W_to_X(self, X):
        """
        Inputs: X: list of length T of struct_arrays. Each struct_array is of shape (M, ). 
                    represents M samples from the reparameterised pathspace smoothing distribution 
        Returns:    inv_transformed_X: list of length T of struct_arrays. Each struct_array is of shape (M, ).
        """
        inv_transformed_X = [0] * len(X)
        M = X[0][self.t_end].shape[0]
        for t in range(len(X)):
            self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), X[t][self.t_end])
            x_start = np.ones(M)*self.cdssm.x0 if t == 0 else X[t-1][self.t_end]
            inv_transformed_X[t] = self.auxiliary_bridge.transform_W_to_X(X[t], x_start)
        return inv_transformed_X

    def sample_transform_W_to_X(self, X):
        """
        Inputs: X: struct_array of shape (T, )
                    Represents a single sample from the reparameterised pathspace smoothing distribution.
        Returns:    inv_transformed_X: struct_array of shape (T, )
        """
        inv_transformed_X = X.copy(); T = X.shape[0]
        for t in range(T):
            self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), X[t][self.t_end])
            x_start = np.array([self.cdssm.x0]) if t == 0 else np.array([X[t-1][self.t_end]])
            inv_transformed_X[t] = self.auxiliary_bridge.transform_W_to_X(X[t:t+1], x_start)
        return inv_transformed_X

class ForwardGuidedDA(BootstrapDA):
    """
    Feynman-Kac Model for 1-D Guided Forward Proposals.
    To create a Feynman-Kac class, this needs to be subclassed with:

    'ModelSDECls' and 'ProposalSDECls' defined as class attributes.
    Conditional observation distribution is set as additive noise: $Y_t | E_t=e_t \sim N(L(t)*e_t, sigmaY(t)^2)$

    Subclass this object and specifiy the 'ModelSDECls' and 'ProposalSDECls to fully define a valid fk_model.

    Example:
    -----------
    
    class ForwardGuidedDA_OU_Optimal(ForwardGuidedDA):
        ModelSDECls = OrnsteinUhlenbeck
        ProposalSDECls = LocalLinearOUProp

    """
    cls_sname = 'FwG'
    
    def __init__(self, cdssm=None, data=None, proposal_sde_cls=None):
        super().__init__(cdssm=cdssm, data=data)
        self.proposal_sde_cls = self.default_proposal_sde_cls if proposal_sde_cls is None else proposal_sde_cls

    @property
    def sname(self):
        name = self.cls_sname
        fw_prop_ext = self.proposal_sde_cls.sname if self.dimX == 1 else self.proposal_sde_cls.sname[2:]
        return name + '_' + fw_prop_ext
    
    @classmethod
    def proposal_sde_cls_options(cls):
        return {**cls.univ_proposal_sde_cls_options(), **cls.mv_proposal_sde_cls_options()}

    @classmethod
    def univ_proposal_sde_cls_options(cls):
        return {cls.__name__: cls for cls in univ_forward_proposals}    

    @classmethod
    def mv_proposal_sde_cls_options(cls):
        return {cls.__name__: cls for cls in mv_forward_proposals}

    @property
    def default_proposal_sde_cls(self):
        return MvOUProposal if self.dimX > 1 else LocalLinearOUProp

    def M0(self, N):                            
        self.proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(0), self.cdssm.s(1), self.data[0], self.cdssm.LY(0), self.cdssm.sigmaY(0))
        return self.proposal_sde.simulate(N, self.cdssm.x0, num=self.num)
        
    def M(self, t, xp):
        self.proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), self.data[t], self.cdssm.LY(t), self.cdssm.sigmaY(t))
        return self.proposal_sde.simulate(xp.shape[0], xp[self.t_end], num=self.num)

    def logG(self, t, xp, x):
        self.proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), self.data[t], self.cdssm.LY(t), self.cdssm.sigmaY(t))
        # The line above needs to be changed!
        if t == 0:
            N = x[self.t_end].shape[0] 
            x_start = np.ones(N)*self.cdssm.x0 if self.dimX == 1 else np.concatenate([self.cdssm.x0]*N)
        else:
            x_start = xp[self.t_end]
        self.proposal_sde.build_linear_sde(x_start=x_start)
        obs_density_logpdf = self.cdssm.PY(t, xp, x[self.t_end]).logpdf(self.data[t])
        log_girsanov_likelihood = self.proposal_sde.log_girsanov(x_start, x)
        return obs_density_logpdf + log_girsanov_likelihood
    
class BackwardGuidedDA(BootstrapDA):
    """
    Guided Backward Proposals, using as an example, the Delyon-Hu bridge:

    Simulation of the end point:

    Can either use the `optimal_proposal_dist` when the model SDE is a linear SDE, or
    can use the `euler_proposal_dist` when the model SDE is a general SDE.
    """

    cls_sname = 'BwG'
    
    def __init__(self, cdssm=None, data=None, end_pt_proposal_sde_cls=None, auxiliary_bridge_cls=None):
        super().__init__(cdssm=cdssm, data=data)
        self.end_pt_proposal_sde_cls = self.default_end_pt_proposal_sde_cls if end_pt_proposal_sde_cls is None else end_pt_proposal_sde_cls
        self.auxiliary_bridge_cls = self.default_auxiliary_bridge_cls if auxiliary_bridge_cls is None else auxiliary_bridge_cls

    @property
    def sname(self):
        name = self.cls_sname
        bridge_ext = self.auxiliary_bridge_cls.sname if self.dimX == 1 else self.auxiliary_bridge_cls.sname[2:]
        end_pt_prop_ext = self.end_pt_proposal_sde_cls.sname if self.dimX == 1 else self.end_pt_proposal_sde_cls.sname[2:]
        return name + '_' + bridge_ext + '_' + end_pt_prop_ext

    @classmethod
    def end_pt_proposal_sde_cls_options(cls):
        return {**cls.univ_end_pt_proposal_sde_cls_options(), **cls.mv_end_pt_proposal_sde_cls_options()}

    @classmethod
    def univ_end_pt_proposal_sde_cls_options(cls):
        return {cls.__name__: cls for cls in univ_forward_proposals}

    @classmethod
    def mv_end_pt_proposal_sde_cls_options(cls):
        return {cls.__name__: cls for cls in mv_forward_proposals}

    @property
    def default_end_pt_proposal_sde_cls(self):
        return MvOUProposal if self.dimX > 1 else LocalLinearOUProp

    @classmethod
    def auxiliary_bridge_cls_options(cls):
        return {**cls.univ_auxiliary_bridge_cls_options(), **cls.mv_auxiliary_bridge_cls_options()}

    @classmethod
    def univ_auxiliary_bridge_cls_options(cls):
        return {cls.__name__: cls for cls in univ_auxiliary_bridges}

    @classmethod
    def mv_auxiliary_bridge_cls_options(cls):
        return {cls.__name__: cls for cls in mv_auxiliary_bridges}

    @property
    def default_auxiliary_bridge_cls(self):
        return MvDelyonHuAuxBridge if self.dimX > 1 else DelyonHuAuxBridge

    def M0(self, N):
        end_point_proposal_dist = self._end_point_proposal_dist(0., self.cdssm.x0)
        end_points = end_point_proposal_dist.rvs(N)                         
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(0), self.cdssm.s(1), end_points)
        return self.auxiliary_bridge.simulate(N, self.cdssm.x0, num=self.num)
        
    def M(self, t, xp):
        N = xp[self.t_end].shape[0]
        end_point_proposal_dist = self._end_point_proposal_dist(t, xp[self.t_end])
        end_points = end_point_proposal_dist.rvs(N)                         
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), end_points)
        return self.auxiliary_bridge.simulate(N, xp[self.t_end], num=self.num)

    def logG(self, t, xp, x):
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), x[self.t_end])
        if t == 0:
            N = x[self.t_end].shape[0] 
            x_start = np.ones(N)*self.cdssm.x0 if self.dimX == 1 else np.concatenate([self.cdssm.x0]*N)
        else:
            x_start = xp[self.t_end] 
        end_point_proposal_dist = self._end_point_proposal_dist(t, x_start)
        end_pt_prop_logpdf = end_point_proposal_dist.logpdf(x[self.t_end])
        obs_density_logpdf = self.cdssm.PY(t, xp, x[self.t_end]).logpdf(self.data[t])
        bridge_log_likelihood = self.auxiliary_bridge.bridge_log_likelihood(x_start, x)
        return obs_density_logpdf + bridge_log_likelihood - end_pt_prop_logpdf
    
    def _end_point_proposal_dist(self, t, x_start):
        if self.end_pt_proposal_sde_cls:
            end_point_proposal_sde = self.end_pt_proposal_sde_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), self.data[int(t)], self.cdssm.LY(t), self.cdssm.sigmaY(t))
            end_point_proposal_dist = end_point_proposal_sde.end_point_proposal(x_start)
        else:
            end_point_proposal_dist = self.model_sde.euler_proposal_dist(self.cdssm.s(t), self.cdssm.s(t+1), x_start, self.y, self.cdssm.LY(t), self.cdssm.sigmaY(t))
        return end_point_proposal_dist

class ForwardReparameterisedDA(ForwardGuidedDA, BootstrapReparameterisedDA):
    """ Data Augmentation of forward proposals
    
        To use, one need to subclass and specify the following class methods:

        ProposalSDECls
        AuxiliaryBridgeCls

        It will be interesting to consider the impact of different auxiliary bridge constructions on algorithm performance.
    """
    
    cls_sname = 'FwR'
    
    def __init__(self, cdssm=None, data=None, proposal_sde_cls=None, auxiliary_bridge_cls=None):
        ForwardGuidedDA.__init__(self, cdssm=cdssm, data=data, proposal_sde_cls=proposal_sde_cls)
        self.auxiliary_bridge_cls = self.default_auxiliary_bridge_cls if auxiliary_bridge_cls is None else auxiliary_bridge_cls

    @property
    def sname(self):
        name = self.cls_sname
        bridge_ext = self.auxiliary_bridge_cls.sname if self.dimX == 1 else self.auxiliary_bridge_cls.sname[2:]
        fw_prop_ext = self.proposal_sde_cls.sname if self.dimX == 1 else self.proposal_sde_cls.sname[2:]
        return name + '_' + bridge_ext + '_' + fw_prop_ext


    def M0(self, N):
        self.proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(0), self.cdssm.s(1), self.data[0], self.cdssm.LY(0), self.cdssm.sigmaY(0))
        X = self.proposal_sde.simulate(N, self.cdssm.x0, num=self.num)
        end_points = X[self.t_end]
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.proposal_sde, 0., self.cdssm.s(1) - self.cdssm.s(0), end_points)
        x_start = np.ones(N)*self.cdssm.x0 if self.dimX == 1 else np.concatenate([self.cdssm.x0]*N)
        return self.auxiliary_bridge.transform_X_to_W(X, x_start)
        
    def M(self, t, xp):
        x_start = xp[self.t_end]; N = xp[self.t_end].shape[0]
        self.proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), self.data[t], self.cdssm.LY(t), self.cdssm.sigmaY(t))
        X = self.proposal_sde.simulate(N, x_start, num=self.num)
        end_points = X[self.t_end]
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.proposal_sde, 0., self.cdssm.s(t+1) - self.cdssm.s(t), end_points)
        return self.auxiliary_bridge.transform_X_to_W(X, x_start)

    def logG(self, t, xp, x):
        self.proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), self.data[t], self.cdssm.LY(t), self.cdssm.sigmaY(t))
        if t == 0:
            N = x[self.t_end].shape[0]
            x_start = np.ones(N)*self.cdssm.x0 if self.dimX == 1 else np.concatenate([self.cdssm.x0]*N)
        else:
            x_start = xp[self.t_end]
        self.proposal_sde.build_linear_sde(x_start=x_start)
        obs_density_logpdf = self.cdssm.PY(t, xp, x[self.t_end]).logpdf(self.data[t])
        end_points = x[self.t_end]
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.proposal_sde, 0., self.cdssm.s(t+1) - self.cdssm.s(t), end_points)
        x = self.auxiliary_bridge.transform_W_to_X(x, x_start)
        log_girsanov_likelihood = self.proposal_sde.log_girsanov(x_start, x)
        return obs_density_logpdf + log_girsanov_likelihood

    def logpt(self, t, xp, x):
        """
        The role of logpt in the smoothing algorithms. Test whether changing the transform improves performance.
        """
        x_start = self.cdssm.x0 if xp is None else xp[self.t_end]
        N = x_start.shape[0] if type(x_start) is np.ndarray else 1
        x = np.stack([x]*N) if self.dimX > 1 and type(x) is np.void else x
        if type(x_start) is not np.ndarray:
            x_start = np.array([x_start])
        # N = x_start.shape[0] It shouldn't be necessary to modify the x process.
        # x = np.stack([x]*N)
        self.proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(t+1), self.cdssm.s(t), self.data[t], self.cdssm.LY(t), self.cdssm.sigmaY(t))
        self.proposal_sde.build_linear_sde(x_start=x_start)
        x_end = x[self.t_end] if self.dimX == 1 else x[self.t_end].reshape((N, self.dimX))
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.proposal_sde, 0., self.cdssm.s(t+1) - self.cdssm.s(t), x_end)
        x = self.auxiliary_bridge.transform_W_to_X(x, x_start)
        bridge_log_likelihood = self.auxiliary_bridge.bridge_log_likelihood(x_start, x) # Think about whether this calculation works/ M_t(z_t|z_{t-1})
        log_girsanov_likelihood = self.proposal_sde.log_girsanov(x_start, x) # G_t(z_{t-1}, z_t))
        return bridge_log_likelihood + log_girsanov_likelihood

    def samples_transform_W_to_X(self, X):
        """
        Inputs: X: list of length T of struct_arrays. Each struct_array is of shape (M, ). 
                    represents M samples from the reparameterised pathspace smoothing distribution 
        Returns:    inv_transformed_X: list of length T of struct_arrays. Each struct_array is of shape (M, ).
        """
        inv_transformed_X = [0] * len(X)
        M = X[0][self.t_end].shape[0]
        for t in range(len(X)):
            x_start = np.ones(M)*self.cdssm.x0 if t == 0 else X[t-1][self.t_end]            
            proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(t+1), self.cdssm.s(t), self.data[t], self.cdssm.LY(t), self.cdssm.sigmaY(t))
            proposal_sde.build_linear_sde(x_start=x_start)
            self.auxiliary_bridge = self.auxiliary_bridge_cls(self.proposal_sde, 0., self.cdssm.s(t+1) - self.cdssm.s(t), X[t][self.t_end])
            inv_transformed_X[t] = self.auxiliary_bridge.transform_W_to_X(X[t], x_start)
        return inv_transformed_X

    def sample_transform_W_to_X(self, X):
        """
        Inputs: X: struct_array of shape (T, )
                    Represents a single sample from the reparameterised pathspace smoothing distribution.
        Returns:    inv_transformed_X: struct_array of shape (T, )
        """
        inv_transformed_X = X.copy(); T = X.shape[0]
        for t in range(T):
            x_start = np.array([self.cdssm.x0]) if t == 0 else np.array([X[t-1][self.t_end]])
            proposal_sde = self.proposal_sde_cls(self.model_sde, self.cdssm.s(t+1), self.cdssm.s(t), self.data[t], self.cdssm.LY(t), self.cdssm.sigmaY(t))
            proposal_sde.build_linear_sde(x_start=x_start)
            self.auxiliary_bridge = self.auxiliary_bridge_cls(proposal_sde, self.cdssm.s(t), self.cdssm.s(t+1), X[t][self.t_end])
            inv_transformed_X[t] = self.auxiliary_bridge.transform_W_to_X(X[t:t+1], x_start)
        return inv_transformed_X
    
class BackwardReparameterisedDA(BackwardGuidedDA, BootstrapReparameterisedDA):
    """For backward, reparameterised DA, the same auxiliary bridge process is used for the transform as in the proposal"""

    cls_sname = 'BwR'
    
    def __init__(self, cdssm=None, data=None, end_pt_proposal_sde_cls=None, auxiliary_bridge_cls=None):
        super().__init__(cdssm=cdssm, data=data, end_pt_proposal_sde_cls=end_pt_proposal_sde_cls, auxiliary_bridge_cls=auxiliary_bridge_cls)
        self.brownian_motion = MvIndepBrownianMotion(dimX=self.dimX) if self.dimX > 1 else BrownianMotion()

    @property
    def sname(self):
        name = self.cls_sname
        bridge_ext = self.auxiliary_bridge_cls.sname if self.dimX == 1 else self.auxiliary_bridge_cls.sname[2:]
        end_pt_prop_ext = self.end_pt_proposal_sde_cls.sname if self.dimX == 1 else self.end_pt_proposal_sde_cls.sname[2:]
        return name + '_' + bridge_ext + '_' + end_pt_prop_ext

    def M0(self, N):
        end_point_proposal_dist = self._end_point_proposal_dist(0., self.cdssm.x0)
        end_points = end_point_proposal_dist.rvs(N)        
        W = self._simulate_W(N, self.cdssm.s(1) - self.cdssm.s(0))
        W[self.t_end] = end_points
        return W

    def M(self, t, xp):
        N = xp[self.t_end].shape[0]
        end_point_proposal_dist = self._end_point_proposal_dist(t, xp[self.t_end])
        end_points = end_point_proposal_dist.rvs(N)
        W = self._simulate_W(N, self.cdssm.s(t+1) - self.cdssm.s(t))
        W[self.t_end] = end_points                      
        return W

    def logG(self, t, xp, x):
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), x[self.t_end])
        if t == 0:
            N = x[self.t_end].shape[0] 
            x_start = np.ones(N)*self.cdssm.x0 if self.dimX == 1 else np.concatenate([self.cdssm.x0]*N)
        else:
            x_start = xp[self.t_end]
        end_point_proposal_dist = self._end_point_proposal_dist(t, x_start)
        end_pt_prop_logpdf = end_point_proposal_dist.logpdf(x[self.t_end])
        obs_density_logpdf = self.cdssm.PY(t, xp, x[self.t_end]).logpdf(self.data[t])
        # Apply the map from the Weiner measure to the auxiliary bridge:
        x = self.auxiliary_bridge.transform_W_to_X(x, x_start)
        bridge_log_likelihood = self.auxiliary_bridge.bridge_log_likelihood(x_start, x)
        return obs_density_logpdf + bridge_log_likelihood - end_pt_prop_logpdf

    def logpt(self, t, xp, x):
        """
        In ON^2 smoothing function, x is a single path, xp is all the possible previous paths
        In Particle Gibbs, x is a single path, xp is a single path

        xp: struct_array of size N
        x: struct_array of size 1
        """
        x_start = self.cdssm.x0 if xp is None else xp[self.t_end]
        if type(x_start) is not np.ndarray:
            x_start = np.array([x_start])
        N = x_start.shape[0] if type(x_start) is np.ndarray else 1
        x = np.stack([x]*N) if self.dimX > 1 and type(x) is np.void else x
        # N = x_start.shape[0] It shouldn't be necessary to modify the x process.
        # x = np.stack([x]*N)
        x_end = x[self.t_end] if self.dimX == 1 else x[self.t_end].reshape((N, self.dimX))
        self.auxiliary_bridge = self.auxiliary_bridge_cls(self.model_sde, self.cdssm.s(t), self.cdssm.s(t+1), x_end)
        x = self.auxiliary_bridge.transform_W_to_X(x, x_start)
        bridge_log_likelihood = self.auxiliary_bridge.bridge_log_likelihood(x_start, x)
        return bridge_log_likelihood

    def _simulate_W(self, N, t_diff):
        x_start = np.zeros((N, self.dimX)) if self.dimX > 1 else 0.
        W = self.brownian_motion.simulate(N, x_start, 0., t_diff, num=self.num)
        return W
    
FK_FILTERING_CLASSES = [BootstrapDA, ForwardGuidedDA, BackwardGuidedDA]
FK_SMOOTHING_CLASSES = [BootstrapReparameterisedDA, ForwardReparameterisedDA, BackwardReparameterisedDA]
FK_CLASSES = FK_FILTERING_CLASSES + FK_SMOOTHING_CLASSES

def gen_all_fk_models(cdssm, data, smoothing=False):
    """
    Constructs all possible Feynman-kac models, with the constructions built in the package.
    Stores in a dictionary that can be fed as an input to the MultiSMC funciton.
    """
    all_fk_models = {}
    proposal_kwarg_names = ['auxiliary_bridge_cls', 'proposal_sde_cls', 'end_pt_proposal_sde_cls']
    fk_classes = FK_SMOOTHING_CLASSES if smoothing else FK_CLASSES
    # Construct all possible Feynman-kac models with the constructions built in the package:
    univ_or_mv = 'mv_' if isinstance(cdssm.model_sde, MvSDE) else 'univ_'
    for fk_cls in fk_classes:
        fk_models = {}
        options_dicts =  {name: getattr(fk_cls, univ_or_mv + name + '_options')() 
                        for name in proposal_kwarg_names if hasattr(fk_cls, name + '_options')}
        fk_kwargs = {'cdssm': cdssm, 'data': data}
        fk_models = _gen_fk_models_rec(fk_models, options_dicts, fk_cls, fk_cls.cls_sname, fk_kwargs)
        all_fk_models.update(fk_models)
    return all_fk_models
        
def _gen_fk_models_rec(fk_models, options_dicts, curr_fk_cls, curr_fk_name, curr_fk_kwargs):
    if not options_dicts:
        fk_models[curr_fk_name] = curr_fk_cls(**curr_fk_kwargs)
        return fk_models
    name = list(options_dicts.keys())[0]
    options_dict = options_dicts.pop(name)
    for proposal_cls_name, option in options_dict.items():
        curr_fk_kwargs[name] = option
        opt_name = option.sname if curr_fk_kwargs['cdssm'].model_sde.dimX == 1 else option.sname[2:]
        curr_fk_name_ext = curr_fk_name + '_' + opt_name
        fk_models = _gen_fk_models_rec(fk_models, options_dicts, curr_fk_cls, curr_fk_name_ext, curr_fk_kwargs)
    options_dicts[name] = options_dict
    return fk_models

class _picklable_f:

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, **kwargs):
        smc_cls = CDSSM_SMC
        pf = smc_cls(**kwargs)
        pf.run()
        return self.fun(pf)
            
@_picklable_f
def _identity(x):
    return x

def get_col(pf):
    cols = pf.summaries._collectors
    for col in cols:
        if col.__class__ not in default_collector_cls + [Moments]:
            return col
    return None

def print_summary(out_func):
    def dec_out_func(pf):
        name = pf.fk.__class__.__name__ if not isinstance(pf.fk, CDSSM_FeynmanKac) else pf.fk.sname 
        col = get_col(pf)
        collector = f'with {col.summary_name} collector ' if col else ''
        print(f'Running {pf.__class__.__name__} {name} {collector}with {pf.N} particles took {round(pf.cpu_time,ndigits=4)} seconds')
        return out_func(pf)
    return dec_out_func

@print_summary
def summaries(pf):
    return pf.summaries
    
def multiCDSSM_SMC(nruns=10, nprocs=0, out_func=None, collect=None, **args):
    """
    Version of particles.multiSMC that is applicable to Feynman-Kac measures
    that have been generated from CDSSMs.
    Necessary to make this into a separate function, as the CDSSM_SMC class
    has the 'num' key word argument that is not present in the standard SMC class.
    """
    f = _identity if out_func is None else _picklable_f(out_func)
    return multiplexer(
        f=f,
        nruns=nruns,
        nprocs=nprocs,
        seeding=True,
        protected_args={"collect": collect},
        **args
    )