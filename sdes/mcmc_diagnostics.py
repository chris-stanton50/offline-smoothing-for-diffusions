

# def add_diagnostics(mcmc):
#     """
#     To do: Change this function when you have added storage of 'x' to PMMH/IMMH.
    
#     Note: the effective sample size is always w.r.t some test function.
#     In these calculations, the test functions considered are just parameter components,
#     So we are calculating the ESS for calculation of the mean of each parameter component.
    
#     This could be extended to consider other test functions: e.g median/mad/sd etc.
#     This analysis can also be done for multiple chains.
#     """
#     param_names = mcmc.chain.theta.dtype.names
#     th_var = mcmc.prior.rvs(size=1)
#     attrs = ['mcse', 'mean', 'sd', 'ess']
#     # Assign containers for each stat as attributes:
#     if isinstance(mcmc, GenericRWHM):
#         for attr in attrs:
#             setattr(mcmc, attr, ssp.ThetaParticles(theta=th_var.copy()))
#     elif isinstance(mcmc, CDSSM_ParticleGibbs):
#         x = mcmc.cdssm_cls.state_container(1, len(mcmc.data), mcmc.num, mcmc.delta_s)
#     else:
#         x = mcmc.ssm_cls.state_container(1, len(mcmc.data))
#     if isinstance(mcmc, GenericGibbs):
#         for attr in attrs:
#             setattr(mcmc, attr, ssp.ThetaParticles(theta=th_var.copy(), x=x.copy()))
#     # Assign the stats for the parameters:
        
#     if isinstance(mcmc, CDSSM_ParticleGibbs):
#         for t in range(len(mcmc.data)):
#             x_t_chain = mcmc.chain.x[:, t] # (niter,)
#             x_t_chain = mcmc_chain.reshape(-1, 1) # (niter, 1)
#             mcmc.mcse.x[0][t] = np.sqrt(MCMC_variance(x_t_chain, method='init_seq'))
#     elif isinstance(mcmc, GenericGibbs):
#         for t in range(len(mcmc.data)):
#             x_t_chain = mcmc.chain.x[:, t] # (niter,)
#             x_t_chain = mcmc_chain.reshape(-1, 1) # (niter, 1)
#             mcmc.mcse.x[0][t] = np.sqrt(MCMC_variance(mcmc.chain.x[t], method='init_seq'))
#     else:
#         pass




# A method that we were writing for the Particle Gibbs class.

#     def gen_summaries(self, discard_frac=0.1):
#         discard = int(self.niter*discard_frac)
#         param_names = self.chain.theta.dtype.names
#         th_var = self.prior.rvs(size=1)
#         attr_dict = {'mcse': lambda th: np.sqrt(MCMC_variance(th.reshape(-1, 1), method='init_seq')/self.niter), 
#                      'mean': lambda th: th.mean(),
#                      'std': lambda th: th.std(),
#                      'ess': lambda th: self.niter * th.var()/MCMC_variance(th.reshape(-1, 1), method='init_seq')
#                      }
#         # Assign containers for each stat as attributes:
#         for attr, attr_func in attr_dict.items():
#             if isinstance(self, CDSSM_ParticleGibbs):
#                 x = self.cdssm_cls.state_container(1, len(self.data), self.num, self.delta_s)
#                 setattr(self, attr, ssp.ThetaParticles(theta=th_var.copy(), x=x.copy()))
#             elif isinstance(self, GenericGibbs):
#                 x = self.ssm_cls.state_container(1, len(self.data))
#                 setattr(self, attr, ssp.ThetaParticles(theta=th_var.copy(), x=x.copy()))
#             else:
#                 setattr(self, attr, ssp.ThetaParticles(theta=th_var.copy()))
#         for param in param_names:
#             param_chain = self.chain.theta[param][discard:] # (niter,)
#             for attr, attr_func in attr_dict.items():
#                 getattr(self, attr).theta[param] = attr_func(param_chain)
#         if isinstance(self, GenericGibbs):
#             for t in range(len(self.data)):
#                 x_t_chain = self.chain.x[:, t][discard:]
#                 for attr, attr_func in attr_dict.items():
#                     getattr(self, attr).x[:, t] = attr_func(x_t_chain)  # The attr funcs needs to be different for CDSSMs
#         else:
            
# """Here, we are applying the 'attr_func' """
