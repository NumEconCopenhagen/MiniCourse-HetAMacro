# find steady state

import time
import numpy as np
from scipy import optimize

from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # a. beta
    par.beta_grid[:] = np.linspace(par.mu_beta-par.sigma_beta,par.mu_beta+par.sigma_beta,par.Nfix)

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,ss.w*par.a_max,par.Na)

    # c. z
    e_grid,par.e_trans,e_ergodic,_,_ = log_rouwenhorst(par.rho_e,par.sigma_psi,n=par.Ne)

    par.e_scale = (e_grid**(1-par.theta))@e_ergodic
    par.e_grid[:] = np.tile(e_grid,2)

    z_trans = np.zeros((par.Nz,par.Nz))
    z_trans[:par.Ne,:par.Ne] = (1-ss.delta)*par.e_trans
    z_trans[:par.Ne,par.Ne:] = ss.delta*par.e_trans

    z_trans[par.Ne:,:par.Ne] = (1-ss.delta)*par.e_trans
    z_trans[par.Ne:,par.Ne:] = ss.delta*par.e_trans

    ss.z_trans[:,:,:] = z_trans

    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:par.Ne,0] = e_ergodic*(1-ss.delta)/par.Nfix
        ss.Dbeg[i_fix,par.Ne:,0] = e_ergodic*ss.delta/par.Nfix
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    model.set_hh_initial_guess() # calls .solve_hh_backwards() with ss=True
        
def find_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.Gamma = 1.0
    ss.N = 1.0
    ss.pi = ss.pi_w = 0.0
    
    # b. monetary policy
    ss.i = ss.r = par.r_target_ss

    # c. firms
    ss.w = ss.Gamma
    ss.Y = ss.Gamma*ss.N

    # d. government
    ss.B = par.BY_ratio
    ss.G = par.GY_ratio
    ss.T = ss.r*ss.B+ss.G

    # e. household
    ss.ra = ss.r
    ss.Z = ss.w*ss.N - ss.T
    ss.Delta = 1.0
    ss.delta = (par.phi_obar-1)/(par.phi_obar-par.phi_ubar)

    def obj(x):
        
        par.mu_beta = x[0]
        
        model.solve_hh_ss(do_print=False)
        model.simulate_hh_ss(do_print=False)
        
        return np.array([ss.A_hh-ss.B])

    x_guess = np.array([par.mu_beta])
    res = optimize.root(obj,x_guess,method='hybr')
    obj(res.x)

    # f. market clearing
    ss.clearing_A = ss.B-ss.A_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.G

    # g. NK wage curve
    C_ast = ss.AUX_hh**(-1/par.sigma)
    num_novarphi = ss.N**par.nu
    denom = C_ast**(-par.sigma)*(1-par.theta)*ss.Z/ss.N
    par.varphi = denom/num_novarphi
    ss.NKWC_res = 0.0