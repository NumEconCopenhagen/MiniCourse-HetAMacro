
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state

class HANKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['ra','Z','Delta','omega'] # direct inputs
        self.inputs_hh_z = ['delta'] # transition matrix inputs
        self.outputs_hh = ['a','c','y','aux'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['G','Gamma'] # exogenous inputs
        self.unknowns = ['pi_w','N','Delta'] # endogenous inputs
        self.targets = ['NKWC_res','clearing_A','clearing_Y'] # targets
        
        # d. all variables
        self.blocks = [
            'blocks.production',
            'blocks.central_bank',
            'blocks.government',
            'blocks.hh_pre',
            'hh',
            'blocks.NKWC',
            'blocks.market_clearing'
        ]        

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        
    def setup(self):
        """ set baseline parameters """

        par = self.par
        par.Nfix = 1 # number of fixed types

        # a. preferences
        par.mu_beta = 0.90 # mean for discount factor, range is [mu-sigma,mu+sigma]
        par.sigma_beta = 0.0 # spread for discount factor, range is [mu-sigma,mu+sigma]

        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution
        par.nu = 1.0 # inverse Frisch elasticity
        par.varphi = np.nan # disutility of labor (determined in ss)

        # b. income parameters

        # e
        par.rho_e = 0.95 # AR(1) parameter for e
        par.sigma_psi = 0.10 # std. of psi
        par.Ne = 11 # number of productivity states
        
        # u
        par.phi_obar = 1.1 # high value
        par.phi_ubar = 0.5 # low value
        par.gamma = 0.80 # correlation with aggregate post-tax income, = 1 is neutrality 
        par.xi = 1.0 # persistence of u, = 1 is no persistence

        # c. price setting
        par.kappa = 0.05 # slope of wage Phillips curve

        # d. central bank
        par.mon_rule = 'taylor' # monetary policy rule
        par.r_target_ss = 0.05 # real interest rate
        par.phi_pi = 1.5 # Taylor rule coefficient on inflation

        # e. government
        par.BY_ratio = 0.30 # government debt to output ratio
        par.GY_ratio = 0.20 # government spending to output ratio
        par.phi_B = 0.75 # elasticity in debt feedback rule

        # f. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. shocks
        par.jump_G = 1.0/100 # jump in G
        par.rho_G = 0.80 # AR(1) parameter for G

        # h. misc.
        par.T = 300 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
        par.py_blocks = False # use python in blocks
        par.py_hh = False # use python in household problem

    def allocate(self):
        """ allocate model """

        par = self.par

        # a. beta
        par.beta_grid = np.zeros(par.Nfix)

        # b. e
        par.Nz = par.Ne*2
        par.e_grid = np.zeros(par.Nz)
        par.e_scale = np.nan
        par.e_trans = np.zeros((par.Nz//2,par.Nz//2))
        
        # c. u
        par.u_grid = np.zeros(par.Nz,dtype=bool)
        par.u_grid[:par.Ne] = False
        par.u_grid[par.Ne:] = True

        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss 

    def fiscal_multiplier(self):
        """ calculate fiscal multiplier """

        par = self.par
        ss = self.ss
        path = self.path

        dY_cum = np.sum([(1+ss.r)**(-t)*(path.Y[t,0]-ss.Y) for t in range(par.T)])
        dT_cum = np.sum([(1+ss.r)**(-t)*(path.T[t,0]-ss.T) for t in range(par.T)])
        
        return dY_cum/dT_cum