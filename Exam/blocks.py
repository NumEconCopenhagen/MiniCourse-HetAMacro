import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def production(par,ini,ss,Gamma,pi_w,N,w,pi,Y):

    Gamma_lag = lag(ini.Gamma,Gamma)

    w[:] = Gamma
    Y[:] = Gamma*N
    pi[:] = (1+pi_w)/(Gamma/Gamma_lag)-1

@nb.njit
def central_bank(par,ini,ss,pi,i,r):

    # a. rule
    if par.mon_rule == 'taylor':

        i[:] = ss.r + par.phi_pi*pi

    elif par.mon_rule == 'fixed_rr':

        pi_plus = lead(pi, ss.pi)
        i[:] = (1+ss.r)*(1+pi_plus)-1
        
    # b. Fisher
    r[:] = (1+i)/(1+pi_plus)-1
        
@nb.njit
def government(par,ini,ss,G,r,B,T):

    for t in range(par.T):
        
        B_lag = B[t-1] if t > 0 else ini.B
        r_lag = r[t-1] if t > 0 else ini.r

        B[t] = ss.B + par.phi_B*(B_lag - ss.B + G[t] - ss.G)
        T[t] = (1+r_lag)*B_lag + G[t] - B[t]

@nb.njit
def hh_pre(par,ini,ss,w,N,T,r,Z,ra,Delta,delta):

    # a. income factor
    Z[:] = w*N - T

    # b. real interest rate
    ra[0] = ss.r
    ra[1:] = r[:-1]

    # c. unemployed
    Delta[:] = (Z/ss.Z)**(par.gamma-1)
    delta[:] = (par.phi_obar-(Z/ss.Z)**(1-par.gamma))/(par.phi_obar-par.phi_ubar)

@nb.njit
def NKWC(par,ini,ss,pi_w,Z,N,AUX_hh,NKWC_res):

    pi_w_plus = lead(pi_w,ss.pi_w)

    C_ast = AUX_hh**(-1/par.sigma)

    num = par.varphi*N**par.nu
    denom = C_ast**(-par.sigma)*(1-par.theta)*Z/N

    LHS = pi_w*(1+pi_w)
    RHS = par.kappa*(num/denom-1)+par.mu_beta*pi_w_plus*(1+pi_w_plus)
    
    NKWC_res[:] = LHS - RHS

@nb.njit
def market_clearing(par,ini,ss,G,B,Y,C_hh,A_hh,clearing_A,clearing_Y):
        
    # b. market clearing
    clearing_A[:] = B-A_hh
    clearing_Y[:] = Y-C_hh-G