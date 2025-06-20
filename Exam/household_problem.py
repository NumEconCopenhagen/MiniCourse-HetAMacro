# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit       
def solve_hh_backwards(par,z_trans,ra,Z,Delta,omega,delta,vbeg_a_plus,vbeg_a,a,c,y,aux,ss=False):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in range(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):
            
            # z
            z = par.e_grid[i_z]*Delta
            if par.u_grid[i_z]:
                z *= par.phi_ubar
            else:
                z *= par.phi_obar

            # m
            y[i_fix,i_z] = Z*z
            m = (1+ra)*par.a_grid + y[i_fix,i_z] + omega

            # c
            if ss :

                a[i_fix,i_z,:] = 0.0

            else:

                # i. EGM
                c_endo = (par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
                m_endo = c_endo + par.a_grid
                
                # ii. interpolation to fixed grid
                interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
                a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
                
            # iii. more outputs
            c[i_fix,i_z] = m-a[i_fix,i_z]
            aux[i_fix,i_z] = c[i_fix,i_z]**(-par.sigma)*z

        # b. update transition matrix
        z_trans[i_fix,:par.Ne,:par.Ne] = (1-delta)*par.e_trans
        z_trans[i_fix,:par.Ne,par.Ne:] = delta*par.e_trans

        z_trans[i_fix,par.Ne:,:par.Ne] = (1-delta)*par.xi*par.e_trans
        z_trans[i_fix,par.Ne:,par.Ne:] = ((1-par.xi)+par.xi*delta)*par.e_trans

        # c. expectation step
        v_a = (1+ra)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a