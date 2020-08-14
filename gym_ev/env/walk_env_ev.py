import sys
import numpy as np
from six import StringIO
from string import ascii_uppercase

from gym import utils
from gym.envs.toy_text import discrete
from scipy.stats import norm


class WalkEnv_ev(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, n_states=500, nQ=100, nR=5):
        ## Begin Markov
        m = 3
        sigma_u  = 0.08 / np.sqrt(12)
        rho  = 0.80
        F = norm(loc=0, scale=sigma_u).cdf
        # standard deviation of y_t
        std_y = np.sqrt(sigma_u**2 / (1-rho**2))
        # top of discrete state space
        x_max = m * std_y
        # bottom of discrete state space
        x_min = - x_max
        # discretized state space
        x = np.linspace(x_min, x_max, nR)
        step = (x_max - x_min) / (nR - 1)
        half_step = 0.5 * step
        Rp = np.empty((nR, nR))
        
        for i in range(nR):
            Rp[i, 0] = F(x[0]-rho * x[i] + half_step)
            Rp[i, nR-1] = 1 - F(x[nR-1] - rho * x[i] - half_step)
            for j in range(1, nR-1):
                z = x[j] - rho * x[i]
                Rp[i, j] = F(z + half_step) - F(z - half_step)
                
        Rg = x
        ## End Markov
    
        beta=0.97
        aap=0.40
        delta=0.1
        a2=1.0
        kss=(aap*np.exp(0.5*(sigma_u**2))/(beta**(-1)-(1-delta)))**(1/(1-aap))
        Qg = np.linspace(kss*0.5,kss*1.5,nQ)
        
        Qg = Qg.reshape((nQ,1), order='F')
        Rg = Rg.reshape((nR,1), order='F')

        Rt = np.kron(Rg,np.ones((nQ,1)))
        Qt = np.kron(np.ones((nR,1)),Qg)
        
        self.shape = (1, n_states + 0)
        self.start_state_index = self.shape[1]//2

        self.nS = nS = np.prod(self.shape)
        self.nA = nA = nQ

        P = {}
        for s in range(nS):
            P[s] = {}
            iq   = s % nQ
            ir   = int(((s-iq)/nQ) % nR)
            
            for a in range(nA):
                P[s][a] = []
                
                for irp in range(nR):
                    p_forward = Rp[ir,irp]
                    s_forward = irp * nQ + a
                    tmp       = Qg[a,0] - (1-delta)*Qt[s,0]
                    r_forward = (Qt[s,0]**aap)*np.exp(Rt[s,0]) - tmp - 0.5*a2*Qt[s,0]*((tmp/Qt[s,0]-delta)**2)
                    d_forward = 0
                    P[s][a].append((p_forward, s_forward, r_forward, d_forward))
    
    
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = np.asarray(['[' + ascii_uppercase[:self.shape[1] - 2] + ']'], dtype='c').tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        color = 'red' if self.s == 0 else 'green' if self.s == self.nS - 1 else 'yellow'
        desc[0][self.s] = utils.colorize(desc[0][self.s], color, highlight=True)
        outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
