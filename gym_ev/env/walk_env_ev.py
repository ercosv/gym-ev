import sys
import numpy as np
from six import StringIO
from string import ascii_uppercase

from gym import utils
from gym.envs.toy_text import discrete


class WalkEnv_ev(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, n_states = 100*3, nQ=100, nR=3, p11=0.8, p12=0.15, p13=0.05, p21=0.15, p22=0.70, p23=0.15, p31=0.05, p32=0.15, p33=0.80):

        # two terminal states added
        Qg = np.linspace(2,7,nQ)
        Qg = Qg.reshape((nQ,1))
        Rg = np.linspace(-0.07,0.07,nR)
        Rg = Rg.reshape((nR,1))
        Rp = np.zeros((nR,nR))
        Rp[0,0] = p11
        Rp[0,1] = p12
        Rp[0,2] = p13
        Rp[1,0] = p21
        Rp[1,1] = p22
        Rp[1,2] = p23
        Rp[2,0] = p31
        Rp[2,1] = p32
        Rp[2,2] = p33    

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
                    tmp       = Qg[a,0] - (1-0.1)*Qt[s,0]
                    r_forward = (Qt[s,0]**0.3) * np.exp(Rt[s,0]) - tmp - 0.1*Qt[s,0]*((tmp/Qt[s,0] - 0.1)**2)
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
