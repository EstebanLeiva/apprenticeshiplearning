import numpy as np

from apprenticheshiplearning.utils.build import dirac_delta

class Mdp:

    def __init__(self, S, A, P, v, c, gamma):        
        self.S = S
        self.A = A
        self.P = P
        self.v = v
        self.c = c
        self.gamma = gamma
        self.T = np.zeros((len(S), len(S) * len(A)))

    def build_T(self):
        T = self.T
        dim = T.shape
        for i in range(dim[0]):
            for j in range(dim[1]):
                T[i, j] = dirac_delta(i,j) - self.gamma * self.P[i, j % len(self.A), int(j / dim[1])]
        self.T = T

class MdpIRL(Mdp):
        
    def __init__(self, S, A, P, v, gamma, mu_e):
        self.S = S
        self.A = A
        self.P = P
        self.v = v
        self.gamma = gamma
        self.T = np.zeros((len(S), len(S) * len(A)))
        self.mu_e = mu_e