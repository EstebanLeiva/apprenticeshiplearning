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
    
    def get_index_from_sa(self, s, a):
        return s + len(self.S) * a

    def get_sa_from_index(self, index):
        return index % len(self.S), int(index / len(self.S))

    def build_T(self):
        T = self.T
        dim = T.shape
        for s in range(dim[0]):
            for index in range(dim[1]):
                t, a = self.get_sa_from_index(index)
                T[s, index] = dirac_delta(s, t) - self.gamma * self.P[s, t, a]
        self.T = T
    
    def get_policy_from_mu(self, mu):
        policy = np.zeros((len(self.S), len(self.A)))
        for s in range(len(self.S)):
            for a in range(len(self.A)):
                mu_s_a = mu[s + len(self.S) * a]
                denominator = 0
                for a_prime in range(len(self.A)):
                    denominator += mu[s + len(self.S) * a_prime]
                policy[s, a] = mu_s_a / denominator
        return policy

class MdpIRL(Mdp):
        
    def __init__(self, S, A, P, v, gamma, mu_e):
        self.S = S
        self.A = A
        self.P = P
        self.v = v
        self.gamma = gamma
        self.T = np.zeros((len(S), len(S) * len(A)))
        self.mu_e = mu_e