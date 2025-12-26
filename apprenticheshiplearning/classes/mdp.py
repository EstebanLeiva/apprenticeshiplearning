import numpy as np
import gurobipy as gp
from gurobipy import GRB

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
        self.T = (1/(1-self.gamma)) * T
    
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
    
    def project_to_policy_space(self, mu_epsilon):
        model = gp.Model("qp_solver")
        model.setParam("OutputFlag", 0)

        mu_proj = model.addMVar(shape=len(mu_epsilon), lb=0.0, name="mu")

        diff = mu_proj - mu_epsilon
        model.setObjective(diff @ diff, GRB.MINIMIZE)

        model.addConstr(self.T @ mu_proj == self.v, name="linear_system")
        model.addConstr(mu_proj.sum() == 1, name="probability_simplex")
        model.optimize()

        if model.status == GRB.OPTIMAL:
            mu_solution = mu_proj.X
        else:
            print("Optimization failed. Status:", model.status)
            return None
        return mu_solution
    
    def policy_evaluation(self, mu_epsilon, c):
        pi = self.get_policy_from_mu(mu_epsilon)
        
        n_states = len(self.S)
        n_actions = len(self.A)
        
        P_pi = np.zeros((n_states, n_states))
        c_pi = np.zeros(n_states)
        
        for s_curr in range(n_states):
            for a in range(n_actions):
                prob = pi[s_curr, a]
                P_pi[s_curr, :] += prob * self.P[:, s_curr, a]
                c_index = self.get_index_from_sa(s_curr, a)
                c_pi[s_curr] += prob * c[c_index]

        I = np.eye(n_states)
        matrix_A = I - self.gamma * P_pi
        J = np.linalg.solve(matrix_A, c_pi)
        J = (1 - self.gamma) * J
        total_expected_cost = np.dot(self.v, J)
        
        return total_expected_cost
