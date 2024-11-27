import cvxpy as cp
import numpy as np
from abc import ABC, abstractmethod

class Solver:

    def __init__(self, mdp, c_hat=None, mu_e=None):
        self.mdp = mdp
        self.c_hat = c_hat
        self.mu_e = mu_e

    @abstractmethod
    def solve(self):
        pass

class SolverMdp(Solver):
    def solve(self):
        mu = cp.Variable(len(self.mdp.S) * len(self.mdp.A))
        objective = cp.Minimize(
                                self.mdp.c @ mu
                                )
        constraints = [mu >= 0,
                       (self.mdp.T @ mu) == self.mdp.v]
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, mu
    
    def solve_dual(self):
        u = cp.Variable(len(self.mdp.S))
        objective = cp.Maximize(
                                self.mdp.v @ u
                                )
        constraints = [(self.mdp.c - np.transpose(self.mdp.T) @ u) >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, u
    
class SolverSMD(Solver):
    def __init__(self, mdp, c_hat, mu_e, c_0, u_0, mu_0, step_size, T):
        super().__init__(mdp, c_hat, mu_e)
        self.c = c_0
        self.u = u_0
        self.mu = mu_0
        self.step_size = step_size
        self.T = T
        self.cs = [c_0]
        self.us = [u_0]
        self.mus = [mu_0]

    def solve(self):
        for t in range(self.T):
            g_c, g_u, g_mu = self.gradient_estimation()
            self.c, self.u, self.mu = self.smd_step(self.c, self.u, self.mu, g_c, g_u, g_mu)
            self.cs.append(self.c)
            self.us.append(self.u)
            self.mus.append(self.mu)
        return self.avg_vec(self.cs), self.avg_vec(self.us), self.avg_vec(self.mus)
    
    def gradient_estimation(self, c, u, mu, mu_e):
        index_mu_t = np.random.choice(len(mu), p=mu)

        index_mu_e = np.random.choice(len(mu_e), p=mu_e)


        return None
    
    def smd_step(self, c, u, mu, g_c, g_u, g_mu):
        c_prime = self.proj_box(c - self.step_size * g_c)
        u_prime = self.proj_box(u - self.step_size * g_u)
        mu_prime = self.proj_simplex(mu * np.exp(-self.step_size * g_mu))
        return c_prime, u_prime, mu_prime
    
    def proj_simplex(self, x):
        return x / np.linalg.norm(x, ord=1)
    
    def proj_box(self, x):
        scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  
        return 2 * scaled -1 
    
    def avg_vec(vec_lst):
        total_sum = np.sum(vec_lst, axis=0)
        return total_sum / len(vec_lst)
            
class SolverIRL(Solver):
    def solve(self):
        c = cp.Variable((len(self.mdp.S) * len(self.mdp.A)))
        u = cp.Variable(len(self.mdp.S))
        
        objective = cp.Minimize(
                                cp.norm1(c - self.c_hat)
                                )
        
        constraints = [(c + (np.transpose(self.mdp.T) @ u)) >= 0,
                       (c + (np.transpose(self.mdp.T) @ u)) @ self.mu_e == 0]
        
        
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, c, u
    
class SolverApp(Solver):
    def solve(self):
        c = cp.Variable((len(self.mdp.S) * len(self.mdp.A)))
        u = cp.Variable(len(self.mdp.S))
        
        objective = cp.Minimize(
                                    cp.norm1(c - self.c_hat) 
                                    + 
                                    (c + (np.transpose(self.mdp.T) @ u)) @ self.mu_e
                                )
        constraints = [c 
                       + 
                       (np.transpose(self.mdp.T) @ u)
                       >= 
                       0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, c, u

    def solve_constraint(self, constraint):
        c = cp.Variable((len(self.mdp.S) * len(self.mdp.A)))
        u = cp.Variable(len(self.mdp.S))
        
        objective = cp.Minimize(
                                    cp.norm1(c - self.c_hat) 
                                    + 
                                    (c + (np.transpose(self.mdp.T) @ u)) @ self.mu_e
                                )
        constraints = [c + (np.transpose(self.mdp.T) @ u) >= 0, 
                       cp.norm1(c - self.c_hat)  <= constraint
                       ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, c, u