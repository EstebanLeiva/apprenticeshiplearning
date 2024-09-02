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
                                - self.mdp.v @ u
                                )
        constraints = [(self.mdp.c + np.transpose(self.mdp.T) @ u) >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, u
        
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