import cvxpy as cp
import numpy as np

class Solver:

    def __init__(self, mdp, c_estimate, mu_expert):
        self.mdp = mdp
        self.c_estimate = c_estimate
        self.mu_expert = mu_expert

    def solve_mdp_forward(self):
        mu = cp.Variable(len(self.mdp.S) * len(self.mdp.A))
        objective = cp.Minimize(
                                self.mdp.c @ mu
                                )
        constraints = [mu >= 0,
                       (self.mdp.T @ mu) == self.mdp.v]
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, mu
    
    def solve_irl(self):
        c = cp.Variable((len(self.mdp.S) * len(self.mdp.A)))
        u = cp.Variable(len(self.mdp.S))
        
        objective = cp.Minimize(
                                cp.norm1(c - self.c_estimate)
                                )
        
        constraints = [(c + (np.transpose(self.mdp.T) @ u)) >= 0,
                       (c + (np.transpose(self.mdp.T) @ u)) @ self.mu_expert == 0]
        
        
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, c, u

    def solve(self):
        c = cp.Variable((len(self.mdp.S) * len(self.mdp.A)))
        u = cp.Variable(len(self.mdp.S))
        
        objective = cp.Minimize(
                                    cp.norm1(c - self.c_estimate) 
                                    + 
                                    (c + (np.transpose(self.mdp.T) @ u)) @ self.mu_expert
                                )
        constraints = [c 
                       + 
                       (np.transpose(self.mdp.T) @ u)
                       >= 
                       0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL, verbose=True)
        return problem, c, u