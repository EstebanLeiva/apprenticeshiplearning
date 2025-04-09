import cvxpy as cp
import numpy as np
import pyomo.environ as pyo
from abc import ABC, abstractmethod

class Solver:

    def __init__(self, mdp, c_hat=None, mu_e=None):
        self.mdp = mdp
        self.c_hat = c_hat
        self.mu_e = mu_e

        self.mdp.build_T()

    @abstractmethod
    def solve(self):
        pass

class SolverMdp(Solver):
    def solve(self, iters=None):
        mu = cp.Variable(len(self.mdp.S) * len(self.mdp.A))
        objective = cp.Minimize(
                                self.mdp.c @ mu
                                )
        constraints = [mu >= 0,
                       (self.mdp.T @ mu) == self.mdp.v]
        problem = cp.Problem(objective, constraints)
    
        solver_kwargs = {"verbose": True}
        if iters is not None:
            solver_kwargs["max_iter"] = iters

        problem.solve(cp.CLARABEL, **solver_kwargs)
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
    def __init__(self, gridworld, c_hat, alpha, mu_e, c_0, u_0, mu_0, eta_cu, eta_mu, T, gap_step=25):
        super().__init__(gridworld.mdp_forward, c_hat, mu_e)
        self.alpha = alpha
        self.c = c_0
        self.u = u_0
        self.mu = mu_0
        self.eta_cu = eta_cu
        self.eta_mu = eta_mu
        self.T = T
        self.cs = 0
        self.us = 0
        self.mus = 0
        self.gamma = self.mdp.gamma
        self.get_state_action = gridworld.get_state_action

        # Graphing information
        self.g_cs = np.zeros(T)
        self.g_us = np.zeros(T)
        self.g_mus = np.zeros(T)
        
        self.gap_step = gap_step
        self.gaps = np.zeros(T//gap_step)
        self.gaps_index = [i for i in range(0, T, gap_step)]

        self.c_iter_prev = None
        self.u_iter_prev = None
        self.mu_iter_prev = None

        self.c_iter = np.zeros(T)
        self.u_iter = np.zeros(T)
        self.mu_iter = np.zeros(T)
    
    def solve_expected(self, sims, graphics=False, graphics_gap=False):
        cs = 0
        us = 0
        mus = 0

        for sim in range(sims):
            print(f"Simulation {sim+1}/{sims}")
            c = np.random.uniform(-1, 1, len(self.c))
            u = np.random.uniform(-1, 1, len(self.u))
            mu = np.random.rand(len(self.mu))
            mu = mu / np.sum(mu)

            output = self.solve(c, u, mu, graphics, graphics_gap)
            cs += (output[0])
            us += (output[1])
            mus += (output[2])

        if graphics:
            self.g_cs = self.g_cs / sims
            self.g_us = self.g_us / sims
            self.g_mus = self.g_mus / sims

            self.c_iter = self.c_iter / sims
            self.u_iter = self.u_iter / sims
            self.mu_iter = self.mu_iter / sims
        
        if graphics_gap:
            self.gaps = self.gaps / sims

        return cs/sims, us/sims, mus/sims

    def solve(self, c, u, mu, graphics, graphics_gap):
        # Initialize the sum of the vectors
        self.cs = 0
        self.us = 0
        self.mus = 0

        self.c = c
        self.u = u
        self.mu = mu

        for t in range(self.T):
            g_c, g_u, g_mu = self.gradient_estimation(self.c, self.u, self.mu, self.mu_e)

            self.c, self.u, self.mu = self.smd_step(self.c, self.u, self.mu, g_c, g_u, g_mu)
            self.cs += (self.c)
            self.us += (self.u)
            self.mus += (self.mu)

            if graphics:
                self.g_cs[t] += np.linalg.norm(g_c)
                self.g_us[t] += np.linalg.norm(g_u)
                self.g_mus[t] += np.linalg.norm(g_mu)
                
                if t == 0:
                    self.c_iter_prev = self.cs / (t+1)
                    self.u_iter_prev = self.us / (t+1)
                    self.mu_iter_prev = self.mus / (t+1)
                else:
                    self.c_iter[t] += np.linalg.norm(self.c_iter_prev - self.cs / (t+1), ord=1)
                    self.u_iter[t] += np.linalg.norm(self.u_iter_prev - self.us / (t+1), ord=1)
                    self.mu_iter[t] += np.linalg.norm(self.mu_iter_prev - self.mus / (t+1), ord=1)
                    self.c_iter_prev = self.cs / (t+1)
                    self.u_iter_prev = self.us / (t+1)
                    self.mu_iter_prev = self.mus / (t+1)

            if graphics_gap: 
                if t % self.gap_step == 0:
                    if t == 0:
                        div = 1
                    else:
                        div = t
                    maxi = self.solve_max_problem(self.cs/div, self.us/div, len(self.mdp.S), len(self.mdp.A), self.lagrangian_max, self.alpha, self.c_hat, self.mu_e, (1/(1-self.mdp.gamma)) * self.mdp.T)
                    mini = self.solve_min_problem(self.mus/div, len(self.mdp.S), len(self.mdp.A), self.lagrangian_min, self.alpha, self.c_hat, self.mu_e, (1/(1-self.mdp.gamma)) *self.mdp.T)
                    self.gaps[t // self.gap_step] += maxi - mini


        return self.cs/self.T, self.us/self.T, self.mus/self.T
    
    def gradient_estimation(self, c, u, mu, mu_e):
        # (c,u) gradient estimation
        index_mu_t = np.random.choice(len(mu), p=mu)
        s_t, a_t = self.get_state_action(index_mu_t)
        s_prime_t = np.random.choice(len(self.mdp.P[:, s_t, a_t]), p=self.mdp.P[:, s_t, a_t])
        
        index_mu_e = np.random.choice(len(mu_e), p=mu_e)
        s_e, a_e = self.get_state_action(index_mu_e)
        s_prime_e = np.random.choice(len(self.mdp.P[:, s_e, a_e]), p=self.mdp.P[:, s_e, a_e])

        #OLD without oracle
        #g_c = 2 * self.alpha * (c - self.c_hat) + (1 - self.alpha) * mu_e - mu
        index_sa = np.random.choice(len(mu))
        s, a = self.get_state_action(index_sa)
        g_c = np.zeros(len(c))
        g_c[index_sa] = len(c) * 2 * self.alpha * (c[index_sa] - self.c_hat[index_sa])
        g_c[index_mu_t] = -1
        g_c[index_mu_e] = 1
        
        g_u = np.zeros(len(self.mdp.S)) 
        g_u[s_t] = (1 / (1-self.mdp.gamma))
        g_u[s_prime_t] = - (1 / (1-self.mdp.gamma)) * self.mdp.gamma
        g_u[s_e] = - (1 / (1-self.mdp.gamma))
        g_u[s_prime_e] = (1 / (1-self.mdp.gamma)) * self.mdp.gamma

        # mu gradient estimation
        index_sa = np.random.choice(len(mu))
        s, a = self.get_state_action(index_sa)
        s_prime = np.random.choice(len(self.mdp.P[:, s, a]), p=self.mdp.P[:, s, a])
        
        g_mu = np.zeros(len(mu))
        g_mu[index_sa] = len(mu) * ( c[index_sa] - (1 / (1 - self.gamma)) * (u[s] - self.gamma * u[s_prime]) )

        return g_c, g_u, g_mu
    
    def smd_step(self, c, u, mu, g_c, g_u, g_mu):
        c_prime = self.proj_box(c - self.eta_cu * g_c)
        u_prime = self.proj_box(u - self.eta_cu * g_u)
        mu_prime = self.proj_simplex(mu * np.exp(-self.eta_mu * g_mu))
        return c_prime, u_prime, mu_prime
    
    def proj_simplex(self, x):
        return x / np.linalg.norm(x, ord=1)
    
    def proj_box(self, x):
        return np.clip(x, -1, 1) 
    
    def avg_vec(self, vec_lst):
        total_sum = np.sum(vec_lst, axis=0)
        return total_sum / len(vec_lst)
    
    def lagrangian_max(self, model, c_e, u_e, n_s, alpha, c_hat, mu_expert, T):
        Tt = np.transpose(T)
        norm_term = alpha * sum(
            (c_e[s + a * n_s] - c_hat[s + a * n_s])**2 
            for s in model.S for a in model.A
        )

        dot_term = 0.0
        for s in model.S:
            for a in model.A:
                Ttu_sa = sum(Tt[s + a * n_s, s] * u_e[s] for s in model.S)
                dot_term += (mu_expert[s + a * n_s] - model.mu[s,a]) * (c_e[s + a * n_s] - Ttu_sa)

        dot_term *= (1 - alpha)

        return norm_term + dot_term

    def lagrangian_min(self, model, mu_e, n_s, alpha, c_hat, mu_expert, T):
        Tt = np.transpose(T)
        norm_term = alpha * sum(
            (model.c[s, a] - c_hat[s + a * n_s])**2 
            for s in model.S for a in model.A
        )

        dot_term = 0.0
        for s in model.S:
            for a in model.A:
                Ttu_sa = sum(Tt[s + a * n_s, s] * model.u[s] for s in model.S)
                dot_term += (mu_expert[s + a * n_s] - mu_e[s + a * n_s]) * (model.c[s, a] - Ttu_sa)

        dot_term *= (1 - alpha)

        return norm_term + dot_term


    def solve_max_problem(self, c_e, u_e, n_s, n_a, lagrangian, alpha, c_hat, mu_expert, T):
        model = pyo.ConcreteModel()

        model.S = pyo.RangeSet(0, n_s - 1)
        model.A = pyo.RangeSet(0, n_a - 1)

        model.mu = pyo.Var(model.S, model.A, within=pyo.NonNegativeReals)

        def mu_simplex_rule(m):
            return sum(m.mu[s, a] for a in m.A for s in m.S) == 1
        
        model.mu_sum_constraint = pyo.Constraint(rule=mu_simplex_rule)

        def objective(m):
            return lagrangian(m, c_e, u_e, n_s, alpha, c_hat, mu_expert, T)

        model.obj = pyo.Objective(rule=objective, sense=pyo.maximize)

        solver = pyo.SolverFactory('ipopt')
        solver.solve(model, tee=False)
        opt_val = pyo.value(model.obj)

        return opt_val

    def solve_min_problem(self, mu_e, n_s, n_a, lagrangian, alpha, c_hat, mu_expert, T):
        model = pyo.ConcreteModel()
        
        model.S = pyo.RangeSet(0, n_s - 1)
        model.A = pyo.RangeSet(0, n_a - 1)
        
        model.c = pyo.Var(model.S, model.A, within=pyo.Reals)
        model.u = pyo.Var(model.S, within=pyo.Reals)
        
        def c_box_rule_leq(m, s, a):
            return m.c[s, a] <= 1
        def c_box_rule_geq(m, s, a):
            return m.c[s, a] >= -1
        model.c_box_lower = pyo.Constraint(model.S, model.A, rule=c_box_rule_leq)
        model.c_box_upper = pyo.Constraint(model.S, model.A, rule=c_box_rule_geq)
        
        def u_box_rule_leq(m, s):
            return m.u[s] <= 1
        def u_box_rule_geq(m, s):
            return m.u[s] >= -1
        model.u_box_lower = pyo.Constraint(model.S, rule=u_box_rule_leq)
        model.u_box_upper = pyo.Constraint(model.S, rule=u_box_rule_geq)
        
        def objective(m):
            return lagrangian(m, mu_e, n_s, alpha, c_hat, mu_expert, T)
        model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)
        
        solver = pyo.SolverFactory('Ipopt')
        solver.solve(model, tee=False)
        c_opt_value = pyo.value(model.obj)
        
        return c_opt_value
            
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