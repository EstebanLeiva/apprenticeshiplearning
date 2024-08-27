import numpy as np
import itertools
from apprenticheshiplearning.classes.mdp import MdpIRL

class GridWorld:

    def __init__(self, grid, obstacles, goal, actions, transitions, init_dist, cost, gamma, expert_occupancy_measure, c_estimate):
        self.grid = grid
        self.obstacles = obstacles
        self.goal = goal
        self.actions = actions
        self.transitions = transitions #function
        self.init_dist = init_dist
        self.cost = cost
        self.gamma = gamma
        self.expert_occupancy_measure = expert_occupancy_measure
        self.c_estimate = c_estimate
    
        # Connection with MDP class
        self.grid_to_S = None
        self.S_to_grid = None
        self.P = None
        self.v = None
        self.c = None
        self.mu_e = None
        self.c_hat = None
        self.mdp = None
        

    def get_grid_S(self, grid):
        dim = grid.shape
        grid_to_S = np.zeros((dim[0], dim[1]), dtype=int)
        S_to_grid = np.zeros(dim[0] * dim[1], dtype=tuple)
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                grid_to_S[i, j] = int(k)
                S_to_grid[k] = tuple((i, j))
                k += 1
        return grid_to_S, S_to_grid
    
    def get_P(self, grid_to_S, S_to_grid, transitions, actions, goal):
        P = np.zeros((len(S_to_grid), len(S_to_grid), len(actions)))
        pairs = list(itertools.product(S_to_grid, repeat=2))
        for pair in pairs:
            for k in range(len(actions)):
                P[grid_to_S[pair[0][0], pair[0][1]], grid_to_S[pair[1][0], pair[1][1]], k] = transitions(pair[0], pair[1], actions[k], goal)
        return P
    
    def get_v(self, S_to_grid, init_dist):
        v = np.zeros(len(S_to_grid))
        for i in range(len(v)):
            v[i] = init_dist[S_to_grid[i][0], S_to_grid[i][1]]
        return v
    
    def get_c(self, S_to_grid, actions, cost, goal, obstacles):
        c = np.zeros((len(S_to_grid) * len(actions)))
        for i in range(len(S_to_grid)):
            for k in range(len(actions)):
                c[i + len(S_to_grid) * k] = cost(S_to_grid[i], actions[k], goal, obstacles)
        return c
    
    def get_mu_e(self, S_to_grid, actions, expert_policy):
        mu_e = np.zeros((len(S_to_grid) * len(actions)))
        for i in range(len(S_to_grid)):
            for k in range(len(actions)):
                mu_e[i + len(S_to_grid)*k] = expert_policy[S_to_grid[i][0], S_to_grid[i][1], k]
        return mu_e
    
    def get_c_estimate(self, S_to_grid, c_estimate, actions, goal, obstacles):
        c_hat = np.zeros((len(S_to_grid) * len(actions)))
        for i in range(len(S_to_grid)):
            for k in range(len(actions)):
                c_hat[i + len(S_to_grid) * k] = c_estimate(S_to_grid[i], actions[k], goal, obstacles)
        return c_hat

    def get_mdp(self):
        self.grid_to_S, self.S_to_grid = self.get_grid_S(self.grid)
        self.P = self.get_P(self.grid_to_S, self.S_to_grid, self.transitions, self.actions, self.goal)
        self.v = self.get_v(self.S_to_grid, self.init_dist)
        self.c = self.get_c(self.S_to_grid, self.actions, self.cost, self.goal, self.obstacles)
        self.mu_e = self.get_mu_e(self.S_to_grid, self.actions, self.expert_occupancy_measure)
        self.c_hat = self.get_c_estimate(self.S_to_grid, self.c_estimate, self.actions, self.goal, self.obstacles)
        self.mdp = MdpIRL(self.S_to_grid, self.actions, self.P, self.v, self.gamma, self.mu_e)