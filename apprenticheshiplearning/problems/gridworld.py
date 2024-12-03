import numpy as np
import itertools
from apprenticheshiplearning.classes.mdp import Mdp, MdpIRL

class GridWorld:

    def __init__(self, grid, obstacles, goal, wind_intensity, actions, init_dist, gamma, expert_occupancy_measure=None, c_estimate=None):
        self.grid = grid
        self.wind_intensity = wind_intensity
        self.obstacles = obstacles
        self.goal = goal
        self.actions = actions
        self.init_dist = init_dist
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

        # MDPs
        self.mdp_forward = None
        self.mdp_forward_dual = None
        self.mdp_IRL = None
        self.mdp_app = None

    def transitions(self, s1, s , a, goal, n, wind_intensity): #go from s to s1 with action a
        if a == "up":
            if s[0] in goal[0] and s[1] in goal[1]:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            elif s[0] != 0 and s[1] == n - 1:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1 - wind_intensity
                elif s1[0] == s[0] - 1 and s1[1] == s[1]:
                    return wind_intensity
                else:
                    return 0
            elif s[0] == 0 and s[1] == n - 1:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            elif s[0] == 0 and s[1] < n - 1:
                if s1[0] == s[0] and s1[1] == s[1] + 1:
                    return 1
                else:
                    return 0
            else:
                if s1[0] == s[0] and s1[1] == s[1] + 1:
                    return 1 - wind_intensity
                elif s1[0] == s[0] - 1 and s1[1] == s[1] + 1:
                    return wind_intensity
                else:
                    return 0
                
        if a == "down":
            if s[0] in goal[0] and s[1] in goal[1]:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            elif s[0] != 0 and s[1] == 0:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1 - wind_intensity
                elif s1[0] == s[0] - 1 and s1[1] == s[1]:
                    return wind_intensity
                else:
                    return 0
            elif s[0] == 0 and s[1] == 0:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            elif s[0] == 0 and s[1] > 0:
                if s1[0] == s[0] and s1[1] == s[1] - 1:
                    return 1
                else:
                    return 0
            else:
                if s1[0] == s[0] and s1[1] == s[1] - 1:
                    return 1 - wind_intensity
                elif s1[0] == s[0] - 1 and s1[1] == s[1] - 1:
                    return wind_intensity
                else:
                    return 0

        if a == "left":
            if s[0] in goal[0] and s[1] in goal[1]:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            elif s[0] == 0:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            elif s[0] == 1:
                if s1[0] == s[0] - 1 and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            else:
                if s1[0] == s[0] - 1 and s1[1] == s[1]:
                    return 1 - wind_intensity
                elif s1[0] == s[0] - 2 and s1[1] == s[1]:
                    return wind_intensity
                else:
                    return 0

        if a == "right":
            if s[0] in goal[0] and s[1] in goal[1]:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            elif s[0] == n - 1:
                if s1[0] == s[0] and s1[1] == s[1]:
                    return 1
                else:
                    return 0
            else:
                if s1[0] == s[0] + 1 and s1[1] == s[1]:
                    return 1 - wind_intensity
                elif s1[0] == s[0] and s1[1] == s[1]:
                    return wind_intensity
                else:
                    return 0

    def cost(self, s, a, goal, obstacles):
        r = 0
        if s[0] in obstacles[0] and s[1] in obstacles[1]:
            r = 1
        elif s[0] in goal[0] and s[1] in goal[1]:
            r = -1
        return r

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
    
    def get_indexing(self, s, a):
        return s + len(self.S_to_grid) * a
    
    def get_state_action(self, index):
        return index % len(self.S_to_grid), index // len(self.S_to_grid)
    
    def get_P(self, actions, goal, S_to_grid, wind_intensity):
        P = np.zeros((len(S_to_grid), len(S_to_grid), len(actions)))

        for next_state in range(len(S_to_grid)):
            for state in range(len(S_to_grid)):
                for action in range(len(actions)):
                    P[next_state, state, action] = self.transitions(S_to_grid[next_state], S_to_grid[state], actions[action], goal, len(S_to_grid), wind_intensity)
        
        for s in range(len(S_to_grid)):
            for a in range(len(actions)):
                if sum(P[:, s, a]) != 1:
                    P[s, s, a] = 1
                    #grid_s = self.S_to_grid[s]
                    #ac = actions[a]
                    #print("Error in state ", grid_s, " and action ", ac, "------------- Sum of probabilities: ", sum(P[:,s,a]))
                    #print("ERROR: Transition probabilities do not sum to 1")
        return P
    
    def get_v(self, S_to_grid, init_dist):
        v = np.zeros(len(S_to_grid))
        for i in range(len(v)):
            v[i] = init_dist[S_to_grid[i][0], S_to_grid[i][1]]
        return v
    
    def get_c(self, S_to_grid, actions, goal, obstacles):
        c = np.zeros((len(S_to_grid) * len(actions)))
        for s in range(len(S_to_grid)):
            for a in range(len(actions)):
                c[s + len(S_to_grid) * a] = self.cost(S_to_grid[s], actions[a], goal, obstacles)
        return c
    
    def get_mu_e(self, S_to_grid, actions, expert_occupancy_measure, transform=True):
        if transform:
            mu_e = np.zeros((len(S_to_grid) * len(actions)))
            for s in range(len(S_to_grid)):
                for a in range(len(actions)):
                    mu_e[s + len(S_to_grid) * a] = expert_occupancy_measure[S_to_grid[s][0], S_to_grid[s][1], a]
            return mu_e
        else:
            return expert_occupancy_measure
    
    def get_c_estimate(self, S_to_grid, c_estimate, actions, goal, obstacles):
        c_hat = np.zeros((len(S_to_grid) * len(actions)))
        for s in range(len(S_to_grid)):
            for a in range(len(actions)):
                c_hat[s + len(S_to_grid) * a] = c_estimate(S_to_grid[s], actions[a], goal, obstacles)
        return c_hat

    def get_mdp_forward(self):
        self.grid_to_S, self.S_to_grid = self.get_grid_S(self.grid)
        self.P = self.get_P(self.actions, self.goal, self.S_to_grid, self.wind_intensity)
        self.v = self.get_v(self.S_to_grid, self.init_dist)
        self.c = self.get_c(self.S_to_grid, self.actions, self.goal, self.obstacles)
        self.mdp_forward = Mdp(self.S_to_grid, self.actions, self.P, self.v, self.c, self.gamma)