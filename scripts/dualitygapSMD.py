# %%
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from apprenticheshiplearning.utils.gridworld import plot_gridworld, plot_cost_function, plot_occupation, sanity_check, plot_policy
from apprenticheshiplearning.problems.gridworld import GridWorld
from apprenticheshiplearning.classes.solver import SolverMdp, SolverSMD

# %% [markdown]
# ### Setting

# %%
# Parameters
n = 6
wind_intensity = 0.2 # between 0 and 1
gamma = 0.7

# %%
grid = np.zeros((n, n))
goal = [                    
        [i for  i in range(0,int(n/1.2))], 
        [n-1]
        ]
obstacles = [
                [i for  i in range(0,int(n/1.2))],
                [int(n/2)]
                
            ]
actions = ["up", "down", "left", "right"]
init_dist = np.ones((n,n))/(n**2)

# %%
plot_gridworld(grid, obstacles, goal)

# %%
def transitions(s1, s , a, goal, n, wind_intensity): #go from s to s1 with action a
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

def cost(s, a, goal, obstacles):
    r = 0
    if s[0] in obstacles[0] and s[1] in obstacles[1]:
        r = 1
    elif s[0] in goal[0] and s[1] in goal[1]:
        r = -1
    return r

# %%
gridworld = GridWorld(grid, obstacles, goal, wind_intensity, actions, init_dist, gamma)
gridworld.get_mdp_forward()

# %% [markdown]
# ### Forward 

# %%
gridworld = GridWorld(grid, obstacles, goal, wind_intensity, actions, init_dist, gamma)
gridworld.get_mdp_forward()
gridworld.mdp_forward.build_T()
solver_expert = SolverMdp(gridworld.mdp_forward)
prob, mu_e = solver_expert.solve()

# %%
policy_e = gridworld.mdp_forward.get_policy_from_mu(mu_e.value)
visualize_policy = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        visualize_policy[i,j] = np.argmax(policy_e[gridworld.grid_to_S[i, j]])

plot_policy(grid, visualize_policy, obstacles, goal)

# %%
mu_expert = mu_e.value

# %%
mu_e.value

# %%
mu_expert = np.maximum(mu_expert, 0)
mu_expert = mu_expert/np.sum(mu_expert)

# %%
mu_expert

# %%
c_hat = np.zeros(n**2 * len(actions))
for i in range(len(goal[0])):
    for j in range(len(goal[1])):
        for a in range(len(actions)):
            if np.random.uniform() < 0.5:
                s = gridworld.grid_to_S[goal[0][i], goal[1][j]]
                c_hat[gridworld.get_indexing(s, a)] = -1

for i in range(len(obstacles[0])):
    for j in range(len(obstacles[1])):
        for a in range(len(actions)):
            if np.random.uniform() < 0.5:
                s = gridworld.grid_to_S[obstacles[0][i], obstacles[1][j]]
                c_hat[gridworld.get_indexing(s, a)] = 1       

def get_step_size(S, A, gamma, alpha, epsilon):
    v_cu = 32 * alpha * S * A + (16 * (1 - gamma)**2 - 8 * alpha * (1 - gamma)**2 + 2 * (1 + gamma**2) * (1 + (1 - alpha)**2)) / (1 - gamma)**2
    v_mu = S * A * (2 + 4*(1 + gamma**2) / (1 - gamma)**2)
    eta_cu = epsilon / (4 * v_cu)
    eta_mu = epsilon / (4 * v_mu)
    return eta_cu, eta_mu

# %%
alpha = 0.9
mu_e = mu_expert
c_0 = np.zeros(n**2 * len(actions))
u_0 = np.zeros(n**2)
mu_0 = np.ones(n**2 * len(actions)) / (n**2 * len(actions))
step_size = get_step_size(n**2, len(actions), gamma, alpha, 0.1)
T = 10000
N = 20

solver_smd = SolverSMD(gridworld, c_hat, alpha, mu_e, c_0, u_0, mu_0, 1e-2, 1e-2, T)
c, u, mu = solver_smd.solve_expected(N, True)

policy_e = gridworld.mdp_forward.get_policy_from_mu(mu)
visualize_policy = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        visualize_policy[i,j] = np.argmax(policy_e[gridworld.grid_to_S[i, j]])

plot_policy(grid, visualize_policy, obstacles, goal)

plt.plot(solver_smd.c_iter)
plt.plot(solver_smd.u_iter)
plt.plot(solver_smd.mu_iter)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend(['c', 'u', 'mu'])
plt.show()

plt.scatter(solver_smd.gaps_index, solver_smd.gaps, alpha=0.7, edgecolor='k')
plt.ylim((0, 50))
plt.xlabel('Iterations')
plt.ylabel('Duality gap')
plt.grid()
plt.show()