import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from apprenticheshiplearning.classes.solver import SolverMdp, SolverIRL, SolverApp

def plot_gridworld(grid, obstacles, goal):
    rows, cols = grid.shape
    fig, ax = plt.subplots()
    
    for i in range(rows):
        for j in range(cols):
            if i in obstacles[0] and j in obstacles[1]:
                color = 'red'
            elif i in goal[0] and j in goal[1]:
                color = 'green'
            else:
                color = 'white'

            rect = plt.Rectangle((i, j), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')

    plt.gca()
    plt.show()

def plot_occupation(grid):
    cmap = mcolors.ListedColormap(['green'])
    norm = mcolors.Normalize(vmin=0, vmax=np.max(np.abs(grid)))
    alpha = np.clip(np.abs(grid) / np.max(np.abs(grid)), 0.1, 1)
    plt.matshow(grid, cmap=cmap, norm=norm, alpha=alpha)
    plt.gca().invert_yaxis()
    plt.show()

def plot_cost_function(grid):
    cmap = mcolors.ListedColormap(['green', 'red'])
    norm = mcolors.BoundaryNorm(boundaries=[-np.max(np.abs(grid)), 0, np.max(np.abs(grid))], ncolors=2)
    alpha = np.clip(np.abs(grid) / np.max(np.abs(grid)), 0.1, 1)
    plt.matshow(grid, cmap=cmap, norm=norm, alpha=alpha)
    plt.gca().invert_yaxis()
    plt.show()

def plot_policy(grid, policy, obstacles, goal):
    rows, cols = grid.shape
    fig, ax = plt.subplots()
    
    for i in range(rows):
        for j in range(cols):
            if i in obstacles[0] and j in obstacles[1]:
                color = 'red'
            elif i in goal[0] and j in goal[1]:
                color = 'green'
            else:
                color = 'white'

            rect = plt.Rectangle((i, j), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            if policy[i, j] == 0:
                plt.arrow(i + 0.5, j + 0.5, 0, 0.2, head_width=0.2, head_length=0.2, fc='k', ec='k')
            elif policy[i, j] == 1:
                plt.arrow(i + 0.5, j + 0.5, 0, -0.2, head_width=0.2, head_length=0.2, fc='k', ec='k')
            elif policy[i, j] == 2:
                plt.arrow(i + 0.5, j + 0.5, -0.2, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
            elif policy[i, j] == 3:
                plt.arrow(i + 0.5, j + 0.5, 0.2, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')

    plt.gca()
    plt.show()

def sanity_check(gridworld, c_estimate):
    # Forward problem
    gridworld.get_mdp_forward()
    gridworld.mdp_forward.build_T()

    solverMdp = SolverMdp(gridworld.mdp_forward)
    problem, mu_e = solverMdp.solve()
    mu_e = mu_e.value
    
    policy_e = gridworld.mdp_forward.get_policy_from_mu(mu_e)
    problem, u_e = solverMdp.solve_dual()
    u_e = u_e.value

    # IRL problem
    gridworld.expert_occupancy_measure = mu_e
    gridworld.c_estimate = c_estimate
    gridworld.get_mdp_IRL(transform=False)
    gridworld.mdp_IRL.build_T()

    solverIRL = SolverIRL(gridworld.mdp_IRL, gridworld.c_hat, gridworld.mu_e)
    problem, c_IRL, u_IRL = solverIRL.solve()
    c_IRL = c_IRL.value
    u_IRL = u_IRL.value

    print(f'||c_IRL - c_true|| = {np.linalg.norm(c_IRL - gridworld.mdp_forward.c)}')
    print(f'||u_IRL - u_true|| = {np.linalg.norm(u_IRL - u_e)}')

    return u_e