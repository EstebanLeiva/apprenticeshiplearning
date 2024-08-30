import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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

            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black')
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