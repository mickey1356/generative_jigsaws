# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
# first generate seed points inside a unit circle
def create_seed_points(N, R=1):
    phi = np.pi * (3 - np.sqrt(5))
    points = []
    for i in range(N):
        r = R * np.sqrt((i + 0.5) / N)
        theta = i * phi
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

# visualize for various N
N_values = [2, 3, 5, 10]
plt.figure(figsize=(len(N_values) * 3, 3))
for i, N in enumerate(N_values):
    points = create_seed_points(N)
    plt.subplot(1, len(N_values), i + 1)
    plt.scatter(points[:, 0], points[:, 1])
    circle = plt.Circle((0, 0), 1, color='r', fill=False)
    plt.gca().add_artist(circle)
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'N={N}')
plt.show()
# %%
# generate N=4 seed points
points = create_seed_points(3)

# compute the voronoi diagram for the seed points
from scipy.spatial import Voronoi, voronoi_plot_2d
voi = Voronoi(points)


fig = voronoi_plot_2d(voi)
circle = plt.Circle((0, 0), 1, color='r', fill=False)
plt.gca().add_artist(circle)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# %%
