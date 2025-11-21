import numpy as np
import matplotlib.pyplot as plt

def imls(query_points, control_points, control_normals, r=0.1):
    # query_points: (Q, 2)
    # control_points/normals: (C, 2)
    # broadcast so x_pts[i] = query_points[i] - control_points (Q, C, 2)
    x_pts = query_points[:, None, :] - control_points[None, ...]
    # normalize each vector (Q, C)
    x_dists = np.linalg.norm(x_pts, axis=2)
    # compute the mls weights (Q, C)
    weights = np.exp(-(x_dists / r) ** 2)
    # multiply vectors by weights
    weighted_x_pts = x_pts * weights[..., None]
    # compute numerator
    numerator = np.tensordot(weighted_x_pts, control_normals, axes=2)
    # denominator
    denominator = np.sum(weights, axis=1)
    # imls values
    return numerator / denominator

if __name__ == "__main__":
    # create a circle
    C = 128
    theta = np.linspace(0, 2 * np.pi, C, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    points = np.vstack((x, y)).T
    points += np.random.randn(*points.shape) * 0.05
    # print(points)

    # compute the normals for each point (done by averaging each edge normal)
    prev_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    prev_edge = points - prev_points
    next_edge = next_points - points
    prev_normal = np.vstack((prev_edge[:, 1], -prev_edge[:, 0])).T
    next_normal = np.vstack((next_edge[:, 1], -next_edge[:, 0])).T
    normals = prev_normal + next_normal
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # query_points = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    # res = imls(query_points, points, normals, r=0.1)

    # create a grid of query points
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    grid_res = imls(grid_points, points, normals, r=0.12)
    grid_res = grid_res.reshape(grid_x.shape)

    # plot the circle
    plt.figure(figsize=(6, 12))
    plt.subplot(2, 1, 1)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1])
    plt.axis('equal')
    plt.subplot(2, 1, 2)
    plt.contourf(grid_x, grid_y, grid_res, levels=100, cmap='RdBu_r', vmin=-3, vmax=3)
    plt.colorbar()
    plt.contour(grid_x, grid_y, grid_res, levels=[0], colors='k')
    plt.axis('equal')
    plt.savefig("imls_circle.png")