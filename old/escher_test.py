#%%
import numpy as np
import math
import igl


def get_hexagonal_mesh(vertices_per_edge = 65):
    """Generator for coordinates in a hexagon."""
    steps = np.floor(math.log2(vertices_per_edge - 1)).astype(int)
    vertices_per_edge = 2**steps + 1
    assert steps.is_integer(),"the number of vertices per edge should be a power of 2 plus, i.e., 2^n+1 "
    steps = int(steps)
    edge_length = 2 * math.sqrt(3) / 3 
    vertices = []
    vertices.append([0,0])
    triangles = []
    for i in range(6):
        angle = i*2*math.pi/6
        point = [math.cos(angle),math.sin(angle)]
        vertices.append(point)
        cur = i+1
        next = i+2
        if next > 6:
            next = 1
        triangle = [0,cur,next]
        triangles.append(triangle)
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    for step in range(steps):
        vertices, triangles = igl.upsample(vertices, triangles)

    bdry = igl.boundary_loop(triangles)
    for i in range(len(bdry)):
        bdry = np.roll(bdry,1)
        ind = bdry[0]
        if vertices[ind,0] == 1 and vertices[ind,1] ==0: #this is the first vertex of the basic hexagon and cos(0) sin(0)
            break
    else:
        raise Exception("couldn't find the start vertex")
    sides = {}
    for i in range(6):
        start = i*(vertices_per_edge-1)
        end = start+vertices_per_edge
        sides[i] = bdry[start:end]
    sides[5] = np.append(sides[5],sides[0][0])
    # from escher.geometry.sanity_checks import check_triangle_orientation
    # print("Hexagon created: sanity check of the triangles orientation...")
    # check_triangle_orientation(vertices,triangles)
    print("done.")
    return vertices,triangles,sides

#%%
V, T, S = get_hexagonal_mesh(8)
# %%
V.shape
# %%
T.shape
# %%
import matplotlib.pyplot as plt

plt.triplot(V[:,0],V[:,1],T)
print(T[:3])
# %%
S[0]
print(V[S[0]])
print(V[S[1]])
# %%
igl.boundary_loop(T)
print(S[0], S[1], S[2], S[3], S[4], S[5])
# %%
adj_list = igl.adjacency_list(T)
edge_pairs = []
for r, i in zip(adj_list, range(len(adj_list))):
    for j in r:
        if i < j:
            edge_pairs.append((i,j))
print(edge_pairs)
print(adj_list)

# %%
E = igl.edges(T)
Ea = np.asarray(edge_pairs)
print(E.shape, Ea.shape)


# %%
def get_2d_square_mesh(resolution, num_labels=1):
    """get_2d_square_mesh

    Args:
        resolution (_type_): 50
        num_labels (int, optional): Split the faces in separate groups. If num_labels=2, the split is diagonal, else num_labels has to be a square number
        and the triangle are split using a grid. Defaults to 1.

    Returns:
        _type_: vertices, faces, and per-face labels
    """
    nx, ny = (resolution, resolution)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()
    yv = yv.ravel()
    points = np.stack((xv, yv), axis=1)

    faces_1 = np.concatenate([np.array([[i, i + 1, i + resolution + 1]]) for i in range((resolution) ** 2)])
    mask_1 = np.stack(
        [
            True if (i % resolution != (resolution - 1)) and (i + resolution + 1 < resolution**2) else False
            for i in range(resolution**2)
        ]
    )
    faces_1[resolution-2] = np.array([resolution-2, resolution-1, resolution-2 + resolution])
    faces_1[resolution ** 2 - 2*resolution] = np.array([resolution ** 2 - 2*resolution, resolution ** 2 - 2*resolution + 1, resolution ** 2 - 2*resolution + resolution])
    faces_1 = faces_1[mask_1]
    
    faces_2 = np.concatenate([np.array([[i + resolution, i, i + resolution + 1]]) for i in range((resolution) ** 2)])
    mask_2 = np.stack(
        [
            True if (i % resolution != (resolution - 1)) and (i + resolution + 1 < resolution**2) else False
            for i in range(resolution**2)
        ]
    )
    faces_2[resolution-2] = np.array([resolution-1, resolution-1 + resolution, resolution-2 + resolution])
    faces_2[resolution ** 2 - 2*resolution] = np.array([resolution ** 2 - resolution + 1, resolution ** 2 - resolution, resolution ** 2 - 2*resolution + 1])
   
    faces_2 = faces_2[mask_2]
    faces = np.concatenate([faces_1, faces_2])

    mask = []
    for tri in faces:
        a, b, c = tri
        pa, pb, pc = points[a], points[b], points[c]
        # if one point of the triangle falls in the upper right diagonal, mask=True
        if num_labels == 2 or num_labels == 1:
            if pa[0] > pa[1] or pb[0] > pb[1] or pc[0] > pc[1]:
                mask.append(False)
            else:
                mask.append(True)
        else:
            grid_size = np.sqrt(num_labels)
            bin_size = 1 / grid_size
            # rescale to 0,1
            pa, pb, pc = (pa + 1) / 2, (pb + 1) / 2, (pc + 1) / 2
            # find bin index
            x_bin = min(max(pa[0] // bin_size, pb[0] // bin_size, pc[0] // bin_size), grid_size - 1)
            y_bin = min(max(pa[1] // bin_size, pb[1] // bin_size, pc[1] // bin_size), grid_size - 1)
            label = x_bin + y_bin * grid_size
            mask.append(label)
    mask = np.stack(mask)

    # print(mask.max())
    if num_labels == 1:
        faces_split = [faces]
    elif num_labels == 2:
        faces_split = [faces[mask], faces[~mask]]
    else:
        faces_split = []
        for label in range(num_labels):
            faces_split.append(faces[mask == label])

    # print(points.shape, faces.shape, mask.shape)
    # return points, faces, None, None
    return points, faces, faces_split, mask

# %%
V, F, Fs, M = get_2d_square_mesh(8, num_labels=4)

import matplotlib.pyplot as plt

plt.triplot(V[:,0],V[:,1],F)
print(F[:3])

fig, axs = plt.subplots(1, 4)
axs[0].triplot(V[:,0],V[:,1],Fs[0])
axs[0].set_title("Face Group 1")
axs[1].triplot(V[:,0],V[:,1],Fs[1])
axs[1].set_title("Face Group 2")
axs[2].triplot(V[:,0],V[:,1],Fs[2])
axs[2].set_title("Face Group 3")
axs[3].triplot(V[:,0],V[:,1],Fs[3])
axs[3].set_title("Face Group 4")

# %%
print(len(M))
print(len(F))
print(Fs[0])
print(np.all(Fs[0] == F[M==0]))

# %%
def get_plane_mesh(subdivs = 32):
    # create a grid of points
    points = []
    for i in range(subdivs + 1):
        for j in range(subdivs + 1):
            points.append([i / subdivs, j / subdivs, 0])
    points = np.array(points)

    # create a grid of faces
    faces = []
    for i in range(subdivs):
        for j in range(subdivs):
            v0 = i * (subdivs + 1) + j
            v1 = v0 + 1
            v2 = v0 + (subdivs + 1)
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces)

    return points, faces

# %%
V, F = get_plane_mesh(4)
plt.triplot(V[:,0],V[:,1],F)
