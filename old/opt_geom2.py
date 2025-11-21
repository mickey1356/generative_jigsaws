# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import igl
import torch
import nvdiffrast.torch as dr
import scipy.sparse as spp
import scipy.sparse.linalg as spla

# %%
def get_plane_mesh(subdivs=32, s=(-0.8, -0.8), e=(0.8, 0.8)):
    assert isinstance(subdivs, int) and subdivs > 0, "subdivs must be a positive integer"

    # create a grid of vertices
    verts = []
    for i in range(subdivs + 1):
        for j in range(subdivs + 1):
            verts.append([i / subdivs, j / subdivs, 0])
    verts = np.array(verts)
    # scale to [-1, 1]
    verts[:, 0] = verts[:, 0] * (e[0] - s[0]) + s[0]
    verts[:, 1] = verts[:, 1] * (e[1] - s[1]) + s[1]

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

    return verts, faces

# %%
def render(V, F, C, background, glctx, res=256):
    rast, _ = dr.rasterize(glctx, V, F, resolution=[res, res])
    mask = rast[..., -1:] == 0
    col, _ = dr.interpolate(C, rast, F)
    out = dr.antialias(torch.where(mask, background, col), rast, V, F)
    return out

# %%
V, F = get_plane_mesh(64)
print(V.shape, F.shape)
E = igl.edges(F)

# find the most central vertex and pick one of its neighbors
dists = np.linalg.norm(V[:, :2], axis=1)
vid = np.argmin(dists)
nbs = E[(E[:, 0] == vid) | (E[:, 1] == vid)]
nbs = nbs[nbs != vid]
nb_vid = np.random.choice(nbs.flatten())
# from the neighbor, pick another neighbor that isnt adjacent to the original vertex
nnbs = E[(E[:, 0] == nb_vid) | (E[:, 1] == nb_vid)]
nnbs = nnbs[nnbs != nb_vid]
print(nbs, nnbs)
# remove shared neighbors
shared = set(nbs.flatten()).intersection(set(nnbs.flatten()))
nnbs = [n for n in nnbs.flatten() if n not in shared and n != vid]
# pick a random one
nnb_vid = np.random.choice(nnbs)
print(f"Central vertex {vid}, neighbor {nb_vid}, second neighbor {nnb_vid}")

# pick a random face
fid = np.random.randint(F.shape[0])
print(f"Random face {fid}")

plt.triplot(V[:, 0], V[:, 1], F)
plt.scatter(V[vid, 0], V[vid, 1], color='red')
plt.scatter(V[nb_vid, 0], V[nb_vid, 1], color='blue')
plt.scatter(V[nnb_vid, 0], V[nnb_vid, 1], color='green')
plt.scatter(V[F[fid], 0], V[F[fid], 1], color='orange')

# %%
def gen_weights(V, F):
    E = igl.edges(F)
    W = np.random.rand(E.shape[0])
    rows = []
    cols = []
    data = []
    for (i, j), w in zip(E, W):
        # x coord
        rows.append(i)
        cols.append(j)
        rows.append(j)
        cols.append(i)
        data.append(w)
        data.append(w)
        # # y coord
        # rows.append(2 * i + 1)
        # cols.append(2 * j + 1)
        # rows.append(2 * j + 1)
        # cols.append(2 * i + 1)
        # data.append(w)
        # data.append(w)
    Wm = spp.coo_matrix((data, (rows, cols)), shape=(V.shape[0], V.shape[0])).tocsc()
    return Wm

def gen_weights2(V, F):
    E = igl.edges(F)
    nV = V.shape[0]
    W = np.random.rand(E.shape[0])
    rows = []
    cols = []
    data = []
    for (i, j), w in zip(E, W):
        # x coord
        rows.append(i)
        cols.append(j)
        rows.append(j)
        cols.append(i)
        data.append(w)
        data.append(w)
        # # y coord
        rows.append(nV + i)
        cols.append(nV + j)
        rows.append(nV + j)
        cols.append(nV + i)
        data.append(w)
        data.append(w)
    Wm = spp.coo_matrix((data, (rows, cols)), shape=(2 * V.shape[0], 2 * V.shape[0])).tocsc()
    return Wm


# %%
np.random.seed(42)
E = igl.edges(F)
W = np.random.rand(E.shape[0])
nV = V.shape[0]

E_r = np.hstack([E[:, 0], E[:, 1]])
E_c = np.hstack([E[:, 1], E[:, 0]])

E_d = np.zeros(nV)
np.add.at(E_d, E[:, 0], W)
np.add.at(E_d, E[:, 1], W)
print(E_d)
Di = np.arange(2 * nV)

# pinned vertices
iV_pin = np.array([0, 64, (64 + 1) * 64, (64 + 1) * (64 + 1) - 1])
V_pin = V[iV_pin, :2]
# construct the constraint indices
C_r = []
C_c = []
C_d = []
for i, c in enumerate(iV_pin):
    # constraint
    C_r.append(2 * nV + i)
    C_c.append(c)
    C_d.append(1)
    C_r.append(2 * nV + i + len(iV_pin))
    C_c.append(c + nV)
    C_d.append(1)
    # and its transpose
    C_r.append(c)
    C_c.append(2 * nV + i)
    C_d.append(1)
    C_r.append(c + nV)
    C_c.append(2 * nV + i + len(iV_pin))
    C_d.append(1)

L2 = spp.coo_matrix((np.concatenate([-W, -W, -W, -W, E_d, E_d, C_d]), (np.concatenate([E_r, E_r + nV, Di, C_r]), np.concatenate([E_c, E_c + nV, Di, C_c]))), shape=(2 * nV + 2 * len(iV_pin), 2 * nV + 2 * len(iV_pin))).tocsc()
print(L2.shape)
print(L2.todense()[nV-1:nV+5, nV-1:nV+5])

rhs = np.concatenate([np.zeros(2 * nV), V[iV_pin, 0], V[iV_pin, 1]])
x = spp.linalg.spsolve(L2, rhs)
V_new = np.zeros_like(V)
V_new[:, 0] = x[:nV]
V_new[:, 1] = x[nV:2 * nV]
print(nV, x.shape)
plt.triplot(V_new[:, 0], V_new[:, 1], F)
plt.scatter(V_new[vid, 0], V_new[vid, 1], color='red')
plt.scatter(V_new[nb_vid, 0], V_new[nb_vid, 1], color='blue')
plt.scatter(V_new[nnb_vid, 0], V_new[nnb_vid, 1], color='green')


# %%
# construct a laplacian matrix
# [ L C^T \ C 0 ] v_new = [0 \ b]
# generate random laplacian weights
np.random.seed(42)
W = gen_weights2(V, F)
D = spp.diags(np.asarray(np.sum(W, axis=1)).squeeze())
L = D - W

# diagonal entries
# for i in range(V.shape[0]):
#     rows.append(2 * i)
#     cols.append(2 * i)
#     rows.append(2 * i + 1)
#     cols.append(2 * i + 1)
#     d = np.sum(W[E[:, 0] == i]) + np.sum(W[E[:, 1] == i])
#     data.append(d)
#     data.append(d)

# L = spp.coo_matrix((data, (rows, cols)), shape=(2 * V.shape[0], 2 * V.shape[0])).tocsc()
# L += 1e-6 * spp.eye(L.shape[0])

# form the rhs
b = np.zeros((2 * V.shape[0]))

# build a constraint matrix to avoid degeneracy
# pin corners to the bounds
# iV_pin = igl.boundary_loop(F)

C_r = []
C_c = []
C_d = []
for i, c in enumerate(iV_pin):
    # x coord
    C_r.append(i)
    C_c.append(c)
    C_d.append(1)
    # y coord
    C_r.append(i + len(iV_pin))
    C_c.append(nV + c)
    C_d.append(1)

C = spp.coo_matrix((C_d, (C_r, C_c)), shape=(2 * len(iV_pin), 2 * V.shape[0])).tocsc()
print(C.shape)

# rhs
cb = np.concatenate((V[iV_pin, 0], V[iV_pin, 1]))
# print(V[i])
# print(cb)

# cb = np.zeros(C.shape[0])
# cb[0] = V[:, 0].sum()
# cb[1] = V[:, 1].sum()
# cb[2] = 1
# cb[3] = -1

# concat the rhs
rhs = np.concatenate([b, cb])

# form the full matrix
M = spp.block_array([[L, C.T], [C, None]]).tocsc()
print(f"M shape: {M.shape}")
print(M.todense()[nV-1:nV+5, nV-1:nV+5])
print(np.allclose(M.todense(), L2.todense()))
# solve the system
x = spp.linalg.spsolve(M, rhs)
V_new = np.zeros_like(V)
V_new[:, 0] = x[:nV]
V_new[:, 1] = x[nV:2 * nV]
print(nV, x.shape)
plt.triplot(V_new[:, 0], V_new[:, 1], F)
plt.scatter(V_new[vid, 0], V_new[vid, 1], color='red')
plt.scatter(V_new[nb_vid, 0], V_new[nb_vid, 1], color='blue')
plt.scatter(V_new[nnb_vid, 0], V_new[nnb_vid, 1], color='green')

# %%
W = gen_weights(V, F)
print(W)

def normalized_spectral_embedding(W, k=2):
    # W: symmetric weight sparse matrix
    deg = np.array(W.sum(axis=1)).ravel()
    D = spp.diags(deg)
    L = D - W
    # generalized eigenproblem: L φ = λ D φ
    vals, vecs = spla.eigsh(L, k=k+1, M=D, which='SM')
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    X = vecs[:, 1:k+1]  # skip constant eigenvector
    return X, vals[1:k+1]

V_n, _ = normalized_spectral_embedding(W)
plt.figure()
plt.triplot(V[:, 0], V[:, 1], F)
plt.scatter(V[:, 0], V[:, 1], color="red", s=10)
plt.figure()
plt.triplot(V_n[:, 0], V_n[:, 1], F)
plt.scatter(V_n[:, 0], V_n[:, 1], color="red", s=10)
with open("test.obj", "w") as f:
    for v in V_n:
        f.write(f"v {v[0]} {v[1]} 0\n")
    for face in (F + 1):
        f.write(f"f {face[0]} {face[1]} {face[2]}\n")