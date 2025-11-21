import numpy as np
import matplotlib.pyplot as plt
import igl
import nvdiffrast.torch as dr
import scipy.sparse as spp
import scipy.sparse.linalg as spla
import tqdm

import torch
from torch.optim.lr_scheduler import LambdaLR


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

# differentiably render V, F using nvdiffrast
# assumes V is in homogeneous coordinates (and batched)
def render(V, F, C, background, glctx, res=256):
    rast, _ = dr.rasterize(glctx, V, F, resolution=[res, res])
    mask = rast[..., -1:] == 0
    col, _ = dr.interpolate(C, rast, F)
    out = dr.antialias(torch.where(mask, background, col), rast, V, F)
    return out

def construct_W(W, E, nV):
    all_edges = np.vstack([E, E[:, [1, 0]]])
    return torch.sparse_coo_tensor(all_edges.T, torch.cat([W, W], dim=0), (nV, nV)).to_dense()

def compute_V(x, E, nV):
    # given W, compute new vertex positions
    x = torch.sigmoid(x)
    W = construct_W(x, E, nV)
    dd = torch.sum(W, dim=1)
    D = torch.diag(dd)
    L = D - W
    _, Q = torch.linalg.eigh(L)
    return Q[:, 1:3]

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, -1)


if __name__ == "__main__":
    DEVICE = "cuda"
    RES = 256
    SCALE = 10
    ITERATIONS = 1000
    LR = 1e-2

    glctx = dr.RasterizeCudaContext()

    # get a plane mesh
    V, F = get_plane_mesh(32)
    E = igl.edges(F)
    print(V.shape, F.shape, E.shape)
    F = torch.tensor(F, device=DEVICE, dtype=torch.int32)

    # colors
    black = torch.zeros((V.shape[0], 3), device=DEVICE)[None, ...]
    bgs = torch.ones((1, RES, RES, 3), device=DEVICE)

    # parameterize the shape by using random weights on the edges
    x = torch.randn(E.shape[0], requires_grad=True, device=DEVICE)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=DEVICE), requires_grad=True)

    target_img = torch.tensor(plt.imread("target.png"), device=DEVICE)

    optimizer = torch.optim.Adam([x, scale], lr=LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(ITERATIONS * 1.5))

    for it in tqdm.trange(ITERATIONS):
        optimizer.zero_grad()

        V_n = compute_V(x, E, V.shape[0]) * scale
        # turn into homogeneous coordinates and batch
        V_hom = torch.hstack([V_n, torch.zeros((V_n.shape[0], 1), device=DEVICE), torch.ones((V_n.shape[0], 1), device=DEVICE)])[None, ...]
        # print(V_hom.shape)
        # render image
        img = render(V_hom, F, black, bgs, glctx, RES)

        loss = (img - target_img).pow(2).sum()
        loss.backward()

        # loss function
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # show it
            if it % 50 == 0 or it == ITERATIONS - 1:
                img = img[0].detach().cpu().numpy()
                plt.imsave(f"optnv/{it}.png", img)
                print(scale)
