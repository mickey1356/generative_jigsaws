import numpy as np
import matplotlib.pyplot as plt
import igl
import nvdiffrast.torch as dr
import scipy.sparse as spp
import scipy.sparse.linalg as spla
import tqdm
import os

import torch
import torch.sparse as tsp
from torch.optim.lr_scheduler import LambdaLR

import mesher
from ts_simple.sd_guidance import StableDiffusionGuidance, StableDiffusionPromptProcessor
from ts_simple.df_guidance import DeepFloydGuidance, DeepFloydPromptProcessor

from helpers import read_image

def get_plane_mesh(subdivs=32, s=(-1, -1), e=(1, 1)):
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

def get_plane_mesh_gmsh(s=(-1, -1), e=(1, 1), d=0.1):
    V = [[s[0], s[1], 0],
         [e[0], s[1], 0],
         [e[0], e[1], 0],
         [s[0], e[1], 0]]
    F = [[0, 1, 2, 3]]
    return mesher.mesh_surface(V, F, d)

# differentiably render V, F using nvdiffrast
# assumes V is in homogeneous coordinates (and batched)
def render(V, F, C, background, glctx, res=256):
    rast, _ = dr.rasterize(glctx, V, F, resolution=[res, res])
    mask = rast[..., -1:] == 0
    col, _ = dr.interpolate(C, rast, F)
    out = dr.antialias(torch.where(mask, background, col), rast, V, F)
    return out

def compute_new_V(rows, cols, data, rhs, nV, nM):
    M = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), data, (nM, nM)).to_dense()
    x = torch.linalg.solve(M, rhs)
    return torch.stack([x[:nV], x[nV:2*nV]], dim=1)

def construct_system_data(x, E, nV, C_d):
    W = torch.sigmoid(x)

    # diagonal entries
    E_d = torch.zeros(nV, device=x.device)
    E_d.index_add_(0, E[:, 0], W)
    E_d.index_add_(0, E[:, 1], W)

    data = torch.cat([-W, -W, -W, -W, E_d, E_d, C_d])
    return data

def precompute_indices(V, F, E, iV_pin, device="cuda"):
    nV = V.shape[0]

    # start with the laplacian indices (edges weights)
    E_r = np.hstack([E[:, 0], E[:, 1]])
    E_c = np.hstack([E[:, 1], E[:, 0]])

    # the diagonal indices
    Di = np.arange(2 * nV)

    C_r = []
    C_c = []
    C_d = []
    for i, c in enumerate(iV_pin):
        # constraint matrix C
        # x coordinate
        C_r.append(2 * nV + i)
        C_c.append(c)
        C_d.append(1)
        # y coordinate
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

    # x, y, then the constraints
    rows = torch.tensor(np.concatenate([E_r, E_r + nV, Di, C_r]), dtype=torch.int32, device=device)
    cols = torch.tensor(np.concatenate([E_c, E_c + nV, Di, C_c]), dtype=torch.int32, device=device)
    C_d = torch.tensor(C_d, dtype=torch.float32, device=device)
    rhs = torch.tensor(np.concatenate([np.zeros(2 * nV), V[iV_pin, 0], V[iV_pin, 1]]), dtype=torch.float32, device=device)
    # returns the row, col indices for the system, the constraint data, and the rhs
    return rows, cols, C_d, rhs
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, -1)

if __name__ == "__main__":
    DEVICE = "cuda:0"
    RES = 256
    ITERATIONS = 1000
    LR = 1e-1
    N_SUBDIVS = 32

    FOLDER = "exp4"

    PROMPTS = ["a silhouette of a cat. trending on artstation.",
               "a silhouette of a cat. trending on artstation.",
               "a silhouette of a cat. trending on artstation."]

    WEIGHTS = [1.0, 1.0, 1.0]

    os.makedirs(FOLDER, exist_ok=True)

    # guidance = StableDiffusionGuidance(device=DEVICE)
    # prompt_processor = StableDiffusionPromptProcessor(device=DEVICE)
    guidance = DeepFloydGuidance(device=DEVICE)
    prompt_processor = DeepFloydPromptProcessor(device=DEVICE)

    embeds = [prompt_processor.get_text_embeddings(p) for p in PROMPTS]
    # text_embeddings = prompt_processor.get_text_embeddings(PROMPT)
    prompt_processor.destroy_text_encoder()
    # print(PROMPT)

    glctx = dr.RasterizeCudaContext(DEVICE)

    # get a plane mesh
    # V, F = get_plane_mesh_gmsh(d=0.025)
    V, F = mesher.circle_in_square(N=32, r=0.8, s=[-1, 1], e=0.03)
    V, F = mesher.semicircles_in_square(N=32, r=0.8, s=[-1, 1], e=0.03)
    nV = V.shape[0]
    E = igl.edges(F)
    print(V.shape, F.shape, E.shape)

    # pin corners
    # iV_pin = np.array([0, N_SUBDIVS, (N_SUBDIVS + 1) * N_SUBDIVS, (N_SUBDIVS + 1) * (N_SUBDIVS + 1) - 1])
    # pin borders
    iV_pin = igl.boundary_loop(F)
    M_r, M_c, C_d, rhs = precompute_indices(V, F, E, iV_pin, device=DEVICE)

    # colors
    red = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)[None, ...].repeat((V.shape[0], 1))
    green = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)[None, ...].repeat((V.shape[0], 1))
    blue = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)[None, ...].repeat((V.shape[0], 1))
    yellow = torch.tensor([1.0, 1.0, 0.0], device=DEVICE)[None, ...].repeat((V.shape[0], 1))
    cyan = torch.tensor([0.0, 1.0, 1.0], device=DEVICE)[None, ...].repeat((V.shape[0], 1))
    magenta = torch.tensor([1.0, 0.0, 1.0], device=DEVICE)[None, ...].repeat((V.shape[0], 1))
    black = torch.zeros((V.shape[0], 3), device=DEVICE)[None, ...]
    bgs = torch.ones((1, RES, RES, 3), device=DEVICE)

    # select all faces that are within a certain distance from the center
    F_centers = np.mean(V[F], axis=1)
    dists = np.linalg.norm(F_centers, axis=1)
    fids = np.where(dists < 0.8)[0]
    # make sure that all such faces are connected
    n, _ = igl.facet_components(F[fids])
    assert n == 1, "faces are not connected"

    # for now, do manual splits
    F_shape_centers = np.mean(V[F[fids]], axis=1)
    F_split_A = np.where(F_shape_centers[:, 1] < 0)[0]
    F_split_B = np.where(F_shape_centers[:, 1] >= 0)[0]

    # get boundary vertices
    iV_shape = np.unique(F[fids].flatten())
    iV_A = np.unique(F[fids[F_split_A]].flatten())
    iV_B = np.unique(F[fids[F_split_B]].flatten())
    iV_bnds = [iV_shape, iV_A, iV_B]

    # convert to torch tensors
    F_shape = torch.tensor(F[fids], device=DEVICE, dtype=torch.int32)
    E = torch.tensor(E, device=DEVICE, dtype=torch.int32)

    F_split_A_tensor = torch.tensor(F[fids[F_split_A]], device=DEVICE, dtype=torch.int32)
    F_split_B_tensor = torch.tensor(F[fids[F_split_B]], device=DEVICE, dtype=torch.int32)

    F_shapes = [F_shape, F_split_A_tensor, F_split_B_tensor]

    # parameterize the shape by using random weights on the edges
    x = torch.randn(E.shape[0], requires_grad=True, device=DEVICE)
    scale = torch.nn.Parameter(torch.tensor(1.0, device=DEVICE), requires_grad=True)

    target_img = torch.tensor(read_image("target3.png", w=RES, h=RES, format="RGB"), device=DEVICE)

    optimizer = torch.optim.Adam([x, scale], lr=LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(ITERATIONS * 1.5))

    # compute V_new
    def compute_V_new(x):
        data = construct_system_data(x, E, nV, C_d)
        V_n = compute_new_V(M_r, M_c, data, rhs, nV, 2 * nV + 2 * len(iV_pin))
        return V_n
    
    # center shape and convert to homogeneous coordinates
    def center_shape(V_n, iV_bnd):
        # V_shape_bnd = V_n[iV_bnd]
        # shape_center = V_shape_bnd.mean(dim=0)
        # V_centered = V_n - shape_center[None, :]
        V_hom = torch.hstack([V_n, torch.zeros((V_n.shape[0], 1), device=DEVICE), torch.ones((V_n.shape[0], 1), device=DEVICE)])[None, ...]
        return V_hom
    
    # render initial shape
    V_init = torch.hstack([torch.tensor(V, device=DEVICE, dtype=torch.float32), torch.ones((V.shape[0], 1), device=DEVICE)])[None, ...]
    print(V_init.shape)
    img_top_red = render(V_init, F_split_A_tensor, red, bgs, glctx, RES)
    img_composite = render(V_init, F_split_B_tensor, blue, img_top_red, glctx, RES)
    img_composite = img_composite[0].detach().cpu().numpy()
    plt.imsave(f"{FOLDER}/-1.png", img_composite)

    for it in tqdm.trange(ITERATIONS):
        optimizer.zero_grad()

        V_n = compute_V_new(x)

        imgs = []
        losses = []

        # render images
        for Fs, iV_bnd, text_embeds, wt in zip(F_shapes, iV_bnds, embeds, WEIGHTS):
            V_hom = center_shape(V_n, iV_bnd)
            img = render(V_hom, Fs, black, bgs, glctx, RES)
            losses.append((guidance(img, text_embeds)["loss_sds"], wt))
            imgs.append(img[0].detach().cpu().numpy())

        # render image
        # V_hom = center_shape(V_n, iV_bnds[0])
        # img = render(V_hom, F_shape, black, bgs, glctx, RES)

        # render the splits
        # img_top = render(V_hom, F_split_A_tensor, black, bgs, glctx, RES)
        # img_btm = render(V_hom, F_split_B_tensor, black, bgs, glctx, RES)

        # losses.append((guidance(img, text_embeddings)["loss_sds"], 1.2))
        # losses.append(((img - target_img).pow(2).sum(), 1))

        loss = sum([l * w for l, w in losses])
        loss.backward()

        # loss function
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # show it
            if it % 50 == 0:
                V_hom = center_shape(V_n, iV_bnds[0])
                img_top_red = render(V_hom, F_split_A_tensor, red, bgs, glctx, RES)
                img_composite = render(V_hom, F_split_B_tensor, blue, img_top_red, glctx, RES)
                img_composite = img_composite[0].detach().cpu().numpy()
                plt.imsave(f"{FOLDER}/{it}.png", img_composite)

                # plt.imsave(f"{FOLDER}/{it}.png", img.detach().cpu().numpy()[0])
                # plt.imsave(f"{FOLDER}/top/{it}.png", imgs[1])
                # plt.imsave(f"{FOLDER}/btm/{it}.png", imgs[2])

    with torch.no_grad():
        data = construct_system_data(x, E, nV, C_d)
        V_n = compute_new_V(M_r, M_c, data, rhs, nV, 2 * nV + 2 * len(iV_pin)) * scale
        # turn into homogeneous coordinates and batch
        V_hom = torch.hstack([V_n, torch.zeros((V_n.shape[0], 1), device=DEVICE), torch.ones((V_n.shape[0], 1), device=DEVICE)])[None, ...]
        
        # render image
        img = render(V_hom, F_shape, black, bgs, glctx, RES)
        img = img[0].detach().cpu().numpy()
        plt.imsave(f"{FOLDER}/final.png", img)

        img_top = render(V_hom, F_split_A_tensor, black, bgs, glctx, RES)
        img_top = img_top[0].detach().cpu().numpy()
        plt.imsave(f"{FOLDER}/final_top.png", img_top)

        img_btm = render(V_hom, F_split_B_tensor, black, bgs, glctx, RES)
        img_btm = img_btm[0].detach().cpu().numpy()
        plt.imsave(f"{FOLDER}/final_btm.png", img_btm)

        # composite the two
        img_top_red = render(V_hom, F_split_A_tensor, red, bgs, glctx, RES)
        img_composite = render(V_hom, F_split_B_tensor, blue, img_top_red, glctx, RES)
        img_composite = img_composite[0].detach().cpu().numpy()
        plt.imsave(f"{FOLDER}/final_composite.png", img_composite)

        V_hom_np = V_hom[0].detach().cpu().numpy()
        F_np = F_shape.detach().cpu().numpy()
        plt.figure(dpi=300)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.triplot(V_hom_np[:, 0], V_hom_np[:, 1], F_np, color="black", lw=0.5)
        plt.gca().invert_yaxis()
        plt.savefig(f"{FOLDER}/mesh_full.png")

        F_top_np = F_split_A_tensor.detach().cpu().numpy()
        plt.figure(dpi=300)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.triplot(V_hom_np[:, 0], V_hom_np[:, 1], F_top_np, color="black", lw=0.5)
        plt.gca().invert_yaxis()
        plt.savefig(f"{FOLDER}/mesh_top.png")

        F_btm_np = F_split_B_tensor.detach().cpu().numpy()
        plt.figure(dpi=300)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.triplot(V_hom_np[:, 0], V_hom_np[:, 1], F_btm_np, color="black", lw=0.5)
        plt.gca().invert_yaxis()
        plt.savefig(f"{FOLDER}/mesh_btm.png")

        with open(f"{FOLDER}/mesh.obj", "w") as f:
            for v in V_hom_np:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in F_np:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

