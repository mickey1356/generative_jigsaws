# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
# we parameterize the sdf using a grid of values
# the grid is defined over [-1, 1]
def create_grid(resolution, domain=1):
    x = np.linspace(-domain, domain, resolution)
    y = np.linspace(-domain, domain, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y

# create an example sdf: circle at origin with radius 0.5
def circle_sdf(X, Y, center=(0, 0), radius=0.5):
    return np.sqrt((X - center[0])**2 + (Y - center[1])**2) - radius


def plot_sdf(X, Y, S, cbar=False, ax=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    # plot the 0 contour
    cf = ax.contourf(X, Y, S, levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
    if cbar:
        plt.colorbar(mappable=cf)
    ax.contour(X, Y, S, levels=[0])
    ax.axis('equal')

# %%
# visualize the sdf
resolution = 30
X, Y = create_grid(resolution)
S = circle_sdf(X, Y, center=(0, 0), radius=0.8)

plt.figure(figsize=(6, 6))
# plot the 0 contour
plt.contourf(X, Y, S, levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
plt.colorbar()
contour = plt.contour(X, Y, S, levels=[0])
plt.title("Circle SDF")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.show()

# %%
from scipy.ndimage import map_coordinates
import skfmm

def redistance_sdf(sdf, domain=1):
    dx = (2 * domain) / (sdf.shape[0] - 1)
    return skfmm.distance(sdf, dx=dx)

def sample_sdf(sdf, x_local, y_local):
    r = sdf.shape[0]
    # map x_local, y_local from [-1, 1] to [0, r-1]
    x_idx = ((x_local + 1) / 2) * (r - 1)
    y_idx = ((y_local + 1) / 2) * (r - 1)
    print(x_idx[:3, :3])
    print(y_idx[:3, :3])
    # using scipy.ndimage.map_coordinates
    coords = np.vstack([y_idx.flatten(), x_idx.flatten()])  # map_coordinates expects (rows, cols)
    return map_coordinates(sdf, coords, order=3, cval=100).reshape(x_local.shape)

def local_to_global(sdf, scale, pos, R, D):
    # create global grid
    X = np.linspace(-D, D, R)
    Y = np.linspace(-D, D, R)
    X, Y = np.meshgrid(X, Y)

    # map from global [-D, D] to local [-1, 1]
    x_l = (X - pos[0]) / scale
    y_l = (Y - pos[1]) / scale

    # map from local to indices
    i = (x_l + 1) / 2 * (sdf.shape[0] - 1)
    j = (y_l + 1) / 2 * (sdf.shape[0] - 1)

    coords = np.vstack([j.flatten(), i.flatten()])
    return map_coordinates(sdf, coords, order=3, cval=100).reshape(R, R)

def global_to_local(sdf, scale, pos, r, D):
    # create local grid
    x = np.linspace(-1, 1, r)
    y = np.linspace(-1, 1, r)
    X, Y = np.meshgrid(x, y)

    # map from local to global
    x_g = X * scale + pos[0]
    y_g = Y * scale + pos[1]

    # map from global to indices
    i = (x_g + D) / (2 * D) * (sdf.shape[0] - 1)
    j = (y_g + D) / (2 * D) * (sdf.shape[0] - 1)

    coords = np.vstack([j.flatten(), i.flatten()])

    return map_coordinates(sdf, coords, order=1, mode="nearest", cval=100).reshape(r, r)

def join_sdfs(R, D, a_sdf, a_scale, a_pos, b_sdf, b_scale, b_pos):
    # R: resolution of the output sdf
    # D: domain of the output sdf [-D, D]^2

    a_vals = local_to_global(a_sdf, a_scale, a_pos, R, D)
    b_vals = local_to_global(b_sdf, b_scale, b_pos, R, D)

    # compute union
    S = np.minimum(a_vals, b_vals)

    # we split a_new, b_new by taking a_new to be the main piece and b_new to be the subtracted piece
    a_new = np.maximum(S, a_vals)
    b_new = np.maximum(-a_new, S)

    a_new = global_to_local(a_new, a_scale, a_pos, a_sdf.shape[0], D)
    b_new = global_to_local(b_new, b_scale, b_pos, b_sdf.shape[0], D)

    # S = np.minimum(a_new, b_new)

    # create a global grid for output
    x = np.linspace(-D, D, R)
    y = np.linspace(-D, D, R)
    X, Y = np.meshgrid(x, y)
    # redistance
    re_S = redistance_sdf(S, domain=D)

    return X, Y, re_S, redistance_sdf(a_new), redistance_sdf(b_new)

def join_sdfs_multiple(r, R, D, sdf_list, scale_list, pos_list):
    assert len(sdf_list) == len(scale_list) == len(pos_list)
    n = len(sdf_list)

    # convert to global
    global_sdfs = [local_to_global(sdf, scale, pos, R, D) for sdf, scale, pos in zip(sdf_list, scale_list, pos_list)]

    # compute union
    S = np.minimum.reduce(global_sdfs)

    # split into individual sdfs
    split_sdfs = []
    split = global_sdfs[0]
    # split1: A & U
    # split2: B & U - split1
    # split3: C & U - split1 - split2
    # ...
    union_so_far = S.copy()
    for i in range(n): 
        split = np.maximum(global_sdfs[i], union_so_far)
        split_sdfs.append(split)
        union_so_far = np.maximum(-split, union_so_far)

    # convert back to local
    local_sdfs = [global_to_local(split, scale, pos, r, D) for split, scale, pos in zip(split_sdfs, scale_list, pos_list)]

    return redistance_sdf(S, domain=D), local_sdfs

resolution = 30
X, Y = create_grid(resolution)
C1 = circle_sdf(X, Y, center=(0, 0), radius=0.8)
C2 = circle_sdf(X, Y, center=(0, 0), radius=0.8)
C3 = circle_sdf(X, Y, center=(0, 0), radius=0.5)

big_res = 128
D = 4
nX, nY = create_grid(big_res, domain=D)

sdfs = [C1, C2, C3]
scales = [1, 0.8, 1]
positions = [(-0.4, 0), (0.5, 0), (0, -0.3)]

nsdf, [na, nb, nc] = join_sdfs_multiple(resolution, big_res, D, sdfs, scales, positions)

plt.figure(figsize=(24, 6))
plt.subplot(1, 4, 1)
plt.contourf(nX, nY, nsdf, levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
# plt.colorbar()
contour = plt.contour(nX, nY, nsdf, levels=[0])
plt.subplot(1, 4, 2)
plt.contourf(X, Y, na, levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
# plt.colorbar()
contour = plt.contour(X, Y, na, levels=[0])
plt.subplot(1, 4, 3)
plt.contourf(X, Y, nb, levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
# plt.colorbar()
contour = plt.contour(X, Y, nb, levels=[0])
plt.subplot(1, 4, 4)
plt.contourf(X, Y, nc, levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
# plt.colorbar()
contour = plt.contour(X, Y, nc, levels=[0])

# %%
import torch
import torchvision

def render_sdf(sdf, k=10, img_res=256):
    # scale the sdf to the image resolution
    sdf_img_res = torch.nn.functional.interpolate(
        sdf.unsqueeze(0).unsqueeze(0), size=(img_res, img_res), mode='bicubic', align_corners=False
    )

    # soft occupancy
    # 0 is inside (black), 1 is outside (white)
    img = 1 - torch.sigmoid(-k * sdf_img_res)
    # [B, C, R, R]
    return img.squeeze()

img = render_sdf(torch.tensor(S, dtype=torch.float32), k=100, img_res=256)
plt.imshow(img.numpy(), cmap='gray')

redist_sdf = redistance_sdf(S)
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, redist_sdf, levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
plt.colorbar()
contour = plt.contour(X, Y, redist_sdf, levels=[0])
plt.show()
# %%
from helpers import read_image
IMG_RES = 256
DEVICE = "cuda:1"

target_img = torch.tensor(read_image("target4.png", w=IMG_RES, h=IMG_RES, format="RGB"), device=DEVICE, dtype=torch.float32)
print(target_img.shape, target_img.max(), target_img.min())
plt.imshow(target_img.cpu().numpy())

# %%
iters = 200

opt_sdf = torch.tensor(S, dtype=torch.float32, device=DEVICE, requires_grad=True)
optimizer = torch.optim.Adam([opt_sdf], lr=0.01)

for it in range(iters):
    optimizer.zero_grad()
    rendered_img = render_sdf(opt_sdf, k=50, img_res=IMG_RES)
    rendered_img_3c = rendered_img.unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0)  # make 3 channels
    loss = torch.nn.functional.mse_loss(rendered_img_3c, target_img)
    loss.backward()
    optimizer.step()
    # redistance the sdf to keep it valid
    with torch.no_grad():
        opt_sdf.copy_(torch.tensor(redistance_sdf(opt_sdf.cpu().numpy()), device=DEVICE, dtype=torch.float32))
    if it % 100 == 0 or it == iters - 1:
        print(f"Iter {it}, Loss: {loss.item()}")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Rendered Image")
        plt.imshow(rendered_img.cpu().detach().numpy(), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Optimized SDF")
        plt.contourf(X, Y, opt_sdf.cpu().detach().numpy(), levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
        contour = plt.contour(X, Y, opt_sdf.cpu().detach().numpy(), levels=[0])
        plt.gca().invert_yaxis()
        plt.show()

# %%
import torch.nn.functional as F

def local_to_global_torch(sdf, scale, pos, R, D):
    # create global grid
    X = torch.linspace(-D, D, R, device=sdf.device)
    Y = torch.linspace(-D, D, R, device=sdf.device)
    X, Y = torch.meshgrid(X, Y, indexing='xy')

    # map from global [-D, D] to local [-1, 1]
    x_l = (X - pos[0]) / scale
    y_l = (Y - pos[1]) / scale

    sdf_unsq = sdf.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # grid_sample expects coords in [-1, 1]
    coords = torch.stack([x_l, y_l], dim=-1).unsqueeze(0)  # [1, R, R, 2]
    vals = F.grid_sample(sdf_unsq, coords, mode='bilinear', padding_mode='border', align_corners=True)
    return vals.squeeze() # [R, R]

def local_to_global_torch_batch(sdfs, scales, positions, R, D):
    # sdfs: [B, r, r]
    # scales: [B, 1]
    # positions: [B, 2]
    # create global grid
    X = torch.linspace(-D, D, R, device=sdfs.device)
    Y = torch.linspace(-D, D, R, device=sdfs.device)
    X, Y = torch.meshgrid(X, Y, indexing='xy')

    batch_size = sdfs.shape[0]
    x_l = (X.unsqueeze(0).repeat(batch_size, 1, 1) - positions[:, 0].unsqueeze(-1).unsqueeze(-1)) / scales.unsqueeze(-1).unsqueeze(-1)
    y_l = (Y.unsqueeze(0).repeat(batch_size, 1, 1) - positions[:, 1].unsqueeze(-1).unsqueeze(-1)) / scales.unsqueeze(-1).unsqueeze(-1)

    sdfs_unsq = sdfs.unsqueeze(1) # [B, 1, H, W]
    coords = torch.stack([x_l, y_l], dim=-1)  # [B, R, R, 2]
    vals = F.grid_sample(sdfs_unsq, coords, mode='bilinear', padding_mode='border', align_corners=True)
    return vals.squeeze(1) # [B, R, R]

def global_to_local_torch(sdf, scale, pos, r, D):
    # create local grid
    x = torch.linspace(-1, 1, r, device=sdf.device)
    y = torch.linspace(-1, 1, r, device=sdf.device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # map from local to global
    x_g = X * scale + pos[0]
    y_g = Y * scale + pos[1]

    sdf_unsq = sdf.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # grid_sample expects coords in [-1, 1]
    coords = torch.stack([x_g, y_g], dim=-1).unsqueeze(0) / D # [1, r, r, 2]
    vals = F.grid_sample(sdf_unsq, coords, mode='bilinear', padding_mode='border', align_corners=True)
    return vals.squeeze() # [r, r]

def global_to_local_torch_batch(sdfs, scales, positions, r, D):
    # sdfs: [B, R, R]
    # scales: [B, 1]
    # positions: [B, 2]
    # create local grid
    x = torch.linspace(-1, 1, r, device=sdfs.device)
    y = torch.linspace(-1, 1, r, device=sdfs.device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    batch_size = sdfs.shape[0]
    x_g = X.unsqueeze(0).repeat(batch_size, 1, 1) * scales.unsqueeze(-1).unsqueeze(-1) + positions[:, 0].unsqueeze(-1).unsqueeze(-1)
    y_g = Y.unsqueeze(0).repeat(batch_size, 1, 1) * scales.unsqueeze(-1).unsqueeze(-1) + positions[:, 1].unsqueeze(-1).unsqueeze(-1)

    sdfs_unsq = sdfs.unsqueeze(1) # [B, 1, H, W]
    coords = torch.stack([x_g, y_g], dim=-1) / D  # [B, r, r, 2]
    vals = F.grid_sample(sdfs_unsq, coords, mode='bilinear', padding_mode='border', align_corners=True)
    return vals.squeeze(1) # [B, r, r]

def closest_splits(initial_sdfs):
    # initial_sdfs: [B, R, R]
    # assign each pixel to the sdf with the smallest value (closest)

    winner = torch.argmin(initial_sdfs, dim=0)
    mask = torch.nn.functional.one_hot(winner, num_classes=initial_sdfs.shape[0]).permute(2, 0, 1)  # [B, R, R]

    splits = initial_sdfs * mask + (1 - mask) * 100
    return splits

def subtraction_splits(initial_sdfs, union_sdf):
    # initial_sdfs: [B, R, R]
    # union_sdf: [R, R]
    # split1: A & U
    # split2: B & U - split1
    # split3: C & U - split1 - split2
    # ...
    n = initial_sdfs.shape[0]
    split_sdfs = []
    split = initial_sdfs[0]
    union_so_far = union_sdf.clone()
    for i in range(n): 
        split = torch.maximum(initial_sdfs[i], union_so_far)
        split_sdfs.append(split)
        union_so_far = torch.maximum(-split, union_so_far)
    return torch.stack(split_sdfs, dim=0)


def join_sdfs_torch(r, R, D, sdfs, transforms):
    assert sdfs.shape[0] == transforms.shape[0]

    # convert to global
    global_sdfs = local_to_global_torch_batch(sdfs, transforms[:, 0], transforms[:, 1:3], R, D)

    # compute union
    S = global_sdfs.min(dim=0).values
    # return S, None

    # splits = closest_splits(global_sdfs)
    splits = subtraction_splits(global_sdfs, S)

    # convert back to local coordinates
    local_sdfs = global_to_local_torch_batch(splits, transforms[:, 0], transforms[:, 1:3], r, D)
    return S, local_sdfs

# %%
resolution = 30
X, Y = create_grid(resolution)
C1 = circle_sdf(X, Y, center=(0, 0), radius=0.8)
C2 = circle_sdf(X, Y, center=(0, 0), radius=0.8)
C3 = circle_sdf(X, Y, center=(0, 0), radius=0.8)

scales = [1, 1, 1.3]
positions = [(-0.4, 0), (0.5, 0), (0, -0.4)]
# scale = 1
# pos = (0, 0)

C_batched = torch.tensor(np.stack([C1, C2, C3]), dtype=torch.float32, device=DEVICE)
scales_batched = torch.tensor(scales, dtype=torch.float32, device=DEVICE)
positions_batched = torch.tensor(positions, dtype=torch.float32, device=DEVICE)

tfms_batched = torch.cat([scales_batched.unsqueeze(1), positions_batched], dim=1)

big_res = 128
D = 4
nX, nY = create_grid(big_res, domain=D)

nsdf, splits = join_sdfs_torch(resolution, big_res, D, C_batched, tfms_batched)

# splits_np = splits.cpu().numpy()
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# plot_sdf(X, Y, redistance_sdf(splits_np[0], domain=D), ax=axs[0])
# plot_sdf(X, Y, redistance_sdf(splits_np[1], domain=D), ax=axs[1])
# plot_sdf(X, Y, redistance_sdf(splits_np[2], domain=D), ax=axs[2])

nsdf_np = nsdf.cpu().numpy()
plot_sdf(nX, nY, redistance_sdf(nsdf_np, domain=D))

# %%
# run a simple optimization test to see whether the gradients carry through the l2g and g2l transforms
def optim_single():
    target_img = torch.tensor(read_image("target1.png", w=IMG_RES, h=IMG_RES, format="RGB"), device=DEVICE, dtype=torch.float32)

    iters = 1000
    opt_sdf_b = torch.tensor(np.stack([C1]), dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt_tfms_b = torch.tensor([[0.8, 0, 0]], dtype=torch.float32, device=DEVICE, requires_grad=True)  # scale, pos_x, pos_y


    optimizer = torch.optim.Adam([opt_sdf_b, opt_tfms_b], lr=0.01)

    for it in range(iters):
        optimizer.zero_grad()
        # transform sdf to global scale
        g_sdf_b = local_to_global_torch_batch(opt_sdf_b, opt_tfms_b[:, 0], opt_tfms_b[:, 1:3], big_res, D)
        g_sdf = g_sdf_b.min(dim=0).values
        # render it
        rendered_img = render_sdf(g_sdf, k=50, img_res=IMG_RES)
        rendered_img_3c = rendered_img.unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0)  # make 3 channels
        loss = torch.nn.functional.mse_loss(rendered_img_3c, target_img)
        loss.backward()
        optimizer.step()

        # redistance the sdf to keep it valid
        with torch.no_grad():
            for i in range(opt_sdf_b.shape[0]):
                opt_sdf_b[i].copy_(torch.tensor(redistance_sdf(opt_sdf_b[i].cpu().numpy()), device=DEVICE, dtype=torch.float32))
        
        if it % 100 == 0 or it == iters - 1:
            print(f"Iter {it}, Loss: {loss.item()}")
            print(opt_tfms_b)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Rendered Image")
            plt.imshow(rendered_img.cpu().detach().numpy(), cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("Optimized SDF")
            plt.contourf(X, Y, opt_sdf_b.cpu().detach().numpy()[0], levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
            plt.contour(X, Y, opt_sdf_b.cpu().detach().numpy()[0], levels=[0])
            plt.gca().invert_yaxis()
            plt.show()

optim_single()

# %%
def draw_contours(X, Y, sdfs, tfms, ax=None):
    # assume sdfs, tfms are torch tensors
    # first convert to global
    global_sdfs = local_to_global_torch_batch(sdfs, tfms[:, 0], tfms[:, 1:3], big_res, D)
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    for i in range(global_sdfs.shape[0]):
        ax.contour(X, Y, global_sdfs[i].cpu().detach().numpy(), levels=[0], colors=f"C{i}")
    ax.set_aspect('equal')

# test batch optimization (multiple sdfs)
def optim_batch():
    target_img = torch.tensor(read_image("target2.png", w=IMG_RES, h=IMG_RES, format="RGB"), device=DEVICE, dtype=torch.float32)

    iters = 1000
    opt_sdfs = torch.tensor(np.stack([C1, C2, C3]), dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt_tfms = torch.tensor([[1, 0.3, 0.5], [1, -0.3, 0.5], [1, 0, -0.5]], dtype=torch.float32, device=DEVICE, requires_grad=True)  # scale, pos_x, pos_y

    optimizer = torch.optim.Adam([opt_sdfs, opt_tfms], lr=0.01)
    for it in range(iters):
        optimizer.zero_grad()
        # join sdfs
        g_sdf, split_sdfs = join_sdfs_torch(resolution, big_res, D, opt_sdfs, opt_tfms)
                
        # render it
        rendered_img = render_sdf(g_sdf, k=50, img_res=IMG_RES)
        rendered_img_3c = rendered_img.unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0)  # make 3 channels
        loss = torch.nn.functional.mse_loss(rendered_img_3c, target_img)
        loss.backward()
        optimizer.step()
        
        # redistance the local sdfs to keep them valid
        with torch.no_grad():
            for i in range(opt_sdfs.shape[0]):
                opt_sdfs[i].copy_(torch.tensor(redistance_sdf(opt_sdfs[i].cpu().numpy()), device=DEVICE, dtype=torch.float32))
        
        if it % 100 == 0 or it == iters - 1:
            print(f"Iter {it}, Loss: {loss.item()}")
            fig, axs = plt.subplots(3, 2, figsize=(12, 12))
            axs[0, 0].set_title("Rendered Image")
            axs[0, 0].imshow(rendered_img.cpu().detach().numpy(), cmap='gray')
            for i in range(split_sdfs.shape[0]):
                # flip the y-axis
                axs[(i+1)//2, (i+1)%2].invert_yaxis()
                plot_sdf(X, Y, split_sdfs[i].cpu().detach().numpy(), ax=axs[(i+1)//2, (i+1)%2])
            draw_contours(nX, nY, split_sdfs, opt_tfms, ax=axs[2, 0])
            axs[2, 0].invert_yaxis()
            plt.show()

optim_batch()

# %%
# optimize using sds loss
import tqdm
from torch.optim.lr_scheduler import LambdaLR
from ts_simple.df_guidance import DeepFloydGuidance, DeepFloydPromptProcessor

def render_splits(sdf, k=10, img_res=256):
    # [B, r, r]
    # scale the sdf to the image resolution (requires [B, 1, r, r])
    sdf_img_res = torch.nn.functional.interpolate(
        sdf.unsqueeze(1), size=(img_res, img_res), mode='bicubic', align_corners=False
    )

    # soft occupancy
    # 0 is inside (black), 1 is outside (white)
    img = 1 - torch.sigmoid(-k * sdf_img_res)

    # turn into 3 channels
    img = img.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)

    # [B, R, R, 3]
    return img.squeeze()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, -1)

def optim_sds():
    guidance = DeepFloydGuidance(device=DEVICE)
    prompt_processor = DeepFloydPromptProcessor(device=DEVICE)

    prompt = "a silhouette of a cat. trending on artstation."
    text_embeddings = prompt_processor.get_text_embeddings(prompt)
    prompt_processor.destroy_text_encoder()
    print(prompt)

    iters = 1000
    opt_sdf_b = torch.tensor(np.stack([C1, C2]), dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt_tfms_b = torch.tensor([[1, -0.5, 0], [1, 0.5, 0]], dtype=torch.float32, device=DEVICE, requires_grad=True)  # scale, pos_x, pos_y

    optimizer = torch.optim.Adam([opt_sdf_b, opt_tfms_b], lr=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(iters * 1.5))

    for it in tqdm.trange(iters):
        optimizer.zero_grad()
        # transform sdf to global scale
        g_sdf, split_sdfs = join_sdfs_torch(resolution, big_res, D, opt_sdf_b, opt_tfms_b)

        # render it
        rendered_img = render_sdf(g_sdf, k=50, img_res=IMG_RES)
        # make into 3 channels + 1 batch dimension
        rendered_img_3c = rendered_img.unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0).unsqueeze(0)

        # render the splits
        rendered_splits = render_splits(split_sdfs, k=50, img_res=IMG_RES)

        loss = 100 * guidance(rendered_img_3c, text_embeddings)['loss_sds']
        for split_img in rendered_splits:
            loss += 50 * guidance(split_img.unsqueeze(0), text_embeddings)['loss_sds']

        loss.backward()
        optimizer.step()
        scheduler.step()

        # redistance the sdf to keep it valid
        with torch.no_grad():
            for i in range(opt_sdf_b.shape[0]):
                opt_sdf_b[i].copy_(torch.tensor(redistance_sdf(opt_sdf_b[i].cpu().numpy()), device=DEVICE, dtype=torch.float32))
        
        if it % 100 == 0 or it == iters - 1:
            print(f"Iter {it}, Loss: {loss.item()}")
            fig, axs = plt.subplots(3, 2, figsize=(12, 12))
            axs[0, 0].set_title("Rendered Image")
            axs[0, 0].imshow(rendered_img.cpu().detach().numpy(), cmap='gray')
            for i in range(split_sdfs.shape[0]):
                # flip the y-axis
                axs[(i+1)//2, (i+1)%2].invert_yaxis()
                plot_sdf(X, Y, split_sdfs[i].cpu().detach().numpy(), ax=axs[(i+1)//2, (i+1)%2])
            draw_contours(nX, nY, split_sdfs, opt_tfms_b, ax=axs[2, 0])
            axs[2, 0].invert_yaxis()
            plt.show()

optim_sds()
# %%
from scipy.ndimage import label

def filter_areas(sdf):
    # compute connected components of the negative region
    neg_mask = sdf < 0
    labeled_array, num_features = label(neg_mask)
    if num_features <= 1:
        return sdf
    else:
        # find the largest component
        max_area = 0
        max_label = 0
        for i in range(1, num_features + 1):
            area = np.sum(labeled_array == i)
            if area > max_area:
                max_area = area
                max_label = i
        # create a mask for the largest component
        largest_component_mask = labeled_array == max_label
        # create a new sdf where only the largest component is negative
        new_sdf = sdf.copy()
        new_sdf[~largest_component_mask] = np.abs(new_sdf[~largest_component_mask])
        return new_sdf

resolution = 30

big_res = 128
D = 4
nX, nY = create_grid(big_res, domain=D)

sdfs = [C1, C2]
scales = [1, 0.5]
positions = [(-0.4, 0), (1, 0)]

G, l = join_sdfs_multiple(resolution, big_res, D, sdfs, scales, positions)
plot_sdf(nX, nY, G)
G_filtered = filter_areas(G)
plot_sdf(nX, nY, G_filtered)
# %%
