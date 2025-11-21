import os

import numpy as np
import matplotlib.pyplot as plt
import skfmm
import tqdm

from scipy.ndimage import label

import torch
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from helpers import read_image
from ts_simple.df_guidance import DeepFloydGuidance, DeepFloydPromptProcessor

# we parameterize the sdf using a grid of values
def create_grid(resolution, domain=1):
    x = np.linspace(-domain, domain, resolution)
    y = np.linspace(-domain, domain, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y

# initial circle sdf
def circle_sdf(X, Y, center=(0, 0), radius=0.5):
    return np.sqrt((X - center[0])**2 + (Y - center[1])**2) - radius

# redistance sdf using fast-marching method
def redistance_sdf(sdf, domain=1):
    dx = (2 * domain) / (sdf.shape[0] - 1)
    return skfmm.distance(sdf, dx=dx)

# filter to keep only the largest negative region in the sdf
def keep_largest_region(sdf):
    # compute connected components of the negative region
    neg_mask = sdf < 0
    labelled_array, num_features = label(neg_mask)
    if num_features <= 1:
        return sdf
    else:
        # find the largest component
        new_sdf = sdf.copy()
        max_area = 0
        max_label = 0
        for i in range(1, num_features + 1):
            area = np.sum(labelled_array == i)
            if area > max_area:
                max_area = area
                max_label = i
        # create a mask
        largest_region_mask = labelled_array == max_label
        new_sdf[largest_region_mask] = -1
        new_sdf[~largest_region_mask] = 1
        return new_sdf

# torch sdf functions
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

def smooth_max(a, b, alpha=5):
    # compute the element-wise smooth max between 2 tensors
    # sum(x_i * exp(alpha * x_i)) / sum(exp(alpha * x_i))
    # a, b: [...]
    exp_a = torch.exp(alpha * a)
    exp_b = torch.exp(alpha * b)
    smooth_max_val = (a * exp_a + b * exp_b) / (exp_a + exp_b)
    return smooth_max_val

def smooth_max_arr(arr, dim=0, alpha=5):
    # compute the element-wise smooth max over a tensor array across a given dimension
    exp_arr = torch.exp(alpha * arr)
    smooth_max_val = (arr * exp_arr).sum(dim=dim) / exp_arr.sum(dim=dim)
    return smooth_max_val

def smooth_min(a, b, alpha=5):
    return -smooth_max(-a, -b, alpha)

def smooth_min_arr(arr, dim=0, alpha=5):
    return -smooth_max_arr(-arr, dim, alpha)

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
    # splits = closest_splits(global_sdfs)
    splits = subtraction_splits(global_sdfs, S)
    # convert back to local coordinates
    local_sdfs = global_to_local_torch_batch(splits, transforms[:, 0], transforms[:, 1:3], r, D)
    return S, local_sdfs

# render sdf using soft occupancy (high k = sharper edges)
def render_sdf_batch(sdf, k=10, img_res=256):
    # [B, r, r]
    # scale the sdf to the image resolution (requires [B, 1, r, r])
    sdf_img_res = torch.nn.functional.interpolate(
        sdf.unsqueeze(1), size=(img_res, img_res), mode='bicubic', align_corners=False
    )

    # soft occupancy
    # 0 is inside (black), 1 is outside (white)
    img = 1 - torch.sigmoid(-k * sdf_img_res)

    # return a mono-channel image [B, H, W]
    return img.squeeze(1)

def blur_img(img, kernel=3, sigma=1.0):
    # img: [B, C, H, W]
    return torchvision.transforms.functional.gaussian_blur(img, kernel_size=kernel, sigma=sigma)

def nonempty_loss(sdfs, ratio=None, weights=None):
    # encourage `sdfs` to be have about be about `ratio` percent negative
    # i.e. this helps encourage the presence of some zero level-set
    # sdfs: [B, R, R]
    total_pixels = sdfs.shape[1] * sdfs.shape[2]
    neg_pixels = (sdfs < 0).sum(dim=(1, 2)).float()
    neg_ratio = neg_pixels / total_pixels
    if ratio is not None:
        loss = (torch.abs(neg_ratio - ratio)).sum()
    else:
        loss = 1 - neg_ratio
    if weights is not None:
        floss = (loss * weights).sum()
    else:
        floss = loss.sum()
    return floss, neg_ratio

def nonempty_loss_img(sdf_imgs, ratio=None, weights=None):
    # sdf_imgs: [B, H, W]
    # sum all the pixel values
    total_pixel_values = sdf_imgs.sum(dim=(1, 2))
    # get the percentage of whiteness
    white_ratio = total_pixel_values / (sdf_imgs.shape[1] * sdf_imgs.shape[2])
    # we want to decrease white ratio
    if ratio is not None:
        loss = (torch.abs(white_ratio - ratio)).sum()
    else:
        loss = white_ratio
    if weights is not None:
        floss = (loss * weights).sum()
    else:
        floss = loss.sum()
    # return loss and black ratio
    return floss, 1 - white_ratio

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, -1)

# helper functions for sdf visualization
def plot_sdf(X, Y, S, domain=1, cbar=False, ax=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    # plot the 0 contour
    cf = ax.contourf(X, Y, S, levels=100, cmap='RdBu', alpha=0.5, vmin=-1, vmax=1)
    if cbar:
        plt.colorbar(mappable=cf)
    ax.contour(X, Y, S, levels=[0], linewidths=0.5, colors="black")
    ax.set_xlim(-domain, domain)
    ax.set_ylim(-domain, domain)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    
def draw_contours(X, Y, global_sdfs, ax=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    for i in range(global_sdfs.shape[0]):
        ax.contour(X, Y, global_sdfs[i].cpu().detach().numpy(), levels=[0], colors=f"C{i}", linewidths=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

if __name__ == "__main__":
    FOLDER = "exp5"
    DEVICE = "cuda:1"
    PROMPT = "a silhouette of sea animals. trending on artstation."

    DIM_LOCAL = 30
    DIM_GLOBAL = 128
    IMG_RES = 256
    
    D_LOCAL = 1
    D_GLOBAL = 4

    ITERATIONS = 2000
    LR = 1e-2

    os.makedirs(FOLDER, exist_ok=True)

    guidance = DeepFloydGuidance(device=DEVICE)
    prompt_processor = DeepFloydPromptProcessor(device=DEVICE)

    text_embeddings = prompt_processor.get_text_embeddings(PROMPT)
    prompt_processor.destroy_text_encoder()
    print(PROMPT)

    # create sdf grids
    X, Y = create_grid(DIM_LOCAL, domain=D_LOCAL)
    nX, nY = create_grid(DIM_GLOBAL, domain=D_GLOBAL)

    # create initial sdfs
    C = circle_sdf(X, Y, center=(0, 0), radius=0.8)

    # initial sdfs and transforms: scale, pos_x, pos_y
    init_rad = 2
    init_pos = 1

    C1 = circle_sdf(X, Y, center=(-init_pos, 0), radius=init_rad)
    C2 = circle_sdf(X, Y, center=(init_pos, 0), radius=init_rad)
    # opt_sdfs = torch.tensor(np.stack([C1, C2,]), dtype=torch.float32, device=DEVICE, requires_grad=True)

    opt_sdfs = torch.tensor(np.stack([C, C,]), dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt_scales = torch.tensor([init_rad for _ in range(opt_sdfs.shape[0])], dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt_positions = torch.tensor([[-init_pos, 0], [init_pos, 0],], dtype=torch.float32, device=DEVICE, requires_grad=True)
    # opt_positions = torch.tensor([[-init_pos, 0], [init_pos, 0], [0, -init_pos], [0, init_pos]], dtype=torch.float32, device=DEVICE, requires_grad=True)
   
    optimizer = torch.optim.Adam([opt_sdfs, opt_scales, opt_positions], lr=LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(ITERATIONS * 1.5))
    
    # initial negative weight
    nw_low = 100
    nw_high = 1000
    neg_weights = torch.tensor([nw_low for _ in range(opt_sdfs.shape[0])], device=DEVICE)

    set_scale = 2 * torch.ones_like(opt_scales, dtype=torch.float32, device=DEVICE)
    set_pos = torch.zeros_like(opt_positions, dtype=torch.float32, device=DEVICE)

    N = opt_sdfs.shape[0]

    loss_history = {"S_sds_loss": [], "ind_sds_loss": [], "pos_loss": [], "S_ne_loss": [], "ind_ne_loss": []}
    grad_history = {"S_sds_grad": [], "ind_sds_grad": [], "pos_grad": [], "S_ne_grad": [], "ind_ne_grad": []}
    
    for it in tqdm.trange(ITERATIONS):
        optimizer.zero_grad()

        opt_tfms = torch.stack([set_scale, opt_positions[:, 0], opt_positions[:, 1]], dim=1)
        # opt_tfms = torch.stack([set_scale, set_pos[:, 0], set_pos[:, 1]], dim=1)
        
        S, split_sdfs = join_sdfs_torch(DIM_LOCAL, DIM_GLOBAL, D_GLOBAL, opt_sdfs, opt_tfms)

        # render sdfs, returns [B, H, W]
        overall_img = render_sdf_batch(S.unsqueeze(0), k=30, img_res=IMG_RES)
        indiv_imgs = render_sdf_batch(split_sdfs, k=30, img_res=IMG_RES)

        # compute guidance loss
        # guidance requires images in [B, H, W, 3] (rgb)
        overall_img_rgb = overall_img.unsqueeze(-1).repeat(1, 1, 1, 3)
        indiv_imgs_rgb = indiv_imgs.unsqueeze(-1).repeat(1, 1, 1, 3)

        loss = 0

        # S_sds_loss = guidance(overall_img_rgb, text_embeddings)["loss_sds"]
        # loss_history["S_sds_loss"].append(S_sds_loss.item())
        # loss += 50 * S_sds_loss

        ind_sds_loss = 0
        for img in indiv_imgs_rgb:
            ind_sds_loss += guidance(img.unsqueeze(0), text_embeddings)["loss_sds"]
        loss_history["ind_sds_loss"].append(ind_sds_loss.item())
        loss += 1 * ind_sds_loss

        # encourage positions to be close to 0, 0
        # pos_loss = torch.sum(torch.abs(opt_positions))
        # loss_history["pos_loss"].append(pos_loss.item())
        # loss += 10 * pos_loss
        
        # make sure that S is not too large too, white ratio should be about 0.6
        # nonempty_loss_img requires [B, H, W]
        S_ne_loss, S_ne_ratio = nonempty_loss_img(overall_img, ratio=None)
        loss += 10 * S_ne_loss
        loss_history["S_ne_loss"].append(S_ne_loss.item())
        
        # small nonempty loss to prevent sdfs from vanishing (becoming all +ve or all -ve)
        ind_ne_loss, ind_ne_ratio = nonempty_loss_img(indiv_imgs, ratio=None, weights=neg_weights)
        loss += ind_ne_loss
        loss_history["ind_ne_loss"].append(ind_ne_loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        # no-grad, post-processing steps
        with torch.no_grad():
            opt_scales.clamp_(0.1, D_GLOBAL / D_LOCAL)
            opt_positions.clamp_(-D_GLOBAL + D_LOCAL, D_GLOBAL - D_LOCAL)

            # S, local_sdfs = join_sdfs_torch(DIM_LOCAL, DIM_GLOBAL, D_GLOBAL, opt_sdfs, opt_tfms)
            # update neg_weights based on current indiv_ratio
            # the smaller the ratio is, the higher the weight should be (exponentially)
            neg_weights = nw_low + (nw_high - nw_low) * torch.exp(-0.5 * ind_ne_ratio)

            # post-process on the split sdfs (and copy them to opt_sdfs)
            for i in range(N):

                # set the edges to be positive
                sdf_i_np = opt_sdfs[i].cpu().numpy()

                # ensures that the zero level-set is entirely within the domain to avoid boundary artifacts in the global transform
                sdf_i_np[:1, :] = 1
                sdf_i_np[-1:, :] = 1
                sdf_i_np[:, :1] = 1
                sdf_i_np[:, -1:] = 1

                # filter away disconnected regions
                sdf_i_np = keep_largest_region(sdf_i_np)
                
                # redistance
                try:
                    sdf_i_np = redistance_sdf(sdf_i_np, domain=D_LOCAL)
                except Exception as e:
                    print("redistancing failed")
                    pass

                # plot_sdf(X, Y, sdf_i_np)
                # plt.savefig(os.path.join(FOLDER, "tmp", f"{i}/{it}.png"))
                # plt.close()

                # copy back to torch
                opt_sdfs[i].copy_(torch.tensor(sdf_i_np, device=DEVICE, dtype=torch.float32))
                # opt_sdfs[i].copy_(torch.tensor(redistance_sdf(opt_sdfs[i].cpu().numpy(), domain=D_LOCAL), device=DEVICE, dtype=torch.float32))

            _, split_sdfs = join_sdfs_torch(DIM_LOCAL, DIM_GLOBAL, D_GLOBAL, opt_sdfs, opt_tfms)

            if it % 50 == 0 or it == ITERATIONS - 1:
                # print(f"Iter {it}, Loss: {loss.item()}")
                fig, axs = plt.subplots(1 + opt_sdfs.shape[0], 2, figsize=(6, 3 * opt_sdfs.shape[0]), dpi=200)

                axs[0, 0].set_title("Rendered Image", fontsize=6)
                axs[0, 0].imshow(overall_img[0].cpu().detach().numpy(), cmap='gray')
                axs[0, 0].set_xticklabels([])
                axs[0, 0].set_yticklabels([])

                global_sdfs = local_to_global_torch_batch(split_sdfs, opt_tfms[:, 0], opt_tfms[:, 1:3], DIM_GLOBAL, D_GLOBAL)
                draw_contours(nX, nY, global_sdfs, ax=axs[0, 1])
                axs[0, 1].set_title("Contours", fontsize=6)
                axs[0, 1].invert_yaxis()

                for i in range(split_sdfs.shape[0]):
                    plot_sdf(X, Y, split_sdfs[i].cpu().detach().numpy(), domain=D_LOCAL, ax=axs[i+1, 0])
                    axs[i+1, 0].invert_yaxis()
                    axs[i+1, 0].set_title(f"split {i+1}", fontsize=6)
                    
                    plot_sdf(X, Y, opt_sdfs[i].cpu().detach().numpy(), domain=D_LOCAL, ax=axs[i+1, 1])
                    axs[i+1, 1].invert_yaxis()
                    axs[i+1, 1].set_title(f"full {i+1}", fontsize=6)
                
                plt.tight_layout()
                if it == ITERATIONS - 1:
                    plt.savefig(os.path.join(FOLDER, f"final.png"))
                else:
                    plt.savefig(os.path.join(FOLDER, f"{it}.png"))
                plt.close()

                # plot loss history
                fig, axs = plt.subplots(len(loss_history), 1, figsize=(4, 3 * len(loss_history)), dpi=200)
                for i, key in enumerate(loss_history):
                    axs[i].plot(loss_history[key], label=key, linewidth=0.5)
                    axs[i].set_yscale('log')
                    axs[i].set_xlabel("Iteration")
                    axs[i].set_ylabel("Loss")
                    axs[i].grid(True, which="both", ls="--", linewidth=0.3)
                    axs[i].set_title(f"{key}")
                plt.tight_layout()
                plt.savefig(os.path.join(FOLDER, f"loss_history.png"))
                plt.close()
    
