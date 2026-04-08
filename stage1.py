import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os, sys
import tomlkit
import random
import pickle
import textwrap

import torch
import torchvision
import torchvision.transforms.functional as tvf
from torchmetrics.multimodal import CLIPScore
import kornia as K

import jigsaw.fsm as fsm
import jigsaw.misc as misc
from jigsaw.ffn import LearnableImageFourier
from jigsaw.optim import sigmoid_scheduler, cosine_warmup_lmb
from jigsaw.sds.misc import get_guidance_and_text_embeds


def scatter(ax, foreground_src, background_src, fg_color='white', bg_color='black', inv_y=True, bg=True):
    fg_np = np.array(foreground_src)
    bg_np = np.array(background_src)
    if inv_y:
        fg_np[:, 1] = -fg_np[:, 1]
        bg_np[:, 1] = -bg_np[:, 1]
    ax.scatter(fg_np[:, 0], fg_np[:, 1], marker='x', color=fg_color, s=8)
    if bg:
        ax.scatter(bg_np[:, 0], bg_np[:, 1], marker='x', color=bg_color, s=8)


def get_bbox_centers(silhouettes, threshold=0.5):
    masks = (silhouettes < threshold).detach().cpu().numpy()
    rows = np.any(masks, axis=2)
    cols = np.any(masks, axis=1)

    ymin = rows.argmax(axis=1)
    ymax = rows.shape[1] - np.flip(rows, axis=1).argmax(axis=1)
    xmin = cols.argmax(axis=1)
    xmax = cols.shape[1] - np.flip(cols, axis=1).argmax(axis=1)

    # get the center and max dimension
    xcenter = (xmin + xmax) // 2
    ycenter = (ymin + ymax) // 2

    # return centers
    return torch.from_numpy(np.stack([xcenter, ycenter], axis=1)).float().to(silhouettes.device)
    
def extract_pieces(silhouettes, threshold=0.8, padding=20, dim=256):
    # given silhouettes of shape (N, H, W), extract the piece as a binary mask (< threshold)
    # extract the bounding box of the piece and pad it by the given amount
    # then, resize it to the given dimension
    
    # pad silhouettes with border of 1
    silhouettes_padded = torch.nn.functional.pad(silhouettes, (padding, padding, padding, padding), value=1)

    # find the bbox
    masks = (silhouettes_padded < threshold).detach().cpu().numpy()
    rows = np.any(masks, axis=2)
    cols = np.any(masks, axis=1)

    ymin = rows.argmax(axis=1)
    ymax = rows.shape[1] - np.flip(rows, axis=1).argmax(axis=1)
    xmin = cols.argmax(axis=1)
    xmax = cols.shape[1] - np.flip(cols, axis=1).argmax(axis=1)

    # add padding for each piece
    ymin = ymin - padding
    ymax = ymax + padding
    xmin = xmin - padding
    xmax = xmax + padding

    # get the max dimension
    pieces = []
    for i in range(silhouettes.shape[0]):
        # tensor slicing
        piece = silhouettes_padded[i, ymin[i]:ymax[i], xmin[i]:xmax[i]]
        # pad to make it square
        h, w = piece.shape
        if h > w:
            padl = padr = (h - w) // 2
            if (h - w) % 2 == 1:
                padr += 1
            piece = torch.nn.functional.pad(piece, (padl, padr, 0, 0), value=1)
        elif w > h:
            padu = padd = (w - h) // 2
            if (w - h) % 2 == 1:
                padd += 1
            piece = torch.nn.functional.pad(piece, (0, 0, padu, padd), value=1)

        # resize to dim x dim
        piece = torch.nn.functional.interpolate(piece.unsqueeze(0).unsqueeze(0), size=(dim, dim), mode="nearest").squeeze(0).squeeze(0) # dim x dim
        pieces.append(piece)
    return torch.stack(pieces, dim=0)


def plot_img(ax, img, title, **kwargs):
    ax.axis("on")
    ax.imshow(img, extent=(-1, 1, -1, 1), **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=6)

def plot_img_with_scatter(ax, img, title, fg_src=None, bg_src=None, fg_col="white", bg_col="black", inv_y=True, bg=True, **kwargs):
    plot_img(ax, img, title, **kwargs)
    scatter(ax, fg_src, bg_src, fg_col, bg_col, inv_y=inv_y, bg=bg)

def plot_losses(ax, losses):
    # losses is a list of dicts, where losses[it] = {key: value}
    loss_names = [key for key in losses[0].keys()]
    for loss_name in loss_names:
        ax.plot([loss[loss_name] for loss in losses], label=loss_name)
    ax.axis("on")
    ax.set_xlabel("Iteration", fontsize=6)
    ax.set_ylabel("Loss", fontsize=6)
    ax.set_title("Losses Over Time", fontsize=6)
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.tick_params(axis="both", which="minor", labelsize=5)
    ax.legend(loc="upper left", fontsize=6)


def pipeline(config: tomlkit.TOMLDocument):
    seed = config["general"].get("seed", -1)
    if seed >= 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = config["general"].get("device", "cuda")

    dim = config["general"].get("dim", 512)
    pieces = config["general"].get("pieces", -1)
    if pieces < 1:
        print("Error: pieces must be specified and greater than 0!")
        return

    # misc config vars
    src_sample_mode = config["misc"].get("src_sample_mode", "fib_disk")
    piece_domain = config["misc"].get("piece_domain", 0.7)

    bg_border = config["misc"].get("bg_border", 5)
    bg_position = config["misc"].get("bg_position", 0.95)
    smax = config["misc"].get("smax", 7.1)
    smin = config["misc"].get("smin", 0.1)

    min_render_beta = config["misc"].get("min_render_beta", 10)
    max_render_beta = config["misc"].get("max_render_beta", 100)
    beta_warmup_steps = config["misc"].get("beta_warmup_steps", 2000)

    iters = config["optimization"].get("iters", 5000)
    lr_warmup_steps = config["optimization"].get("lr_warmup", 2000)
    lr = config["optimization"].get("lr", 5e-4)
    lr_src = config["optimization"].get("lr_src", 1e-4)
    lr_angle = config["optimization"].get("lr_angle", 1e-4)

    # weights for loss terms
    global_img_wt = config["weights"].get("global_img_wt", 10)
    indiv_img_wt = config["weights"].get("indiv_img_wt", 20)
    size_wt = config["weights"].get("size_wt", 300)
    smoothness_wt = config["weights"].get("smoothness_wt", 50)
    reg_wt = config["weights"].get("reg_wt", 10)

    reg_radius = config["weights"].get("reg_radius", 0.1)

    # create output folders
    out_folder = config["folder"].get("out_folder", "experiments")
    save_folder = config["folder"].get("name", "base")
    os.makedirs(os.path.join(out_folder, save_folder), exist_ok=True)
    os.makedirs(os.path.join(out_folder, save_folder, "iters"), exist_ok=True)

    # copy the config file to the output folder for reference
    with open(os.path.join(out_folder, save_folder, "config.toml"), "w") as f:
        f.write(tomlkit.dumps(config))

    model_type = config["models"].get("silhouette_model_type", "L")
    cfg = config["models"].get("silhouette_cfg", 10)
    use_saved_embeds = config["models"].get("use_saved_embeds", True)

    # deal with the prompts
    raw_prompt = config["prompts"].get("prompt", None)
    raw_overall_prompt = config["prompts"].get("overall_prompt", None)
    add_prompt = f'{config["prompts"].get("add_prompt", "")}'
    neg_prompt = f'{config["prompts"].get("neg_prompt", "")}'


    # if prompt is None:
    #     print("Error: No prompt provided in config!")
    #     return
    if isinstance(raw_prompt, str):
        prompts = [f"{raw_prompt}. {add_prompt}"] * pieces
    elif isinstance(raw_prompt, list):
        none_prompts = pieces - len(raw_prompt)
        # if len(prompt) != pieces:
            # print("Error: Length of prompt list must match number of pieces!")
            # return
        prompts = [f"{p}. {add_prompt}" for p in raw_prompt] + [None] * none_prompts
    else:
        print("Error: Prompt must be either a string or a list of strings!")
        return

    if raw_overall_prompt is not None:
        oprompts = f"{raw_overall_prompt}. {add_prompt}"
    else:
        oprompts = None

    # shuffle the prompts
    random.shuffle(prompts)

    prompted_indices = [i for i in range(len(prompts)) if prompts[i] is not None]

    # save dictionary
    save_prompts = {"prompt": prompts, "overall_prompt": oprompts}
    with open(os.path.join(out_folder, save_folder, "prompts.pkl"), "wb") as f:
        pickle.dump(save_prompts, f)


    # get sds stuff
    guidance, text_embeddings = get_guidance_and_text_embeds(model_type, prompts + [oprompts], guidance_scale=cfg, device=device, use_saved=use_saved_embeds, save_path="text_embeds", neg_prompt=neg_prompt)

    # split the embeddings into piece-wise and overall
    piece_embeddings = text_embeddings[:-1] # (pieces, 2, D, E)
    overall_embeddings = text_embeddings[-1] # (2, D, E)

    # generate PIECES src positions in [-1, 1]^2
    if src_sample_mode == "fib_disk":
        foreground_samples = misc.fibonacci_lattice(pieces, "disk")
    elif src_sample_mode == "fib_square":
        foreground_samples = misc.fibonacci_lattice(pieces, "square")
    elif src_sample_mode == "poisson_disk":
        foreground_samples = misc.poisson_disk_sampling(pieces, r=0.3, seed=seed)
    elif src_sample_mode == "grid":
        foreground_samples = misc.grid(pieces)
    else:
        raise ValueError(f"Unknown sample method: {src_sample_mode}")
    # scale down to [-PIECE_DOMAIN, PIECE_DOMAIN]^2
    foreground_src = foreground_samples * piece_domain

    # initialize background sources
    background_src = [[bg_position, bg_position], [-bg_position, -bg_position], [bg_position, -bg_position], [-bg_position, bg_position], [0.0, bg_position], [0.0, -bg_position], [bg_position, 0.0], [-bg_position, 0.0]]
    # background_src = [[BG_POSITION, BG_POSITION], [-BG_POSITION, -BG_POSITION], [BG_POSITION, -BG_POSITION], [-BG_POSITION, BG_POSITION]]
    # background_src = [[0.0, BG_POSITION], [0.0, -BG_POSITION], [BG_POSITION, 0.0], [-BG_POSITION, 0.0]]

    # form all the source points
    foregrounds_tensor = torch.from_numpy(foreground_src).float().to(device).requires_grad_(True)
    backgrounds_tensor = torch.tensor(background_src, device=device).float()
    foregrounds = list(range(pieces))

    # initialize the slowness map
    f = LearnableImageFourier(dim, dim, channels=1).to(device)

    # rotation angles
    rotation_angles = torch.from_numpy(np.random.uniform(0, 360, size=pieces)).float().to(device)

    # create optimizer for silhouette
    opt_params = [
        {'params': f.parameters(), 'lr': lr},
        {'params': [foregrounds_tensor], 'lr': lr_src},
        {'params': [rotation_angles], 'lr': lr_angle}
    ]

    lr_lambda = [
        lambda it: cosine_warmup_lmb(it, warmup=lr_warmup_steps, total=int(1.5 * iters)),
        lambda it: cosine_warmup_lmb(it, warmup=lr_warmup_steps, total=int(1.5 * iters)),
        lambda it: cosine_warmup_lmb(it, warmup=lr_warmup_steps, total=int(1.5 * iters)),
    ]

    optim = torch.optim.AdamW(opt_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    print()
    print()
    print("====================================================")
    print("Starting:", os.path.join(out_folder, save_folder))
    print("Overall prompt:", raw_overall_prompt)
    print("Piecewise prompts:", raw_prompt)
    print("Additional prompts:", add_prompt)
    print("Negative prompts:", neg_prompt)
    print("====================================================")
    print()

    losses = []

    # main loop
    for it in tqdm.trange(iters):
        # keep track of losses to be backpropped
        backprop_losses = []

        # "hardness" of silhouette
        render_beta = sigmoid_scheduler(min_render_beta, max_render_beta, it, beta_warmup_steps, iters, k=10)
        indiv_wt = sigmoid_scheduler(indiv_img_wt, indiv_img_wt * 2, it, 500, iters, k=10)

        optim.zero_grad()

        # construct the seed points tensor
        srcs = torch.cat([foregrounds_tensor, backgrounds_tensor], dim=0)

        # get the slowness map
        f_tex = f().squeeze()
        slowness = (smax - smin) * f_tex + smin
        # set the border to be low slowness
        slowness[:bg_border, :] = smin
        slowness[-bg_border:, :] = smin
        slowness[:, :bg_border] = smin
        slowness[:, -bg_border:] = smin

        fn_iters = 4 * dim
        T = fsm.FSMGpuFn.apply(srcs, slowness, fn_iters)

        # apply softmax with render_beta [pieces + srcs, DIM, DIM] 
        soft_T = fsm.soft_voronoi(T, beta=render_beta)

        # combined image
        img_bw = 1 - fsm.rasterize_T(soft_T, foregrounds) # DIM x DIM
        img_rgb = img_bw.unsqueeze(2).repeat(1, 1, 3).unsqueeze(0) # 1 x DIM x DIM x 3
        img_rgb = torch.clamp(img_rgb, 0, 1)

        if oprompts is not None:
            global_img_loss = guidance(img_rgb, overall_embeddings)["loss_sds"]
            backprop_losses.append((global_img_wt, global_img_loss))

        # individual images
        img_indiv_bw = 1 - fsm.rasterize_T_index(soft_T, foregrounds) # N x DIM x DIM, black on white
        img_indiv_rgb = img_indiv_bw.unsqueeze(3).repeat(1, 1, 1, 3) # N x DIM x DIM x 3

        # rotate (to fix: rotated pieces can be cropped, figure out how to deal with this)
        # get the centers of the square bbox of each piece
        # centers = get_bbox_centers(img_indiv_bw)
        # img_indiv_rgb_rot = 1 - K.geometry.transform.rotate((1 - img_indiv_rgb.permute(0, 3, 1, 2)), angle=rotation_angles, center=centers).permute(0, 2, 3, 1)

        # extract piece into squares and rotate
        extracted_pieces = extract_pieces(img_indiv_bw, padding=50).unsqueeze(3).repeat(1, 1, 1, 3)
        img_indiv_rgb_rot = 1 - K.geometry.transform.rotate((1 - extracted_pieces.permute(0, 3, 1, 2)), angle=rotation_angles).permute(0, 2, 3, 1)
        
        img_indiv_rgb_rot = torch.clamp(img_indiv_rgb_rot, 0, 1)

        if len(prompted_indices) > 0:
            indiv_img_loss = guidance(img_indiv_rgb_rot[foregrounds][prompted_indices], piece_embeddings)["loss_sds"]
            backprop_losses.append((indiv_wt, indiv_img_loss))

        # size loss (prevent collapse)
        indiv_pct = torch.mean(1 - img_indiv_bw, dim=[1, 2])
        # size_loss = (indiv_pct - avg_area).square().mean()
        area_balance = (indiv_pct * torch.log(indiv_pct * pieces + 1e-8)).sum()
        pixel_entropy = -(img_indiv_bw * torch.log(img_indiv_bw + 1e-8)).sum(dim=0).mean()
        size_loss = area_balance + 0.05 * pixel_entropy
        backprop_losses.append((size_wt, size_loss))

        # smooth out the slowness map (squared laplacian)
        data_padded = torch.nn.functional.pad(slowness, (1, 1, 1, 1), value=smin)
        lap = data_padded[:-2, 1:-1] + data_padded[2:, 1:-1] + data_padded[1:-1, :-2] + data_padded[1:-1, 2:] - 4 * data_padded[1:-1, 1:-1]
        smoothness_loss = torch.mean(lap * lap)
        backprop_losses.append((smoothness_wt, smoothness_loss))

        # add a regularization loss on the slowness texture
        avg_area = torch.mean(indiv_pct).detach()
        reg_loss = 0
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, dim, device=device), torch.linspace(-1, 1, dim, device=device), indexing="ij")
        for i in range(pieces):
            area = indiv_pct[i]
            # if the current area is too small, add a regularization directly on the texture map
            # if area < avg_area * (1 - REG_RANGE):
            # minimize the slowness around the piece (weighted using a gaussian distribution)
            gaussian = torch.exp(-((xx - foregrounds_tensor[i, 0]) ** 2 + (yy - foregrounds_tensor[i, 1]) ** 2) / (2 * (reg_radius ** 2))).clamp(0, 1).detach()
            # plt.imsave("tpng.png", (gaussian > 0.2).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            # return
            # regularization: mse between gaussian (mapped to slowness) and actual slowness value
            # reg_loss += 0.05 * (((SMIN + (SMAX - SMIN) * gaussian) - slowness) ** 2).mean()
            # regularization: directly minimize the slowness weighted by the gaussian
            # add an additional weight that increases as the area gets smaller
            reg_loss += (slowness * (gaussian > 0.1).float()).mean() / (area + 1e-8)

        # minimize the slowness in general
        reg_loss += 0.1 * torch.mean(slowness)
        backprop_losses.append((reg_wt, reg_loss))

        # combine losses
        loss = sum(wt * l for wt, l in backprop_losses)

        losses.append({
            # "global_img_loss": global_img_loss.item(),
            # "indiv_img_loss": indiv_img_loss.item(),
            # "area_bal": area_balance.item(),
            # "px_entropy": pixel_entropy.item(),
            # "size_loss": size_loss.item(),
            "reg_loss": reg_loss.item(),
            # "smoothness_loss": smoothness_loss.item(),
            # "total_loss": loss.item()
        })

        loss.backward()
        optim.step()
        scheduler.step()

        with torch.no_grad():
            foregrounds_tensor.clamp_(-piece_domain, piece_domain)

            if it % 100 == 0 or it == iters - 1:
                hard_T = fsm.hard_voronoi(T)

                layout = [
                    ["slowness", "image", "soft_voronoi", "hard_voronoi", "losses"]
                ]
                for i in range(pieces):
                    layout.append([f"piece_{i}_rot", f"piece_{i}", f"piece_{i}_hard", f"piece_{i}_prompt", f"texture_{i}"])
                fig, ax = plt.subplot_mosaic(layout, figsize=(3 * len(layout[0]), 3 * len(layout)), dpi=150)
                
                for a in ax.values():
                    a.axis("off")

                fg_src = foregrounds_tensor.detach().cpu().numpy()
                bg_src = backgrounds_tensor.detach().cpu().numpy()

                plot_img_with_scatter(ax["slowness"], slowness.detach().cpu().numpy(), f"Slowness (it={it})", fg_src, bg_src, fg_col="green", bg_col="black", cmap="coolwarm", vmin=smin, vmax=smax)
                plot_img_with_scatter(ax["image"], img_rgb.detach().cpu().numpy()[0], f"Image (it={it})", fg_src, bg_src, fg_col="green", bg_col="black")
                plot_img_with_scatter(ax["soft_voronoi"], fsm.render_soft_voronoi(soft_T), f"Soft Voronoi (it={it}, beta={render_beta:.2f})", fg_src, bg_src, fg_col="white", bg_col="black")
                plot_img_with_scatter(ax["hard_voronoi"], hard_T, f"Hard Voronoi (it={it})", fg_src, bg_src, fg_col="white", bg_col="black")

                plot_losses(ax["losses"], losses)

                for i in range(pieces):
                    plot_img(ax[f"piece_{i}_rot"], img_indiv_rgb_rot[i].detach().cpu().numpy(), f"Soft foreground {i} (rotated) (it={it})")
                    plot_img(ax[f"piece_{i}"], img_indiv_rgb[i].detach().cpu().numpy(), f"Soft foreground {i} (it={it})")
                    plot_img(ax[f"piece_{i}_hard"], hard_T != i, f"Hard foreground {i} (it={it})", cmap="gray")
                    ax[f"piece_{i}_prompt"].text(0.5, 0.5, textwrap.fill(prompts[i], width=30), fontsize=12, ha="center", va="center", wrap=True)

                fig.colorbar(ax["slowness"].images[0], ax=ax["slowness"], fraction=0.046, pad=0.04)
                plt.tight_layout()
                
                # save a (overwriting) file so we can check progress
                fig.savefig(os.path.join(out_folder, save_folder, "final.png"))

                if it == 0:
                    fig.savefig(os.path.join(out_folder, save_folder, "init.png"))
                else:
                    fig.savefig(os.path.join(out_folder, save_folder, f"iters/{it}.png"))
                plt.close(fig)

                # save as "checkpoints"
                hard_T_bg = hard_T.copy()
                for i in range(len(background_src)):
                    hard_T_bg[hard_T == pieces + i] = -1
                np.save(os.path.join(out_folder, save_folder, f"pieces.npy"), hard_T_bg)
                # also save the rotation angles
                np.save(os.path.join(out_folder, save_folder, f"rotation_angles.npy"), rotation_angles.detach().cpu().numpy())


    print()
    print("====================================================")
    print("Done:", os.path.join(out_folder, save_folder))
    print("====================================================")
    print()

    # extract the final textures and silhouettes
    with torch.inference_mode():
        # get the slowness map
        f_tex = f().squeeze()
        slowness = (smax - smin) * f_tex + smin

        # set the border to be low slowness
        slowness[:bg_border, :] = smin
        slowness[-bg_border:, :] = smin
        slowness[:, :bg_border] = smin
        slowness[:, -bg_border:] = smin

        fn_iters = 4 * dim
        T = fsm.FSMGpuFn.apply(srcs, slowness, fn_iters)
        hard_T = fsm.hard_voronoi(T)
        # make the background pieces -1
        hard_T_bg = hard_T.copy()
        for i in range(len(background_src)):
            hard_T_bg[hard_T == pieces + i] = -1
        np.save(os.path.join(out_folder, save_folder, f"pieces.npy"), hard_T_bg)

        # also save the rotation angles
        np.save(os.path.join(out_folder, save_folder, f"rotation_angles.npy"), rotation_angles.detach().cpu().numpy())


if __name__ == "__main__":
    base_config = "configs/base.toml"
    with open(base_config, "rb") as f:
        base = tomlkit.load(f)

    configfile = "configs/base.toml" if len(sys.argv) == 1 else sys.argv[1]

    print("Loading config from", configfile)
    with open(configfile, "rb") as f:
        config = tomlkit.load(f)

    # update the base config with the new config
    for section in config:
        if section not in base:
            base[section] = config[section]
        else:
            for key in config[section]:
                base[section][key] = config[section][key]
    
    pipeline(base)