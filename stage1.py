import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os, sys
import tomlkit
import random
import pickle
import textwrap
import argparse
from pathlib import Path

import torch
import kornia as K

import jigsaw.fsm as fsm
import jigsaw.samplers as samplers
from jigsaw.ffn import LearnableImageFourier, LearnableImageFourierNoFixed, get_uv_grid
from jigsaw.optim import sigmoid_scheduler, cosine_warmup_lmb
from jigsaw.sds.misc import get_guidance_and_text_embeds
from jigsaw.helpers import unique_name


def scatter(ax, foreground_src, background_src, fg_color='white', bg_color='black', inv_y=True, bg=True):
    fg_np = np.array(foreground_src)
    bg_np = np.array(background_src)
    if inv_y:
        fg_np[:, 1] = -fg_np[:, 1]
        bg_np[:, 1] = -bg_np[:, 1]
    ax.scatter(fg_np[:, 0], fg_np[:, 1], marker='x', color=fg_color, s=8)
    if bg:
        ax.scatter(bg_np[:, 0], bg_np[:, 1], marker='x', color=bg_color, s=8)

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


def pipeline(config: tomlkit.TOMLDocument):
    seed = config["general"].get("seed", -1)
    if seed < 0:
        seed = random.randint(0, 1000000)
        config["general"]["seed"] = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config["general"].get("device", "cuda")

    dim = config["general"].get("dim", 512)
    bg_border = config["misc"].get("bg_border", 8)

    pieces = config["general"].get("pieces", -1)
    if pieces < 1:
        print("Error: pieces must be specified and greater than 0!")
        return

    # misc config vars
    src_sample_mode = config["misc"].get("src_sample_mode", "fib_disk")
    piece_domain = config["misc"].get("piece_domain", 0.7)

    bg_position = config["misc"].get("bg_position", 0.97)
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

    # get the schedule for the image dimensions and border
    # img_schedule = config["optimization"].get("img_schedule", None)
    # if img_schedule is None:
    #     img_schedule = [[0, dim, bg_border]]
    # else:
    #     img_schedule = sorted(img_schedule, key=lambda x: x[0])

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
    raw_prompt = config["prompts"].get("prompt", "")
    raw_overall_prompt = config["prompts"].get("overall_prompt", "")
    add_prompt = f'{config["prompts"].get("add_prompt", "")}'
    neg_prompt = f'{config["prompts"].get("neg_prompt", "")}'

    if isinstance(raw_prompt, str):
        if raw_prompt != "":
            prompts = [f"{raw_prompt}. {add_prompt}"] * pieces
        else:
            prompts = [None] * pieces
        raw_prompt = [raw_prompt]
    elif isinstance(raw_prompt, list):
        prompts = [f"{p}. {add_prompt}" for p in raw_prompt if p]
        none_prompts = pieces - len(prompts)
        prompts += [None] * none_prompts
    else:
        print("Error: Prompt must be either a string or a list of strings!")
        return

    if raw_overall_prompt != "":
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
    guidance, piece_embeddings, overall_embeddings = get_guidance_and_text_embeds(model_type, prompts, oprompts, guidance_scale=cfg, device=device, use_saved=use_saved_embeds, save_path="text_embeds", neg_prompt=neg_prompt)

    # piece_embeddings should have size (non-none pieces, 2, D, E)
    # overall_embeddings should have size (2, D, E)
    assert piece_embeddings.shape[0] == len(prompted_indices), "Error, number of prompted pieces and text embeds don't match"

    # generate PIECES src positions in [-1, 1]^2
    if src_sample_mode == "fib_disk":
        foreground_samples = samplers.fibonacci_lattice(pieces, "disk")
    elif src_sample_mode == "fib_square":
        foreground_samples = samplers.fibonacci_lattice(pieces, "square")
    elif src_sample_mode == "poisson_disk":
        foreground_samples = samplers.poisson_disk_sampling(pieces, r=0.3, seed=seed)
    elif src_sample_mode == "grid":
        foreground_samples = samplers.grid(pieces)
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
    # f = LearnableImageFourier(dim, dim, channels=1).to(device)
    f = LearnableImageFourierNoFixed(channels=1).to(device)

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

    optim = torch.optim.AdamW(opt_params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    print()
    print()
    print("====================================================")
    print(f"Starting: {os.path.join(out_folder, save_folder)} - ({pieces} pieces)")
    print("\tOverall prompt:", raw_overall_prompt)
    print("\tPiecewise prompts:", [p for p in raw_prompt if p])
    print("\tAdditional prompts:", add_prompt)
    print("\tNegative prompts:", neg_prompt)
    print("====================================================")
    print()

    uv_grid = get_uv_grid(dim, dim).to(device)

    losses = []
    # main loop
    for it in tqdm.trange(iters):
        # keep track of losses to be backpropped
        backprop_losses = []

        # "hardness" of silhouette
        render_beta = sigmoid_scheduler(min_render_beta, max_render_beta, it, beta_warmup_steps, iters, k=10)
        indiv_wt = sigmoid_scheduler(indiv_img_wt, indiv_img_wt * 1.5, it, 500, iters, k=10)

        optim.zero_grad()

        # construct the seed points tensor
        srcs = torch.cat([foregrounds_tensor, backgrounds_tensor], dim=0)

        # get the slowness map
        f_tex = f(uv_grid).squeeze()
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

        # overall image
        img_bw = 1 - fsm.rasterize_T(soft_T, foregrounds) # DIM x DIM
        img_rgb = img_bw.unsqueeze(2).repeat(1, 1, 3).unsqueeze(0) # 1 x DIM x DIM x 3
        img_rgb = torch.clamp(img_rgb, 0, 1)

        if overall_embeddings is not None:
            overall_img_loss = guidance(img_rgb, overall_embeddings)["loss_sds"]
            backprop_losses.append((global_img_wt, overall_img_loss))

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
            area = indiv_pct[i].detach()
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
            reg_loss += (slowness * gaussian).mean() / (area + 1e-8)

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

                    text_lbl = prompts[i] if prompts[i] is not None else ""
                    ax[f"piece_{i}_prompt"].text(0.5, 0.5, textwrap.fill(text_lbl, width=30), fontsize=12, ha="center", va="center", wrap=True)

                fig.colorbar(ax["slowness"].images[0], ax=ax["slowness"], fraction=0.046, pad=0.04)
                plt.tight_layout()
                
                # save a (overwriting) file so we can check progress
                fig.savefig(os.path.join(out_folder, save_folder, "final.png"))

                if it == 0:
                    fig.savefig(os.path.join(out_folder, save_folder, "init.png"))
                elif it < iters - 1:
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
        f_tex = f(uv_grid).squeeze()
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

        # save the model
        torch.save(f.state_dict(), os.path.join(out_folder, save_folder, "model_weights.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate jigsaw puzzles from a text prompt.")
    parser.add_argument("-b", "--base_config", type=str, default="configs/base.toml", help="Path to the base config file (TOML format). This will be used as the default config and updated with the provided config.")
    parser.add_argument("-c", "--config", type=str, help="Path to the config file (TOML format).")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility. This will override the seed provided in the config file (if any).")
    parser.add_argument("--pieces", type=int, help="Number of pieces to generate. This will override the pieces provided in the config file (if any).")
    parser.add_argument("--prompts", type=str, help="Text prompt for both the pieces and the overall image. This will override ALL prompts provided in the config file (if any).")
    parser.add_argument("--overall_prompt", type=str, help="Text prompt for the overall image. This will override the overall prompt provided in the config file (if any).")
    parser.add_argument("--name", type=str, help="Name of the output folder. This will override the name provided in the config file (if any).")

    args = parser.parse_args()

    base_config = args.base_config
    # base_config = "configs/base.toml"
    with open(base_config, "rb") as f:
        base = tomlkit.load(f)

    configfile = args.config if args.config else None

    if configfile is not None:
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

    # override prompt if provided
    if args.prompts is not None:
        base["prompts"]["prompt"] = args.prompts
        base["prompts"]["overall_prompt"] = args.prompts

        # override the name of the output folder
        tname = args.prompts.replace(" ", "_")[:50]
        # make it unique
        tname = unique_name(base["folder"]["out_folder"], tname)
        base["folder"]["name"] = tname

    # override overall prompt if provided
    if args.overall_prompt is not None:
        base["prompts"]["overall_prompt"] = args.overall_prompt

    # override name if provided
    if args.name is not None:
        base["folder"]["name"] = args.name

    # override pieces if provided
    if args.pieces is not None:
        base["general"]["pieces"] = args.pieces
    
    # override seed if provided
    if args.seed is not None:
        base["general"]["seed"] = args.seed

    pipeline(base)