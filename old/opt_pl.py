import numpy as np
import pydiffvg
import matplotlib.pyplot as plt
import tqdm

import torch
from torch.optim.lr_scheduler import LambdaLR

from ts_simple.sd_guidance import StableDiffusionGuidance, StableDiffusionPromptProcessor
from ts_simple.df_guidance import DeepFloydGuidance, DeepFloydPromptProcessor

def circle(radius, N, device="cuda"):
    points = []
    avg_deg = 2 * np.pi / N
    for i in range(N):
        pt = (radius * np.cos(i * avg_deg), radius * np.sin(i * avg_deg))
        points.append(pt)
    ps = torch.tensor(points, device=device).float()
    ps.requires_grad = True
    return ps

def map_to_canvas(points, domain=[-1, 1], canvas_dims=256):
    # maps from domain^2 to [0, canvas_dims]^2
    points = (points - domain[0]) / (domain[1] - domain[0]) * canvas_dims
    return points

def render_curve(points, bg=None, closed=True, canvas_dims=256, seed=0):
    n_cps = torch.LongTensor([0] * points.shape[0])
    path = pydiffvg.Path(num_control_points=n_cps, points=points, is_closed=closed, stroke_width=torch.tensor(0.0))
    shapes = [path]
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
    shape_groups = [path_group]
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_dims, canvas_dims, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(canvas_dims, canvas_dims, 2, 2, seed, None, *scene_args)
    # RenderFunction.apply gives RGBA image
    # composite onto background
    if bg is None:
        bg = torch.ones_like(img[:, :, :3])
    img = img[:, :, :3] * img[:, :, 3:4] + bg * (1 - img[:, :, 3:4])
    return img

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, -1)

def pixel_loss(img, target):
    return (img - target).pow(2).sum()

def smoothness_loss(points):
    # compute p[i-1] + p[i+1] - 2 * p[i]
    return torch.sum((points.roll(1, 0) + points.roll(-1, 0) - 2 * points).pow(2))

def plot_curve(points, closed=True, canvas_dims=256):
    img = render_curve(points, closed=closed, canvas_dims=canvas_dims)
    img_np = img.cpu().detach().numpy()
    pts_np = points.cpu().detach().numpy()
    # plot the points 
    plt.figure(dpi=300)
    plt.xlim(0, canvas_dims)
    plt.ylim(0, canvas_dims)
    plt.imshow(img_np)
    plt.scatter(pts_np[:, 0], pts_np[:, 1], color="red", s=5)
    plt.plot(pts_np[:, 0], pts_np[:, 1], color="cyan", linewidth=1)
    plt.gca().invert_yaxis()

if __name__ == "__main__":
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    CANVAS_DIMS = 256
    ITERATIONS = 1000
    LR = 5e-3
    DEVICE = "cuda"
    DOMAIN = [-1, 1]

    SMOOTH_WT = 100
    # XING_WT = 10000
    # PX_WT = 100

    PROMPT = "a silhouette of a cat. trending on artstation"

    guidance = StableDiffusionGuidance(device=DEVICE)
    prompt_processor = StableDiffusionPromptProcessor(device=DEVICE)
    # guidance = DeepFloydGuidance(device=DEVICE)
    # prompt_processor = DeepFloydPromptProcessor(device=DEVICE)

    text_embeddings = prompt_processor.get_text_embeddings(PROMPT)
    prompt_processor.destroy_text_encoder()
    print(PROMPT)

    target_img = torch.tensor(plt.imread("target.png"), device=DEVICE)

    points = circle(0.8, 300, device=DEVICE)

    optimizer = torch.optim.AdamW([points], lr=LR, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(ITERATIONS * 1.5))
    pbar = tqdm.trange(ITERATIONS)

    for it in pbar:
        optimizer.zero_grad()

        points_mapped = map_to_canvas(points, canvas_dims=CANVAS_DIMS)
        img = render_curve(points_mapped, closed=True, canvas_dims=CANVAS_DIMS, seed=it + 1).unsqueeze(0)

        sds_l = guidance(img, text_embeddings)["loss_sds"]
        smooth_l = smoothness_loss(points)
        # px_l = pixel_loss(img, target_img)

        # pbar.set_postfix_str(f"Pixel loss: {px_l.item():.6f}")
        
        # points.grad = sds_g + XING_WT * xing_g
        # loss = sds_l + XING_WT * xing_l
        # loss = PX_WT * px_l + XING_WT * xing_l
        loss = sds_l + smooth_l * SMOOTH_WT
        loss.backward()
        # print(points.grad.norm(), sds_g.norm() + XING_WT * xing_g.norm())
        optimizer.step()
        scheduler.step()

        # limit points to domain
        with torch.no_grad():
            points.clamp_(min=DOMAIN[0], max=DOMAIN[1])

        if it % 50 == 0:
            with torch.no_grad():
                points_mapped = map_to_canvas(points, canvas_dims=CANVAS_DIMS)
                img = render_curve(points_mapped, closed=True, canvas_dims=CANVAS_DIMS, seed=it + 1)
                img_np = img.detach().cpu().numpy()
                plt.imsave(f"optpl/{it}.png", img_np)

    with torch.no_grad():
        points_mapped = map_to_canvas(points, canvas_dims=CANVAS_DIMS)
        
        img = render_curve(points_mapped, closed=True, canvas_dims=CANVAS_DIMS, seed=it + 1)
        img_np = img.detach().cpu().numpy()
        plt.imsave(f"optpl/final.png", img_np)

        plot_curve(points_mapped, closed=True, canvas_dims=CANVAS_DIMS)
        plt.savefig(f"optpl/plots.png")


