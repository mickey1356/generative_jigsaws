import numpy as np
import pydiffvg
import matplotlib.pyplot as plt
import tqdm

import torch
from torch.optim.lr_scheduler import LambdaLR

from ts_simple.sd_guidance import StableDiffusionGuidance, StableDiffusionPromptProcessor
from ts_simple.df_guidance import DeepFloydGuidance, DeepFloydPromptProcessor

# returns torch tensor of shape (3N, 2) in the domain [-1, 1]
def bezier_circle(radius, N, cp_dist_ratio=0.2, device="cuda"):
    points = []
    cp_dist = radius * cp_dist_ratio
    avg_deg = 2 * np.pi / N
    for i in range(N):
        pt = (radius * np.cos(i * avg_deg), radius * np.sin(i * avg_deg))
        # the control points will be perpendicular to the radius
        cp1 = (pt[0] + cp_dist * np.cos(i * avg_deg + np.pi / 2), pt[1] + cp_dist * np.sin(i * avg_deg + np.pi / 2))
        cp2 = (pt[0] + cp_dist * np.cos(i * avg_deg - np.pi / 2), pt[1] + cp_dist * np.sin(i * avg_deg - np.pi / 2))
        points.append(cp2)
        points.append(pt)
        points.append(cp1)
    ps = torch.tensor(points[1:] + points[:1], device=device).float()
    ps.requires_grad = True
    return ps

def map_to_canvas(points, domain=[-1, 1], canvas_dims=256):
    # maps from domain^2 to [0, canvas_dims]^2
    points = (points - domain[0]) / (domain[1] - domain[0]) * canvas_dims
    return points

def compute_sine_theta(s1, s2):
    v1 = s1[1, :] - s1[0, :]
    v2 = s2[1, :] - s2[0, :]
    sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2))
    return sine_theta

# takes as input (3N, 2) tensor
def xing_loss(x):
    seg_loss = 0
    N = x.size(0)
    x = torch.cat([x, x[0, :].unsqueeze(0)], dim=0) # close loop
    segments = torch.cat([x[:-1, :].unsqueeze(1), x[1:, :].unsqueeze(1)], dim=1) # (N, start/end, 2)
    assert N % 3 == 0, "num segments is not correct"
    num_segs = N // 3
    for i in range(num_segs):
        cs1 = segments[i * 3, :, :]
        cs2 = segments[i * 3 + 1, :, :]
        cs3 = segments[i * 3 + 2, :, :]
        direct = (compute_sine_theta(cs1, cs2) >= 0).float()
        opst = 1 - direct
        sine_a = compute_sine_theta(cs1, cs3)
        seg_loss += direct * torch.relu(-sine_a) + opst * torch.relu(sine_a)
    return seg_loss / num_segs

def render_curve(points, bg=None, closed=True, canvas_dims=256, seed=0):
    n_cps = torch.LongTensor([2] * (points.shape[0] // 3))
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

def plot_curve(points, closed=True, canvas_dims=256):
    img = render_curve(points, closed=closed, canvas_dims=canvas_dims)
    img_np = img.cpu().detach().numpy()
    pts_np = points.cpu().detach().numpy()
    # plot the points 
    plt.figure(figsize=(8, 8))
    plt.xlim(0, canvas_dims)
    plt.ylim(0, canvas_dims)
    plt.imshow(img_np)
    plt.scatter(pts_np[::3, 0], pts_np[::3, 1], color='red')
    plt.scatter(pts_np[1::3, 0], pts_np[1::3, 1], color='blue', marker="+")
    plt.scatter(pts_np[2::3, 0], pts_np[2::3, 1], color='blue', marker="x")
    plt.gca().invert_yaxis()

if __name__ == "__main__":
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    CANVAS_DIMS = 256
    ITERATIONS = 1000
    LR = 1e-2
    DEVICE = "cuda"
    XING_WT = 10000
    # PX_WT = 100
    DOMAIN = [-1, 1]

    PROMPT = "a silhouette of a cat. trending on artstation"

    target_img = torch.tensor(plt.imread("target.png"), device=DEVICE)

    # setup guidance
    guidance = StableDiffusionGuidance(device=DEVICE)
    prompt_processor = StableDiffusionPromptProcessor(device=DEVICE)
    # guidance = DeepFloydGuidance(device=DEVICE)
    # prompt_processor = DeepFloydPromptProcessor(device=DEVICE)

    text_embeddings = prompt_processor.get_text_embeddings(PROMPT)
    prompt_processor.destroy_text_encoder()
    print(PROMPT)

    points = bezier_circle(0.8, 64, cp_dist_ratio=0.1, device=DEVICE)

    optimizer = torch.optim.AdamW([points], lr=LR, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(ITERATIONS * 1.5))

    pbar = tqdm.trange(ITERATIONS)

    for it in pbar:
        optimizer.zero_grad()

        points_mapped = map_to_canvas(points, canvas_dims=CANVAS_DIMS)
        img = render_curve(points_mapped, closed=True, canvas_dims=CANVAS_DIMS, seed=it + 1).unsqueeze(0)

        sds_l = guidance(img, text_embeddings)["loss_sds"]
        points.grad = None
        sds_l.backward(retain_graph=True)
        sds_g = points.grad

        xing_l = xing_loss(points_mapped)
        points.grad = None
        xing_l.backward(retain_graph=True)
        xing_g = points.grad

        pbar.set_postfix_str(f"SDS grad: {sds_g.norm():.6f}, Xing grad: {xing_g.norm():.6f}")

        # px_l = pixel_loss(img, target_img)
        # pbar.set_postfix_str(f"Pixel loss: {px_l.item():.6f}, Xing loss: {xing_l.item():.6f}")

        points.grad = sds_g + XING_WT * xing_g
        # loss = sds_l + XING_WT * xing_l
        # loss = PX_WT * px_l + XING_WT * xing_l
        # loss.backward()
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
                plt.imsave(f"optvg/{it}.png", img_np)

    points_mapped = map_to_canvas(points, canvas_dims=CANVAS_DIMS)
    with torch.no_grad():
        plot_curve(points_mapped, closed=True, canvas_dims=CANVAS_DIMS)
        plt.savefig(f"optvg/final.png")


