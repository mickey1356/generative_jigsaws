# %%
import numpy as np
import pydiffvg
import torch
import matplotlib.pyplot as plt

pydiffvg.set_use_gpu(torch.cuda.is_available())

# %%
# points is made up of [pt, cp, cp] * N
def render_curve(points, closed=False, canvas_dims=256):
    n_cps = torch.LongTensor([2] * (points.shape[0] // 3))
    path = pydiffvg.Path(num_control_points=n_cps, points=points, is_closed=closed, stroke_width=torch.tensor(0.0))
    shapes = [path]
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
    shape_groups = [path_group]
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_dims, canvas_dims, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(canvas_dims, canvas_dims, 2, 2, 0, None, *scene_args)
    img_np = img.cpu().numpy()
    # plot the points 
    plt.figure(figsize=(8,8))
    plt.imshow(img_np)
    plt.scatter(points[::3, 0], points[::3, 1], color='red')
    plt.scatter(points[1::3, 0], points[1::3, 1], color='blue', marker="+")
    plt.scatter(points[2::3, 0], points[2::3, 1], color='blue', marker="x")

# %%
# generates a bezier circle with N main points (and 2N control points)
# each main point has 2 control points (cp, pt, cp)
# returns [pt0, cp0, cp1, pt1, cp1, cp2, pt2, cp2 ... cp0]
def bezier_circle(radius, center, N, cp_dist_ratio=0.2):
    points = []
    cp_dist = radius * cp_dist_ratio
    avg_deg = 2 * np.pi / N
    for i in range(N):
        pt = (radius * np.cos(i * avg_deg) + center[0], radius * np.sin(i * avg_deg) + center[1])
        # the control points will be perpendicular to the radius
        cp1 = (pt[0] + cp_dist * np.cos(i * avg_deg + np.pi / 2), pt[1] + cp_dist * np.sin(i * avg_deg + np.pi / 2))
        cp2 = (pt[0] + cp_dist * np.cos(i * avg_deg - np.pi / 2), pt[1] + cp_dist * np.sin(i * avg_deg - np.pi / 2))
        points.append(cp2)
        points.append(pt)
        points.append(cp1)
    return torch.tensor(points[1:] + points[:1]).float()

# %%
# points = torch.tensor([
#     [120.0,  30.0], # pt
#     [150.0,  80.0], # cp
#     [ 45.0, 198.0], # cp
#     [ 60.0, 218.0], # pt
#     [ 90.0, 180.0], # cp
#     [200.0,  65.0], # cp
#     [210.0,  98.0], # pt
#     [220.0,  70.0], # cp
#     [130.0,  55.0], # cp
# ])
# render_curve(points, closed=True)

points2 = bezier_circle(80, (128, 128), 8, cp_dist_ratio=0.2)
render_curve(points2, closed=False)
# %%
