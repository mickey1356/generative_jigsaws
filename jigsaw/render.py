import numpy as np
import torch
import nvdiffrast.torch as dr
import igl
import pydiffvg

def preprocess_piece(Vs, F, angle, domain=0.8):
    # given (V, F) for a piece, rotate by angle and scale
    # then convert to homogeneous coordinates with z=0 and w=1
    # compute area of each triangle
    triangles = Vs[F].view(-1, 3, 2) # (num_tris, [verts_pos])
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    tri_areas = 0.5 * torch.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    # compute center of mass as area-weighted average of triangle centroids
    tri_centroids = triangles.mean(dim=1) # (num_tris, 2)
    center_of_mass = (tri_centroids * tri_areas[:, None]).sum(dim=0) / tri_areas.sum()
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32).to(Vs.device)
    Vs_rotated = (Vs - center_of_mass.detach()) @ R.T
    # scale uniformly so it takes up as much space in [-domain, domain]
    bbox = Vs_rotated[F].view(-1, 2).min(dim=0)[0], Vs_rotated[F].view(-1, 2).max(dim=0)[0]
    scale = 2 * domain / max(bbox[1] - bbox[0])
    Vs_new = Vs_rotated * scale.detach()

    Vs_4d_rotated = torch.cat([Vs_new, torch.zeros((Vs.shape[0], 1), device=Vs.device), torch.ones((Vs.shape[0], 1), device=Vs.device)], dim=1)
    # (|V|, 4)
    return Vs_4d_rotated


# probably can be removed
def render_tris(Vs, Fs, ranges, glctx, bg=None, canvas_dims=512):
    if bg is None:
        bg = torch.ones((1, canvas_dims, canvas_dims, 3), device=Vs.device)
    C = torch.zeros((Vs.shape[0], 3), device=Vs.device)

    # convert Vs (N, 2) to (N, 4) by adding z=0 and w=1
    Vs_4d = torch.cat([Vs, torch.zeros((Vs.shape[0], 1), device=Vs.device), torch.ones((Vs.shape[0], 1), device=Vs.device)], dim=1)
    rast, _ = dr.rasterize(glctx, Vs_4d, Fs, resolution=[canvas_dims, canvas_dims], ranges=ranges)
    mask = rast[..., -1:] == 0
    col, _ = dr.interpolate(C, rast, Fs)
    out = dr.antialias(torch.where(mask, bg, col), rast, Vs_4d, Fs)
    return out



def render_list(Vs, Fs, angles, glctx, bg=None, canvas_dims=512, domain=0.8):
    # given a list of triangles, render each piece separately
    if bg is None:
        bg = torch.ones((1, canvas_dims, canvas_dims, 3), device=Vs.device)
    C = torch.zeros((Vs.shape[0], 3), device=Vs.device)

    out_imgs = []
    for i in range(len(Fs)):
        F = Fs[i]
        if i >= len(angles):
            angle = 0
        else:
            angle = angles[i]

        Vs_4d_rotated = preprocess_piece(Vs, F, angle, domain=domain)[None, ...]
        rast, _ = dr.rasterize(glctx, Vs_4d_rotated, F, resolution=[canvas_dims, canvas_dims])
        mask = rast[..., -1:] == 0
        col, _ = dr.interpolate(C, rast, F)
        out = dr.antialias(torch.where(mask, bg, col), rast, Vs_4d_rotated, F)
        out_imgs.append(out)
    return torch.concat(out_imgs, dim=0)



def render_single(V, F, angle, glctx, bg=None, canvas_dims=512, domain=0.8):
    # renders a single (V, F) piece with the given angle
    # V: (V, 2), F: (F, 3), uv: (V, 2) in [0, 1], texture: (H, W, 3) in [0, 1]
    if bg is None:
        bg = torch.ones((1, canvas_dims, canvas_dims, 3), device=V.device)
    
    Vs_4d = preprocess_piece(V, F, angle, domain=domain)[None, ...]
    rast, _ = dr.rasterize(glctx, Vs_4d, F, resolution=[canvas_dims, canvas_dims])
    mask = rast[..., -1:] == 0
    C = torch.zeros((V.shape[0], 3), device=V.device)
    col, _ = dr.interpolate(C, rast, F)
    out = dr.antialias(torch.where(mask, bg, col), rast, Vs_4d, F)
    return out


def render_texture(V, F, uv, glctx, angle=0, texture=None, bg=None, canvas_dims=512, preprocess=True):
    # renders a TEXTURED piece
    # V: (V, 2), F: (F, 3), uv: (V, 2) in [0, 1], texture: (1, H, W, 3) in [0, 1], bg: (3, ) in [0, 1]
    if bg is None:
        bg = torch.ones((1, canvas_dims, canvas_dims, 3), device=V.device)
    else:
        bg = bg[None, None, None, :].repeat(1, canvas_dims, canvas_dims, 1)
    if texture is None:
        # create a dummy texture based on the uv-vis
        r = np.linspace(0, 1, canvas_dims)
        g = np.linspace(0, 1, canvas_dims)
        rg = np.stack(np.meshgrid(r, g), axis=-1)
        rgb = np.concatenate([rg, np.zeros((canvas_dims, canvas_dims, 1))], axis=-1)
        texture = torch.tensor(rgb, dtype=torch.float32).to(V.device)[None, ...]

    # convert V (N, 2) to (1, N, 4) by adding z=0 and w=1
    if preprocess:
        V_4d = preprocess_piece(V, F, angle, domain=0.8)[None, ...]
    else:
        V_4d = torch.cat([V, torch.zeros((V.shape[0], 1), device=V.device), torch.ones((V.shape[0], 1), device=V.device)], dim=1)[None, ...]
    rast_out, rast_db = dr.rasterize(glctx, V_4d, F, resolution=[canvas_dims, canvas_dims])
    uv_out, uv_da = dr.interpolate(uv, rast_out, F, rast_db=rast_db, diff_attrs="all")
    # sample texture using texc
    col = dr.texture(texture, uv_out, uv_da, filter_mode="linear-mipmap-linear", max_mip_level=9)
    # alpha = torch.clamp(rast_out[..., -1:], max=1)
    # depth = rast_out[:, :, :, 2]
    # col = torch.concat((col, alpha), dim=-1)
    # anti-aliasing
    mask = rast_out[..., -1:] == 0
    col = dr.antialias(torch.where(mask, bg, col), rast_out, V_4d, F)
    return col


def render_single_nvd(V, F, glctx, bg=None, canvas_dims=512):
    # renders (V, F)
    # V: (V, 2), F: (F, 3)
    if bg is None:
        bg = torch.ones((1, canvas_dims, canvas_dims, 3), device=V.device)
    else:
        bg = bg[None, None, None, :].repeat(1, canvas_dims, canvas_dims, 1)
    
    # convert to homogeneous coordinates
    V_4d = torch.cat([V, torch.zeros((V.shape[0], 1), device=V.device), torch.ones((V.shape[0], 1), device=V.device)], dim=1)[None, ...]
    rast, _ = dr.rasterize(glctx, V_4d, F, resolution=[canvas_dims, canvas_dims])
    mask = rast[..., -1:] == 0
    C = torch.zeros((V.shape[0], 3), device=V.device)
    col, _ = dr.interpolate(C, rast, F)
    out = dr.antialias(torch.where(mask, bg, col), rast, V_4d, F)
    return out


def render_pieces_nvd(V, tris, nFs, tri_overall, n_F_overall, angles, glctx, bg=None, canvas_dims=512, prep=True, scale_factor=0.8, domain=[-1, 1]):
    out_imgs = []
    N = len(nFs)
    for i in range(N):
        # get polyline vertices
        Ps = V[nFs[i]]
        if prep:
            # center and then rotate by given angle (in deg)
            Ps = preprocess_vertices(Ps, angles[i], scale_factor=scale_factor, domain=domain)
        # # flip y-axis
        Ps[:, 1] = -Ps[:, 1]
        # # map to canvas coordinates
        # Ps_canvas = map_to_canvas(Ps, domain=domain, canvas_dims=canvas_dims)
        # render
        img = render_single_nvd(Ps, tris[i], glctx, bg=bg, canvas_dims=canvas_dims)
        out_imgs.append(img)
    # render the overall piece
    Ps_overall = V[n_F_overall]
    # # flip y-axis
    Ps_overall[:, 1] = -Ps_overall[:, 1]
    # # map to canvas coordinates
    # Ps_overall_canvas = map_to_canvas(Ps_overall, domain=domain, canvas_dims=canvas_dims)
    overall_img = render_single_nvd(Ps_overall, tri_overall, glctx, bg=bg, canvas_dims=canvas_dims)
    
    return torch.concat(out_imgs, dim=0), overall_img


def preprocess_vertices(Ps, angle, scale_factor=0.8, domain=[-1, 1], bezier=False):
    angle = angle / 180 * np.pi
    R = torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]], device=Ps.device)
    if bezier:
        center = Ps[::3].mean(dim=0).detach()
    else:
        center = Ps.mean(dim=0).detach()
    Ps_rotated = (Ps - center) @ R.T
    # scale so that it fills the domain
    bbox_min = Ps_rotated.min(dim=0)[0]
    bbox_max = Ps_rotated.max(dim=0)[0]
    longest_dim = torch.max(bbox_max - bbox_min).detach()
    scale = scale_factor * (domain[1] - domain[0]) / longest_dim
    Ps_new = Ps_rotated * scale
    return Ps_new


def map_to_canvas(V, domain=[-1, 1], canvas_dims=512):
    # maps V in given domain to [0, canvas_dims]
    V_mapped = (V - domain[0]) / (domain[1] - domain[0]) * canvas_dims
    return V_mapped

def render_polyline(Ps, fill=[0, 0, 0], bg=None, closed=True, canvas_dims = 512, seed=0):
    # Differentiably renders the given polyline using DiffVG
    # Ps: (|Ps|, 2) torch tensor of the vertex positions of the polyline in canvas coordinates ([0, canvas_dims])
    n_cps = torch.LongTensor([0] * Ps.shape[0])
    path = pydiffvg.Path(num_control_points=n_cps, points=Ps, is_closed=closed, stroke_width=torch.tensor(0.0))
    shapes = [path]
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor(fill + [1.0]))
    shape_groups = [path_group]
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_dims, canvas_dims, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(canvas_dims, canvas_dims, 2, 2, seed, None, *scene_args)
    # RenderFunction.apply gives RGBA image
    # composite onto background
    if bg is None:
        bg = torch.ones_like(img[:, :, :3])
    img = img[:, :, :3] * img[:, :, 3:4] + bg * (1 - img[:, :, 3:4])
    return img

def render_polylines_raw(V, Fs, overall_F, angles, prep=True, scale_factor=0.8, domain=[-1, 1], fill=[0, 0, 0], bg=None, closed=True, canvas_dims = 512, seed=0):
    # Differentiably renders the given polylines using DiffVG
    # V: (|V|, 2) torch tensor of all vertex positions in domain
    # Fs: list of lists of vertex indices for each polyline
    # overall_F: list of vertex indices for the overall piece
    # angles: list of angles for each polyline
    out_imgs = []
    N = len(Fs)
    for i in range(N):
        # get polyline vertices
        Ps = V[Fs[i]]
        if prep:
            # center and then rotate by given angle (in deg)
            Ps = preprocess_vertices(Ps, angles[i], scale_factor=scale_factor, domain=domain)
        # flip y-axis
        Ps[:, 1] = -Ps[:, 1]
        # map to canvas coordinates
        Ps_canvas = map_to_canvas(Ps, domain=domain, canvas_dims=canvas_dims)
        # render
        img = render_polyline(Ps_canvas, fill=fill, bg=bg, closed=closed, canvas_dims=canvas_dims, seed=seed)
        out_imgs.append(img)
    # render the overall piece
    Ps_overall = V[overall_F]
    # flip y-axis
    Ps_overall[:, 1] = -Ps_overall[:, 1]
    # map to canvas coordinates
    Ps_overall_canvas = map_to_canvas(Ps_overall, domain=domain, canvas_dims=canvas_dims)
    overall_img = render_polyline(Ps_overall_canvas, fill=fill, bg=bg, closed=True, canvas_dims=canvas_dims, seed=seed)
    
    return torch.stack(out_imgs, dim=0), overall_img.unsqueeze(0)



def render_bezier(Ps, fill=[0., 0., 0., 1.], bg=None, closed=True, canvas_dims = 512, seed=0):
    # renders a cubic bezier curve defined by [pt, cp1, cp2, pt, cp1, cp2, ...]
    ncps = torch.LongTensor([2] * (Ps.shape[0] // 3))
    path = pydiffvg.Path(num_control_points=ncps, points=Ps, is_closed=closed, stroke_width=torch.tensor(0.0))
    shapes = [path]
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor(fill), stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
    shape_groups = [path_group]
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_dims, canvas_dims, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(canvas_dims, canvas_dims, 2, 2, seed, None, *scene_args)
    # composite onto background
    if bg is None:
        bg = torch.ones_like(img[:, :, :3])
    img = img[:, :, :3] * img[:, :, 3:4] + bg * (1 - img[:, :, 3:4])
    return img


def render_bezier_raw(V, Fs, overall_F, angles, prep=True, scale_factor=0.8, domain=[-1, 1], fill=[0., 0., 0., 1.], bg=None, closed=True, canvas_dims = 512, seed=0):
    # renders bezier curves defined by control points in V and vertex indices in Fs and overall_F
    out_imgs = []
    N = len(Fs)
    for i in range(N):
        # get control points for this piece
        Ps = V[Fs[i]]
        if prep:
            Ps = preprocess_vertices(Ps, angles[i], scale_factor=scale_factor, domain=domain, bezier=True)
        # flip y-axis
        Ps[:, 1] = -Ps[:, 1]
        # map to canvas coordinates
        Ps_canvas = map_to_canvas(Ps, domain=domain, canvas_dims=canvas_dims)
        # render
        img = render_bezier(Ps_canvas, fill=fill, bg=bg, closed=closed, canvas_dims=canvas_dims, seed=seed)
        out_imgs.append(img)
    # render the overall piece
    Ps_overall = V[overall_F]
    # flip y-axis
    Ps_overall[:, 1] = -Ps_overall[:, 1]
    # map to canvas coordinates
    Ps_overall_canvas = map_to_canvas(Ps_overall, domain=domain, canvas_dims=canvas_dims)
    overall_img = render_bezier(Ps_overall_canvas, fill=fill, bg=bg, closed=closed, canvas_dims=canvas_dims, seed=seed)
    
    return torch.stack(out_imgs, dim=0), overall_img.unsqueeze(0)