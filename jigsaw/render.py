import numpy as np
import torch
import nvdiffrast.torch as dr
import igl

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
