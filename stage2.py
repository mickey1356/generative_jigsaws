import numpy as np
import matplotlib.pyplot as plt
import tomlkit
import argparse
import tqdm
import os
import pickle
import random
import triangle as tr
import trimesh
import igl

import torch

import jigsaw.fsm as fsm
from jigsaw.samplers import fibonacci_lattice, poisson_disk_sampling, grid
from jigsaw.ffn import LearnableImageFourier, LearnableImageFourierNoFixed, get_uv_grid
from jigsaw.postprocess import extract_polygons, simplify_polygons_topological, convert_polygons_to_paths
from jigsaw.mesher import triangulate
from jigsaw.helpers import unique_name

def laplacian_kernel(radius):
    # create a (2 * radius + 1) stencil for computing the laplacian of a 2D curve
    offsets = np.arange(-radius, radius + 1)
    K = 2 * radius + 1
    A = np.zeros((K, K))
    for k in range(K):
        A[k] = offsets ** k
    b = np.zeros(K)
    b[2] = 2
    coeffs = np.linalg.solve(A, b)
    return coeffs

def apply_laplacian(P, kernel_weights, numpy=False):
    if numpy:
        roll = lambda x, i: np.roll(x, i, axis=0)
    else:
        roll = lambda x, i: torch.roll(x, i, dims=0)
    r = (len(kernel_weights) - 1) // 2
    L = 0
    for i, w in enumerate(kernel_weights):
        L += w * roll(P, i - r)
    return L

def write_obj(V, UV, F, filename):
    with open(filename, "w") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} 0\n")
        for uv in UV:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        for face in F:
            f.write('f ' + ' '.join([f'{v+1}/{v+1}' for v in face]) + '\n')

def main(folder, config: tomlkit.TOMLDocument):
    # general settings
    device = config["general"].get("device", "cuda")

    seed = config["general"].get("seed", -1)
    if seed < 0:
        seed = random.randint(0, 1000000)
        config["general"]["seed"] = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    # reintialize source points
    N_pieces = config["general"].get("pieces", -1)
    smax = config["misc"].get("smax", 7.1)
    smin = config["misc"].get("smin", 0.1)

    # other params
    bg_border = config["misc"].get("bg_border", 8)

    # post-processing params
    post_dim = config["postprocess"].get("post_dim", 1024)
    smoothing_kernel = config["postprocess"].get("smoothing_kernel", 2)
    smoothing_iters = config["postprocess"].get("smoothing_iters", 1000)
    smoothing_lr = config["postprocess"].get("smoothing_lr", 5e-3)
    simplify_eps = config["postprocess"].get("simplify_eps", 1e-4)

    # params for output
    print_dim = config["output"].get("print_dim", 10) # in mm
    thickness = config["output"].get("thickness", 0.2) # in mm
    uv_padding = config["output"].get("uv_padding", 0.9)
    uv_x_dim = config["output"].get("uv_x_dim", 3000)
    uv_dpi = config["output"].get("uv_dpi", 300)
    cut_padding = config["output"].get("cut_padding", 0.01)
    cut_scale = config["output"].get("cut_scale", 20)
    cut_padding_uv = config["output"].get("cut_padding_uv", 0.01)
    # combined_scale = config["output"].get("combined_scale", 2)
    combined_x_dim = config["output"].get("combined_x_dim", 3000)

    # create folders
    # create an output folder for final results, if provided
    ofolder = config["output"].get("output_folder", "")
    if ofolder != "":
        name = config["folder"].get("name", "base")
        os.makedirs(os.path.join(ofolder, name), exist_ok=True)
        output_folder = os.path.join(ofolder, name)
    else:
        os.makedirs(os.path.join(folder, "refine"), exist_ok=True)
        output_folder = os.path.join(folder, "refine")


    model_weights_exist = os.path.exists(os.path.join(folder, "model_weights.pth"))
    final_vars_exist = os.path.exists(os.path.join(folder, "final_vars.pkl"))
    pieces_exist = os.path.exists(os.path.join(folder, "pieces.npy"))
    angles_exist = os.path.exists(os.path.join(folder, "rotation_angles.npy"))


    if model_weights_exist and final_vars_exist:
        # create uv grid for high-res slowness and eikonal
        uv_grid = get_uv_grid(post_dim, post_dim, device=device)

        model = LearnableImageFourierNoFixed(channels=1).to(device)
        model.load_state_dict(torch.load(os.path.join(folder, "model_weights.pth"), map_location=device))

        with open(os.path.join(folder, "final_vars.pkl"), "rb") as f:
            final_vars = pickle.load(f)
        srcs = torch.from_numpy(final_vars["locations"]).float().to(device)
        angles = final_vars["rotation_angles"]

        # get the slowness map
        with torch.inference_mode():
            f_tex = model(uv_grid).squeeze()
        slowness = (smax - smin) * f_tex + smin
        slowness[:bg_border, :] = smin
        slowness[-bg_border:, :] = smin
        slowness[:, :bg_border] = smin
        slowness[:, -bg_border:] = smin

        fn_iters = 4 * post_dim
        T = fsm.FSMGpuFn.apply(srcs, slowness, fn_iters)

        hard_T = fsm.hard_voronoi(T)
        pieces = hard_T.copy()
        for i in range(len(srcs) - N_pieces):
            pieces[hard_T == N_pieces + i] = -1
        pieces += 1

    elif pieces_exist and angles_exist:
        pieces = np.load(os.path.join(folder, "pieces.npy"), allow_pickle=True) + 1
        angles = np.load(os.path.join(folder, "rotation_angles.npy"), allow_pickle=True)

    else:
        raise FileNotFoundError("Could not find either the model weights or pieces. Make sure you have run stage 1 and that the folder path is correct.")


    # if os.path.exists(os.path.join(folder, "prompts.pkl")):
    #     prompts_dict = pickle.load(open(os.path.join(folder, "prompts.pkl"), "rb"))
    #     prompts = prompts_dict['prompt']
    #     overall_prompt = prompts_dict['overall_prompt']
    # else:
    #     raise FileNotFoundError("Could not load prompts. Make sure you have run stage 1 and that the folder path is correct.")


    # print("Extracting polygons...")
    vertices, Fs, overall_piece = extract_polygons(pieces)


    # print("Smoothing polygons...")
    laplacian_coeffs = laplacian_kernel(smoothing_kernel)
    for _ in range(smoothing_iters):
        delta = np.zeros_like(vertices)
        for F in Fs:
            P = vertices[F]
            L = apply_laplacian(P, laplacian_coeffs, numpy=True)
            delta[F] += L
        dV = smoothing_lr * delta
        vertices += dV

    # print("Simplifying polygons...")
    vertices, all_Fs = simplify_polygons_topological(vertices, Fs + [overall_piece], simplify_eps)
    Fs = all_Fs[:-1]
    overall_piece = all_Fs[-1]

    all_V = []
    all_F = []
    all_UV = []

    # get dims of the overall piece
    oV = vertices[overall_piece]
    o_dims = oV.max(axis=0) - oV.min(axis=0)
    # find the scale so that the shortest side is print_dim
    print_scale = print_dim / o_dims.min()

    # uv map will be a grid
    uv_map_nw = np.ceil(np.sqrt(N_pieces)).astype(int)
    uv_map_nh = np.ceil(N_pieces / uv_map_nw).astype(int)

    # offsets for cutting
    # cut_xoff = 0
    # cut_yoff = 0
    # max_y_so_far = 0
    all_pieces = []
    cut_unscaled_max_dims = [0, 0]
    all_unscaled = []
    # max_dim_x = 0
    # max_dim_y = 0

    # uv_unscaled_max_dims = [0, 0]

    # save raw files
    pieces = []
    comb = []
    min_vx = np.inf
    min_vy = np.inf
    for i, F in enumerate(Fs):
        poly = vertices[F]
        comb.append(poly)
        angle = angles[i] / 180 * np.pi
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        poly_rot = (poly - poly.mean(axis=0)) @ R.T
        # shift corner to 0, 0
        poly_rot -= poly_rot.min(axis=0)
        pieces.append(poly_rot)
        
        min_vx = min(min_vx, poly[:, 0].min())
        min_vy = min(min_vy, poly[:, 1].min())
    # shift combined pieces
    for piece in comb:
        piece -= np.array([min_vx, min_vy])
        pieces.append(piece)

    overall_piece_v = vertices[overall_piece]
    overall_piece_v -= overall_piece_v.min(axis=0)
    pieces.append(overall_piece_v)
    with open(os.path.join(output_folder, "pieces_raw.pkl"), "wb") as f:
        pickle.dump(pieces, f)


    # print("Saving results...")
    for i, F in enumerate(Fs):
        # get vertices of the piece
        poly = vertices[F]
        # triangulate for texturing
        A = dict(vertices=poly, segments=np.array([[i, (i + 1) % len(poly)] for i in range(len(poly))]))
        T = tr.triangulate(A, "pqa0.001")
        V_tri, F_tri = T['vertices'], T['triangles']
        all_V.append(V_tri)
        all_F.append(F_tri + sum(len(vs) for vs in all_V[:-1]))

        # rotate V_tri according to angle
        angle = angles[i] / 180 * np.pi
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        V_tri_rot = (V_tri - poly.mean(axis=0)) @ R.T

        # translate it so min is at (0, 0)
        V_tri_rot -= V_tri_rot.min(axis=0)

        # save obj separately for each piece
        V_indv = V_tri_rot * print_scale
        # extrude in z direction by thickness
        piece_mesh = trimesh.creation.extrude_triangulation(V_indv, F_tri, thickness)
        piece_mesh.export(os.path.join(output_folder, f"piece_{i}.obj"))

        # now we need to create the uvs
        # scale V_tri_rot (for uv coordinates)
        bbox_dims = V_tri_rot.max(axis=0) - V_tri_rot.min(axis=0)
        # scale it so the max_dim leaves some padding
        max_dim = bbox_dims.max()
        scale = uv_padding / max_dim
        V_tri_uv = V_tri_rot * scale
        # move it to the center
        ndims = V_tri_uv.max(axis=0) - V_tri_uv.min(axis=0)
        V_tri_uv += (1 - ndims) / 2

        # now put it in the uv map
        uv_i = i % uv_map_nw
        uv_j = i // uv_map_nw
        V_tri_uv[:, 0] += uv_i
        V_tri_uv[:, 1] += uv_j
        V_tri_uv /= np.array([uv_map_nw, uv_map_nh]) # normalize to [0, 1]

        all_UV.append(V_tri_uv)

        # offset piece for cutting
        # rotate piece
        poly_rot = (poly - poly.mean(axis=0)) @ R.T
        poly_dim = poly_rot.max(axis=0) - poly_rot.min(axis=0)
        # max_y_so_far = max(max_y_so_far, poly_dim[1])
        # translate so min is at (0, 0)
        poly_rot -= poly_rot.min(axis=0)
        # add padding
        # poly_rot += cut_padding_uv
        all_pieces.append(poly_rot)
        # triangulate for uv map
        A = dict(vertices=poly_rot, segments=np.array([[i, (i + 1) % len(poly)] for i in range(len(poly))]))
        T = tr.triangulate(A, "pqa0.001")
        V_tri, F_tri = T['vertices'], T['triangles']
        # add to unscaled uv map
        all_unscaled.append((V_tri, F_tri))
        # keep track of size of pieces
        cut_unscaled_max_dims[0] = max(cut_unscaled_max_dims[0], poly_dim[0] + 2 * cut_padding_uv)
        cut_unscaled_max_dims[1] = max(cut_unscaled_max_dims[1], poly_dim[1] + 2 * cut_padding_uv)



    # convert all pieces into one svg for cutting
    cut_paths = convert_polygons_to_paths(Fs)
    piece_v = vertices[overall_piece]
    bbox_min, bbox_max = piece_v.min(axis=0), piece_v.max(axis=0)
    v_scale = (vertices - bbox_min) * cut_scale
    cut_paths_v = [v_scale[p] + cut_padding for p in cut_paths]
    svg_width = v_scale[:, 0].max() + 2 * cut_padding
    svg_height = v_scale[:, 1].max() + 2 * cut_padding

    # write the svg file for cutting
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">\n'
    for pl in cut_paths_v:
        path = "M " + " L ".join([f"{p[0]},{p[1]}" for p in pl])
        svg += f'<path d="{path}" fill="none" stroke="red" stroke-width="0.001px"/>\n'
    svg += '</svg>\n'

    with open(os.path.join(output_folder, "cutting.svg"), "w") as f:
        f.write(svg)

    # make svg with loops for ease of figures
    with open(os.path.join(output_folder, "full_figures.svg"), "w") as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2" viewBox="0 0 2 2">\n')
        for i, F in enumerate(Fs):
            poly = vertices[F]
            path = "M " + " L ".join([f"{p[0] + 1},{p[1] + 1}" for p in poly]) + " Z"
            f.write(f'<path d="{path}" fill="none" stroke="red" stroke-width="0.001"/>\n')
        f.write('</svg>\n')

    # write obj with uv map
    V_full = np.concatenate(all_V, axis=0)
    F_full = np.concatenate(all_F, axis=0)
    UV_full = np.concatenate(all_UV, axis=0)

    write_obj(V_full, UV_full, F_full, os.path.join(output_folder, "full.obj"))

    # scale one full set for printing
    overall_piece_v *= print_scale
    A = dict(vertices=overall_piece_v, segments=np.array([[i, (i + 1) % len(overall_piece_v)] for i in range(len(overall_piece_v))]))
    T = tr.triangulate(A, "pq0")
    V_overall, F_overall = T['vertices'], T['triangles']
    full_piece = trimesh.creation.extrude_triangulation(V_overall, F_overall, thickness)
    full_piece.export(os.path.join(output_folder, f"full_print.obj"))

    
    # write uv map
    uv_y_dim = uv_x_dim / uv_map_nw * uv_map_nh
    fig_width = uv_x_dim / uv_dpi
    fig_height = uv_y_dim / uv_dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=uv_dpi)
    ax.set_xlim(0, uv_x_dim)
    ax.set_ylim(0, uv_y_dim)
    ax.axis("off")
    UV_full_plot = UV_full * np.array([uv_x_dim, uv_y_dim])
    ax.triplot(UV_full_plot[:, 0], UV_full_plot[:, 1], F_full, color='black', linewidth=0.1)
    plt.savefig(os.path.join(output_folder, "uv_map.png"), dpi=uv_dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


    # write the corresponding uv maps

    # overall_piece_cut = vertices[overall_piece] - vertices[overall_piece].min(axis=0)
    # overall_piece_cut *= cut_scale
    # overall_piece_cut[:, 0] += dim_x / 2
    # overall_piece_cut[:, 1] += dim_y + cut_padding * cut_scale
    # dim_y += overall_piece_cut[:, 1].max() + cut_padding * cut_scale

    overall_piece_v = vertices[overall_piece]
    overall_piece_dim = overall_piece_v.max(axis=0) - overall_piece_v.min(axis=0)

    raw_width = cut_unscaled_max_dims[0] * uv_map_nw
    raw_height = cut_unscaled_max_dims[1] * uv_map_nh + cut_padding + overall_piece_dim[1]

    combined_scale = combined_x_dim / raw_width
    combined_y_dim = raw_height * combined_scale

    fig_width = combined_x_dim / uv_dpi
    fig_height = combined_y_dim / uv_dpi
    fig, ax = plt.subplots(dpi=uv_dpi)
    max_coords = [0, 0]
    svg_uv = ''
    for i, (V_tri, F_tri) in enumerate(all_unscaled):
        uv_i = i % uv_map_nw
        uv_j = i // uv_map_nw
        V_tri_plot = V_tri * combined_scale
        # offset by uv map position
        V_tri_plot[:, 0] += uv_i * cut_unscaled_max_dims[0] * combined_scale
        V_tri_plot[:, 1] += uv_j * cut_unscaled_max_dims[1] * combined_scale
        max_coords[0] = max(max_coords[0], V_tri_plot[:, 0].max())
        max_coords[1] = max(max_coords[1], V_tri_plot[:, 1].max())
        ax.triplot(V_tri_plot[:, 0], V_tri_plot[:, 1], F_tri, color='black', linewidth=0.1)

        bnd = igl.boundary_loop(F_tri)
        path = "M " + " L ".join([f"{V_tri_plot[v, 0]},{combined_y_dim - V_tri_plot[v, 1]}" for v in bnd]) + " Z"
        svg_uv += f'<path d="{path}" fill="none" stroke="red" stroke-width="0.001px"/>\n'
    

    # copy for overall_piece
    svg_uv_overall = svg_uv
    
    # only pieces
    svg_uv = f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_coords[0]}" height="{max_coords[1]}" viewBox="0 0 {max_coords[0]} {max_coords[1]}">\n' + svg_uv
    svg_uv += '</svg>\n'

    with open(os.path.join(output_folder, "pieces.svg"), "w") as f:
        f.write(svg_uv)

    # add overall_piece
    # move corner to 0, 0
    overall_piece_v -= overall_piece_v.min(axis=0)
    overall_piece_v *= combined_scale
    overall_piece_v[:, 1] += max_coords[1] + cut_padding * combined_scale

    ax.fill(overall_piece_v[:, 0], overall_piece_v[:, 1], color='black', fill=False)
    svg_uv_overall += f'<path d="M ' + " L ".join([f"{p[0]},{combined_y_dim - p[1]}" for p in overall_piece_v]) + ' Z" fill="none" stroke="red" stroke-width="0.001px"/>\n'

    max_coords[0] = max(max_coords[0], overall_piece_v[:, 0].max())
    max_coords[1] = max(max_coords[1], overall_piece_v[:, 1].max())

    svg_uv_overall = f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_coords[0]}" height="{max_coords[1]}" viewBox="0 0 {max_coords[0]} {max_coords[1]}">\n' + svg_uv_overall
    svg_uv_overall += '</svg>\n'

    with open(os.path.join(output_folder, "cutting_uv.svg"), "w") as f:
        f.write(svg_uv_overall)


    ax.set_xlim(0, max_coords[0])
    ax.set_ylim(0, max_coords[1])
    ax.axis("off")
    ax.set_aspect('equal')
    plt.savefig(os.path.join(output_folder, "cutting_unscaled.png"), dpi=uv_dpi, bbox_inches='tight', pad_inches=0)
    plt.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate jigsaw puzzles from a text prompt.")
    parser.add_argument("-b", "--base_config", type=str, default="configs/postprocess.toml", help="Path to the base config file (TOML format). This will be used as the default config and updated with the provided config.")
    
    parser.add_argument("-f", "--folder", type=str, help="Path to the folder created after stage 1.")
    parser.add_argument("-m", "--multi_folder", type=str, help="Path to a folder containing multiple folders created after stage 1. If provided, the script will run on all subfolders and save results in corresponding subfolders within the output folder.")
    parser.add_argument("-o", "--output_folder", type=str, help="Path to the folder where final results will be saved. If not provided, results will be saved in the refine folder within the input folder.")
    
    parser.add_argument("--print_dim", type=float, help="Length of the longest side of the printed puzzle (in mm). The generated UV coordinates will be scaled to fit within a square of this size.")
    parser.add_argument("--thickness", type=float, help="Thickness of piece when extruded in 3D, (in mm).")

    args = parser.parse_args()

    base = tomlkit.load(open(args.base_config))

    if args.multi_folder is not None:
        subfolders = [f.path for f in os.scandir(args.multi_folder) if f.is_dir()]
        for i, subfolder in enumerate(subfolders):
            try:
                print(f"Processing folder: {subfolder} ({i+1}/{len(subfolders)})")
                config = tomlkit.load(open(os.path.join(subfolder, "config.toml")))

                # update the base config with the new config
                for section in config:
                    if section not in base:
                        base[section] = config[section]
                    else:
                        for key in config[section]:
                            base[section][key] = config[section][key]

                if args.print_dim is not None:
                    base["output"]["print_dim"] = args.print_dim

                if args.thickness is not None:
                    base["output"]["thickness"] = args.thickness

                if args.output_folder is not None:
                    base["output"]["output_folder"] = args.output_folder

                main(subfolder, base)
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except:
                print(f"Error processing folder: {subfolder}. Skipping...")
    else:
        folder = args.folder
        if folder is None:
            raise ValueError("Please provide a folder path using -f or --folder.")
        print(f"Processing folder: {folder}")
        config = tomlkit.load(open(os.path.join(folder, "config.toml")))

        # update the base config with the new config
        for section in config:
            if section not in base:
                base[section] = config[section]
            else:
                for key in config[section]:
                    base[section][key] = config[section][key]

        if args.print_dim is not None:
            base["output"]["print_dim"] = args.print_dim

        if args.thickness is not None:
            base["output"]["thickness"] = args.thickness

        if args.output_folder is not None:
            base["output"]["output_folder"] = args.output_folder

        main(folder, base)