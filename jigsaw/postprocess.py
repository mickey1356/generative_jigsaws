import numpy as np
import tqdm
from collections import defaultdict


def extract_edges(grid, disable=False):
    # grid is an int array where 0 is the background, and 1 - N is the piece index
    # we assume that 0, 0 is the top-left corner of the image
    # we also assume that the nodes start from 0, 0 and end at H, W (inclusive)
    H, W = grid.shape
    # we want to store edges for each piece, and also edges for the full piece (stored at the first index)
    # edges are stored as an adjacency list of (r, c) -> set of neighboring vertices
    piece_edges = [{} for _ in range(np.max(grid) + 1)]
    def add_edge(piece, p1, p2):
        piece_edges[piece].setdefault(p1, set()).add(p2)
        piece_edges[piece].setdefault(p2, set()).add(p1)

    for y in tqdm.trange(H, disable=disable):
        for x in range(W):
            piece = grid[y, x]
            # check neighbors
            if x + 1 < W and grid[y, x+1] != piece:
                add_edge(piece, (y, x+1), (y+1, x+1))
                add_edge(grid[y, x+1], (y, x+1), (y+1, x+1))
            if y + 1 < H and grid[y+1, x] != piece:
                add_edge(piece, (y+1, x), (y+1, x+1))
                add_edge(grid[y+1, x], (y+1, x), (y+1, x+1))
            # check boundary (technically this can be skipped because there the pieces shouldn't touch the boundary)
            if piece > 0:
                if y == 0:
                    add_edge(piece, (y, x), (y, x+1))
                if y == H - 1:
                    add_edge(piece, (y+1, x), (y+1, x+1))
                if x == 0:
                    add_edge(piece, (y, x), (y+1, x))
                if x == W - 1:
                    add_edge(piece, (y, x+1), (y+1, x+1))
    return piece_edges

def trace_polygon(adjlist):
    # given an adjacency list of vertices, trace out the polygon by following edges
    # return list of vertices in order
    start = list(adjlist.keys())[0]
    polygon = [start]
    visited = set([start])
    current = start

    while True:
        neighbors = adjlist[current]
        next_vertex = None
        for neighbor in neighbors:
            if neighbor not in visited:
                next_vertex = neighbor
                break
        
        if next_vertex is None:
            break
        
        polygon.append(next_vertex)
        visited.add(next_vertex)
        current = next_vertex

    return polygon

def extract_polygons(pieces):
    # given the pieces array, extract the edges of each piece and trace them to form polygons (CCW)
    # returns both the vertex positions, and the face indices of each piece
    # also returns the face indices for the full piece
    H, W = pieces.shape
    N = np.max(pieces)

    piece_edges = extract_edges(pieces)

    polygons = []
    for i in range(N):
        # piece_mask = (pieces == i+1)
        edges = piece_edges[i+1]
        polygon = trace_polygon(edges)
        polygons.append(polygon)

    # get the full piece
    # full_piece_mask = (pieces > 0)
    edges = piece_edges[0]
    full_piece_polygon = trace_polygon(edges)
    polygons.append(full_piece_polygon)

    vertices = []
    vertices_map = {}
    for polygon in polygons:
        for (r, c) in polygon:
            if (c, r) not in vertices_map:
                vertices.append((c, r))
                vertices_map[(c, r)] = len(vertices) - 1
    
    vertices = np.array(vertices, dtype=np.float32)
    # flip y-axis to match image coordinates
    vertices[:, 1] = H - vertices[:, 1]
    # convert from integer coordinates to [-1, 1] range
    vertices[:, 0] = vertices[:, 0] / W * 2 - 1
    vertices[:, 1] = vertices[:, 1] / H * 2 - 1

    faces = []
    for polygon in polygons:
        face = [vertices_map[(c, r)] for r, c in polygon]
        edges = [(vertices[i], vertices[j]) for i, j in zip(face, face[1:] + [face[0]])]
        det = sum((x2 - x1) * (y2 + y1) for (x1, y1), (x2, y2) in edges)
        if det > 0:
            face.reverse() # ensure counter-clockwise order
        faces.append(face)

    return vertices, faces[:-1], faces[-1]


def rdp(points, epsilon):
    if len(points) < 3:
        return points
    
    a = points[0]
    b = points[-1]
    
    if np.all(a == b):
        dists = np.linalg.norm(points[1:-1] - a, axis=1)
    else:
        # Distance from p to line ab: |(p-a) x (b-a)| / |b-a|
        vec_ab = b - a
        normal_ab = np.linalg.norm(vec_ab)
        vec_ap = points[1:-1] - a
        dists = np.abs(vec_ap[:, 0] * vec_ab[1] - vec_ap[:, 1] * vec_ab[0]) / normal_ab
    
    if len(dists) == 0:
        return np.array([a, b])
        
    idx = np.argmax(dists)
    dmax = dists[idx]
    
    if dmax > epsilon:
        # idx in dists corresponds to points[idx+1]
        res1 = rdp(points[:idx+2], epsilon)
        res2 = rdp(points[idx+1:], epsilon)
        return np.vstack([res1[:-1], res2])
    else:
        return np.array([a, b])

def simplify_polygons_topological(vertices, Fs, epsilon):
    from collections import defaultdict
    # 1. Build global edge graph to find junctions
    adj = defaultdict(set)
    for F in Fs:
        for i in range(len(F)):
            u, v = F[i], F[(i+1) % len(F)]
            adj[u].add(v)
            adj[v].add(u)
    
    # Junctions are vertices where boundaries split/merge (degree != 2)
    junctions = {v for v, neighbors in adj.items() if len(neighbors) != 2}
    
    simplified_segments = {} # (path_tuple) -> simplified_coords
    
    new_Fs_coords = []
    for F in Fs:
        new_F_coords = []
        # Find a starting junction if possible
        start_idx = 0
        for i, v in enumerate(F):
            if v in junctions:
                start_idx = i
                break
        
        # Reorder F to start at a junction for consistent segment extraction
        F_rotated = F[start_idx:] + F[:start_idx]
        
        i = 0
        while i < len(F_rotated):
            segment_indices = [F_rotated[i]]
            j = i + 1
            while j < len(F_rotated) and F_rotated[j] not in junctions:
                segment_indices.append(F_rotated[j])
                j += 1
            
            # Closing the segment
            target_idx = j % len(F_rotated)
            segment_indices.append(F_rotated[target_idx])
            
            seg_key = tuple(segment_indices)
            rev_seg_key = tuple(segment_indices[::-1])
            
            if seg_key in simplified_segments:
                s_coords = simplified_segments[seg_key]
            elif rev_seg_key in simplified_segments:
                s_coords = simplified_segments[rev_seg_key][::-1]
            else:
                s_coords = rdp(vertices[segment_indices], epsilon)
                simplified_segments[seg_key] = s_coords
            
            # Avoid duplicating the last point which is the first point of the next segment
            new_F_coords.extend(s_coords[:-1])
            i = j
            
        new_Fs_coords.append(new_F_coords)
    
    # 2. Rebuild global V and Fs to maintain the (V, [F1...Fn]) format
    new_vertices = []
    coord_to_idx = {}
    final_Fs = []
    
    for F_coords in new_Fs_coords:
        f_indices = []
        for coord in F_coords:
            c_tuple = tuple(coord)
            if c_tuple not in coord_to_idx:
                coord_to_idx[c_tuple] = len(new_vertices)
                new_vertices.append(coord)
            f_indices.append(coord_to_idx[c_tuple])
        final_Fs.append(f_indices)
        
    return np.array(new_vertices, dtype=np.float32), final_Fs

def get_edges(Vind):
    # get edges, i.e. returns a list of pairs of vertex indices that form edges
    edges = [(Vind[i], Vind[(i + 1) % len(Vind)]) for i in range(len(Vind))]
    return np.array(edges, dtype=int)

def subdivide_polygons_edges(V, Fs, full_piece, esize=None):
    # first compute min length of edges over all polygons
    if esize is None:
        for F in Fs:
            edges = get_edges(F)
            edge_vecs = V[edges[:, 0]] - V[edges[:, 1]]
            edge_lengths = np.linalg.norm(edge_vecs, axis=1)
            esize = min(esize, edge_lengths.min()) if esize is not None else edge_lengths.min()
    
    edge_to_faces = defaultdict(list)

    for fi, F in enumerate(Fs):
        edges = get_edges(F)
        for e in edges:
            edge_to_faces[tuple(sorted(e))].append(fi)
    
    new_V = V.tolist()
    edges_splits = {} # (i, j) -> [i ... j] with the subdivided vertices in between

    # edge splits
    for (i, j) in edge_to_faces.keys():
        vi, vj = V[i], V[j]
        length = np.linalg.norm(vi - vj)

        if length <= esize:
            edges_splits[(i, j)] = [i, j]
            continue

        k = int(np.ceil(length / esize))
        indices = [i]
        for t in np.linspace(0, 1, k + 1)[1:-1]:
            new_pt = (1 - t) * vi + t * vj
            new_V.append(new_pt)
            indices.append(len(new_V) - 1)
        indices.append(j)

        edges_splits[(i, j)] = indices

    # rebuild faces
    new_Fs = []
    for F in Fs:
        edges = get_edges(F)
        new_F = []
        for e in edges:
            key = tuple(sorted(e))
            split_edge = edges_splits[key]

            # check if edge was flipped
            if split_edge[0] == e[0]:
                new_F.extend(split_edge[:-1])
            else:
                new_F.extend(split_edge[:0:-1])
        new_Fs.append(new_F)

    # rebuild full piece
    full_edges = get_edges(full_piece)
    new_full_piece = []
    for e in full_edges:
        key = tuple(sorted(e))
        split_edge = edges_splits[key]

        if split_edge[0] == e[0]:
            new_full_piece.extend(split_edge[:-1])
        else:
            new_full_piece.extend(split_edge[:0:-1])
    
    return np.array(new_V), new_Fs, new_full_piece

def subdivide_polygons_edges_count(V, Fs, full_piece, target=80):
    def find_edge(F, i, j, k):
        nF = []
        # find the edge (i, j) or (j, i) in F and insert k in between, returning the new F
        for idx in range(len(F)):
            u, v = F[idx], F[(idx + 1) % len(F)]
            if (u == i and v == j) or (u == j and v == i):
                nF.append(u)
                nF.append(k)
            else:
                nF.append(u)
        return nF

    edge_to_faces = defaultdict(list)

    for fi, F in enumerate(Fs):
        edges = get_edges(F)
        for e in edges:
            edge_to_faces[tuple(sorted(e))].append(fi)
    
    new_V = V.tolist()
    nV_np = np.array(new_V)
    edges_splits = {} # (i, j) -> [i ... j] with the subdivided vertices in between

    vertex_cnt = [len(F) for F in Fs]

    nFs = [F[:] for F in Fs]
    n_full_piece = full_piece[:]

    while min(vertex_cnt) < target:
        # figure out which piece has the least vertices
        piece_idx = np.argmin(vertex_cnt)
        # find the longest edge in that piece
        edges = get_edges(nFs[piece_idx])
        edge_vecs = nV_np[edges[:, 0]] - nV_np[edges[:, 1]]
        edge_lengths = np.linalg.norm(edge_vecs, axis=1)
        longest_edge_idx = np.argmax(edge_lengths)
        i, j = edges[longest_edge_idx]
        # subdivide the edge
        vnew = (nV_np[i] + nV_np[j]) / 2
        # add new vertex and update numpy array
        new_V.append(vnew)
        nV_np = np.array(new_V)
        # update polygons
        for fi in edge_to_faces[tuple(sorted((i, j)))]:
            nFs[fi] = find_edge(nFs[fi], i, j, len(new_V) - 1)
            vertex_cnt[fi] += 1
        # update full piece
        n_full_piece = find_edge(n_full_piece, i, j, len(new_V) - 1)
        # update edge_to_faces
        edge_to_faces[tuple(sorted((i, len(new_V) - 1)))].extend(edge_to_faces[tuple(sorted((i, j)))])
        edge_to_faces[tuple(sorted((j, len(new_V) - 1)))].extend(edge_to_faces[tuple(sorted((i, j)))])
        edge_to_faces[tuple(sorted((i, j)))].clear()

    return nV_np, nFs, n_full_piece
    

def convert_to_bezier(V, Fs, overall_piece, alpha_scale=0.3):
    adj = defaultdict(set)
    edge_to_faces = defaultdict(list)

    # build adjacency and edge-to-face mapping
    for fi, F in enumerate(Fs):
        edges = get_edges(F)
        for e in edges:
            edge_to_faces[tuple(sorted(e))].append(fi)
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
    
    # compute per-vertex tangents
    T = np.zeros_like(V)
    for i in range(len(V)):
        vi = V[i]
        neighbors = adj[i]

        for j in neighbors:
            T[i] += V[j] - vi

    # normalize tangents
    T_norm = np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
    T = T / T_norm

    # create bezier edges
    new_V = V.tolist()
    edges_bezier = {} # (i, j) -> [i c1 c2 j] with the control points in between

    for (i, j) in edge_to_faces.keys():
        vi = V[i]
        vj = V[j]
        ti = T[i]
        tj = T[j]

        edge_vec = vj - vi
        edge_len = np.linalg.norm(edge_vec)

        alpha = alpha_scale * edge_len
        c1 = vi + (vj - vi) / 3
        c2 = vj - (vj - vi) / 3

        new_V.extend([c1, c2])
        edges_bezier[(i, j)] = [i, len(new_V) - 2, len(new_V) - 1, j]
        edges_bezier[(j, i)] = [j, len(new_V) - 1, len(new_V) - 2, i]

    # rebuild faces with control points
    new_Fs = []
    for F in Fs:
        edges = get_edges(F)
        new_F = []
        for e in edges:
            key = tuple(e)
            new_F.extend(edges_bezier[key][:-1])
        new_Fs.append(new_F)
    
    # rebuild overall piece
    full_edges = get_edges(overall_piece)
    new_full_piece = []
    for e in full_edges:
        key = tuple(e)
        new_full_piece.extend(edges_bezier[key][:-1])


    return V, np.array(new_V), new_Fs, new_full_piece