import numpy as np
import tqdm


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
