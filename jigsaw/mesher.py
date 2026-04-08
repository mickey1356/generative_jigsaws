import gmsh
import numpy as np

def triangulate(Vs, Fs, e_size=0.01, boundary=None):
    # triangulate the 2d polygons using gmsh
    # boundary should be a list of vertex indices that represent the boundary of the full piece, if provided
    Vs_3d = np.hstack([Vs, np.zeros((len(Vs), 1))])
    V, tris = mesh_polygons(Vs_3d, Fs, e_size=e_size, boundary=boundary)
    return V[:, :2], tris

def mesh_polygons(Vs, Fs, e_size, boundary=None):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.model.add("t1")

    for i, (x, y, z) in enumerate(Vs):
        gmsh.model.geo.addPoint(x, y, z, e_size, i+1)

    if boundary is not None:
        gmsh.model.geo.addPoint(-1, -1, 0, e_size, len(Vs)+1)
        gmsh.model.geo.addPoint(1, -1, 0, e_size, len(Vs)+2)
        gmsh.model.geo.addPoint(1, 1, 0, e_size, len(Vs)+3)
        gmsh.model.geo.addPoint(-1, 1, 0, e_size, len(Vs)+4)
    
    edges = {}
    for vis in Fs:
        for i in range(len(vis)):
            a = 1 + vis[i]
            b = 1 + vis[(i + 1) % len(vis)]
            edge = (min(a, b), max(a, b))
            if edge not in edges:
                edges[edge] = 1 + len(edges)

    if boundary is not None:
        edges[(len(Vs)+1, len(Vs)+2)] = 1 + len(edges)
        edges[(len(Vs)+2, len(Vs)+3)] = 1 + len(edges)
        edges[(len(Vs)+3, len(Vs)+4)] = 1 + len(edges)
        edges[(len(Vs)+4, len(Vs)+1)] = 1 + len(edges)

    for (s, e), tag in edges.items():
        gmsh.model.geo.addLine(s, e, tag)

    loop_tags = []
    for f, vis in enumerate(Fs):
        loop = []
        for i in range(len(vis)):
            a = 1 + vis[i]
            b = 1 + vis[(i + 1) % len(vis)]
            edge = (a, b)
            if edge in edges:
                loop.append(edges[edge])
            else:
                loop.append(-edges[(b, a)])
        loop_tag = gmsh.model.geo.addCurveLoop(loop)
        loop_tags.append(loop_tag)
    
    # polygons = []
    for f, loop_tag in enumerate(loop_tags):
        s = gmsh.model.geo.addPlaneSurface([loop_tag])
        # polygons.append(s)

    if boundary is not None:
        # add the outer loop
        loop = [edges[(len(Vs)+1, len(Vs)+2)], edges[(len(Vs)+2, len(Vs)+3)], edges[(len(Vs)+3, len(Vs)+4)], edges[(len(Vs)+4, len(Vs)+1)]]
        outer_loop = gmsh.model.geo.addCurveLoop(loop)
        # add the boundary loop
        bloop = []
        for i in range(len(boundary)):
            a = 1 + boundary[i]
            b = 1 + boundary[(i + 1) % len(boundary)]
            edge = (b, a)
            if edge in edges:
                bloop.append(edges[edge])
            else:
                bloop.append(-edges[(a, b)])
        boundary_loop = gmsh.model.geo.addCurveLoop(bloop)
        outer_surface = gmsh.model.geo.addPlaneSurface([outer_loop, boundary_loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    v_n_tags, v_n_coords, _ = gmsh.model.mesh.getNodes()
    V = np.zeros((len(v_n_tags), 3))
    tag_to_idx = {}
    for i, tag in enumerate(v_n_tags):
        V[tag - 1] = v_n_coords[3*i:3*i+3]
        tag_to_idx[tag] = i

    triangles = []
    for f in range(len(Fs) + 1):
        types, e_tags, n_tags = gmsh.model.mesh.getElements(2, f+1)
        tris = []
        for i, t in enumerate(types):
            if t == 2:
                tri = n_tags[i].reshape(-1, 3)
                tri = np.vectorize(tag_to_idx.get)(tri)
                tris.extend(tri)
        triangles.append(np.array(tris))
    return V, triangles
