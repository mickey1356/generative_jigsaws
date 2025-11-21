import gmsh
import numpy as np

def mesh_surface(verts, faces, e_size, holes=None):
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.model.add("t1")

    for i, (x, y, z) in enumerate(verts):
        gmsh.model.geo.add_point(x, y, z, e_size, i+1)

    edges = {}
    for vis in faces:
        for i in range(len(vis)):
            a = 1 + vis[i]
            b = 1 + vis[(i + 1) % len(vis)]
            edge = (min(a, b), max(a, b))
            if edge not in edges:
                edges[edge] = 1 + len(edges)

    # holes is a nested list of lists for each face
    # [[hole1_face1, hole2_face1, ...], [hole1_face2, ...], ...]
    # where holei_facej is a list of vertex indices defining the hole
    if holes:
        assert len(holes) == len(faces)
        for face_holes in holes:
            for hole in face_holes:
                # hole is a list of vertex indices
                for i in range(len(hole)):
                    a = 1 + hole[i]
                    b = 1 + hole[(i + 1) % len(hole)]
                    edge = (min(a, b), max(a, b))
                    if edge not in edges:
                        edges[edge] = 1 + len(edges)

    for (s, e) in edges:
        gmsh.model.geo.add_line(s, e, edges[(s, e)])

    loops = len(faces)
    for f, vis in enumerate(faces):
        loop = []
        for i in range(len(vis)):
            a = 1 + vis[i]
            b = 1 + vis[(i + 1) % len(vis)]
            edge = (a, b)
            if edge in edges:
                loop.append(edges[edge])
            else:
                loop.append(-edges[(b, a)])
        gmsh.model.geo.add_curve_loop(loop, f + 1)
        plane_tags = [f + 1]
        if holes:
            for hole in holes[f]:
                loop = []
                loops += 1
                for i in range(len(hole)):
                    a = 1 + hole[i]
                    b = 1 + hole[(i + 1) % len(hole)]
                    edge = (a, b)
                    if edge in edges:
                        loop.append(edges[edge])
                    else:
                        loop.append(-edges[(b, a)])
                plane_tags.append(loops)
                gmsh.model.geo.add_curve_loop(loop, loops)

        # add plane surface with possible holes (first element is the outer loop, rest are holes)
        gmsh.model.geo.add_plane_surface(plane_tags, f + 1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    _, V, _ = gmsh.model.mesh.getNodes()
    V = V.reshape(-1, 3)
    F = gmsh.model.mesh.getElementFaceNodes(2, 3).reshape(-1, 3) - 1
    return V, F

def circle_in_square(N=64, r=0.8, s=[-1, 1], e=0.025):
    verts = []
    faces = []
    # outer square
    verts.extend([[s[0], s[0], 0],
                  [s[1], s[0], 0],
                  [s[1], s[1], 0],
                  [s[0], s[1], 0]])
    faces.append([0, 1, 2, 3])
    # inner circle
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        verts.append([x, y, 0])
    hole = [i + 4 for i in range(N)]
    holes = [[hole]]
    # add the circle face
    faces.append(hole)
    holes.append([[]]) # no holes in circle face
    return mesh_surface(verts, faces, e_size=e, holes=holes)


def semicircles_in_square(N=64, r=0.8, s=[-1, 1], e=0.025):
    verts = []
    faces = []
    # outer square
    verts.extend([[s[0], s[0], 0],
                  [s[1], s[0], 0],
                  [s[1], s[1], 0],
                  [s[0], s[1], 0]])
    faces.append([0, 1, 2, 3])
    # inner circle
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        verts.append([x, y, 0])
    hole = [i + 4 for i in range(N)]
    holes = [[hole]]
    # add 2 semicircle faces
    semicircle = [i + 4 for i in range(N // 2)] + [4 + N // 2]
    faces.append(semicircle)
    holes.append([[]]) # no holes in semicircle face
    semicircle = [i + 4 for i in range(N // 2, N)] + [4]
    faces.append(semicircle)
    print(faces)
    holes.append([[]]) # no holes in semicircle face
    return mesh_surface(verts, faces, e_size=e, holes=holes)

# def split_circle(pieces):
#     # generate a circle split into `pieces` faces
#     # returns verts, [face1, face2, ...]


if __name__ == "__main__":
    V, F = semicircles_in_square()
    with open("test_s.obj", "w") as f:
        for v in V:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in F:
            f.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))