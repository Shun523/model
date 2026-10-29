"""ビッグももんじゃ (Dragon Quest Monsters: Caravan Heart) fan model generator.

Features (from web descriptions):
  - plump, round one-head-tall body covered in white fur
  - crown worn on the head (a nod to King Slime)
  - chocolate-coloured mohawk crest, duck-like bill and tail tip
  - tanuki-like markings around the eyes, slanted eyes with dot pupils
  - duck-like webbed feet, long monkey-like tail

Run:  python3 big_momonja.py
Writes ../ビッグももんじゃ.3mf (one object, one part per colour) and preview.png.
Units: millimetres. Front of the face points to -Y, the bed is z = 0.
"""
import math
import os
import uuid
import zipfile

import numpy as np
from manifold3d import Manifold, OpType

SEG = 96  # sphere resolution

# ---------------------------------------------------------------- helpers


def ellipsoid(r, c=(0, 0, 0), seg=SEG):
    return Manifold.sphere(1.0, seg).scale(list(r)).translate(list(c))


def union(parts):
    return Manifold.batch_boolean(list(parts), OpType.Add)


def frame(normal, up=(0, 0, 1)):
    """Rotation matrix whose z-axis is `normal` and whose y-axis is as close to `up` as possible."""
    z = np.asarray(normal, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def place(m, pos, normal, spin_deg=0.0):
    """Orient local z of `m` along `normal`, rotate it about that axis by spin, move to `pos`."""
    m = m.rotate([0, 0, spin_deg])
    rot = frame(normal)
    mat = np.column_stack([rot, np.asarray(pos, float)])
    return m.transform(mat.tolist())


def tube(points, radii, seg=48):
    """Chain of hulled spheres along a polyline (a tapered round tube)."""
    balls = [Manifold.sphere(r, seg).translate(list(p)) for p, r in zip(points, radii)]
    return union(Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:]))


def catmull_rom(ctrl, n_per_seg=8):
    ctrl = np.asarray(ctrl, float)
    pts = [ctrl[0], *ctrl, ctrl[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for t in np.linspace(0, 1, n_per_seg, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    out.append(ctrl[-1])
    return np.array(out)


# ---------------------------------------------------------------- body

BC = np.array([0.0, 0.0, 22.0])   # body centre
BR = np.array([26.0, 24.0, 25.0])  # body radii


def surface(az_deg, el_deg):
    """Point and outward normal on the body ellipsoid. az=0 is the face centre, + is the model's left."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    p = BC + BR * d
    n = d / BR
    return p, n / np.linalg.norm(n)


body = ellipsoid(BR, BC)

# stubby arms hanging at the sides
arms = union(
    Manifold.batch_hull([ellipsoid((5.5, 5.5, 6.0), (s * 22.5, -2, 25)),
                         ellipsoid((4.8, 4.8, 4.8), (s * 25.5, -9, 14))])
    for s in (-1, 1))

# ---------------------------------------------------------------- face

patches, whites, pupils = [], [], []
for s in (-1, 1):
    p, n = surface(s * 23, 17)
    # tanuki-like ring around the eye
    patches.append(place(ellipsoid((8.0, 9.0, 2.6)), p - n * 1.3, n, s * 12))
    # slanted (tsurime) white of the eye
    whites.append(place(ellipsoid((5.6, 4.0, 2.0)), p + n * 1.2, n, s * 22))
    # dot pupil, nudged towards the nose
    q, m = surface(s * 21.5, 17)
    pupils.append(place(ellipsoid((1.8, 2.0, 1.8)), q + m * 2.6, m))

# duck bill: wide flat upper bill and a smaller lower bill
upper_bill = Manifold.batch_hull([ellipsoid((9.5, 4, 4.6), (0, -21.5, 18.5)),
                                  ellipsoid((10.5, 3.2, 3.0), (0, -33, 17.0))])
lower_bill = Manifold.batch_hull([ellipsoid((7.0, 4, 3.0), (0, -21.0, 10.2)),
                                  ellipsoid((7.8, 3.0, 2.2), (0, -30.5, 10.4))])
nostrils = union(ellipsoid((1.2, 1.6, 1.4), (s * 3.2, -29.5, 20.2)) for s in (-1, 1))
beak = (upper_bill + lower_bill) - nostrils

# ---------------------------------------------------------------- crown + mohawk

CC = np.array([0.0, -1.0, 43.0])  # crown base centre (sunk into the head)
CR_OUT, CR_IN, CR_H = 10.5, 8.8, 10.0
crown = (Manifold.cylinder(CR_H, CR_OUT, CR_OUT, SEG) - Manifold.cylinder(CR_H + 2, CR_IN, CR_IN, SEG).translate([0, 0, -1]))
spikes = []
wall_r = (CR_OUT + CR_IN) / 2
for k in range(5):
    a = math.radians(-90 + k * 72)  # first spike straight to the front
    spike = Manifold.cylinder(6.0, 2.8, 0.5, 32).scale([0.55, 1, 1])  # thinned radially (local x)
    spike = spike.rotate([0, 0, math.degrees(a)]).translate([wall_r * math.cos(a), wall_r * math.sin(a), CR_H])
    ball = Manifold.sphere(1.5, 32).translate([wall_r * math.cos(a), wall_r * math.sin(a), CR_H + 6.3])
    spikes += [spike, ball]
crown = union([crown, *spikes]).translate(CC.tolist())

jewels = []
for deg in (-90, -90 - 50, -90 + 50):  # front and the two front sides
    a = math.radians(deg)
    pos = CC + [CR_OUT * math.cos(a), CR_OUT * math.sin(a), CR_H * 0.55]
    jewels.append(place(ellipsoid((2.0, 2.6, 1.6)), pos, [math.cos(a), math.sin(a), 0]))
jewels = union(jewels)

# mohawk tufts poking up through the crown and running down the back of the head


def tuft(r, pos, tilt_deg):
    # leaf/flame shape: ellipsoid sharpened at the top by hulling with a small tip sphere
    leaf = Manifold.batch_hull([ellipsoid(r, (0, 0, 0)), Manifold.sphere(0.6, 24).translate([0, 0, r[2] * 1.25])])
    return leaf.rotate([tilt_deg, 0, 0]).translate(list(pos))


mohawk = union([
    tuft((2.3, 3.4, 9.5), (0, -5.0, 51.0), -20),
    tuft((2.6, 4.0, 12.0), (0, -0.5, 53.0), 2),
    tuft((2.3, 3.4, 9.5), (0, 4.0, 51.0), 26),
    tuft((2.1, 3.0, 6.0), (0, 11.5, 44.5), 48),
    tuft((1.9, 2.6, 4.5), (0, 16.5, 39.0), 68),
])

# ---------------------------------------------------------------- feet


def foot(s):
    heel = (s * 11.0, -8.0)
    toes = []
    for k, spread in enumerate((-24, 0, 24)):
        a = math.radians(spread + s * 12)  # toes fan out, foot turned slightly outwards
        toes.append((heel[0] + 19 * math.sin(a), heel[1] - 19 * math.cos(a)))
    shape = Manifold.batch_hull([ellipsoid((5.5, 5.5, 3.6), (*heel, 1.8))]
                                + [ellipsoid((2.8, 2.8, 2.4), (x, y, 1.4)) for x, y in toes])
    # scallop the web between the toes
    for (x0, y0), (x1, y1) in zip(toes, toes[1:]):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = mx - heel[0], my - heel[1]
        L = math.hypot(dx, dy)
        shape -= Manifold.cylinder(10, 2.6, 2.6, 32).translate([mx + dx / L * 1.6, my + dy / L * 1.6, -2])
    return shape


feet = foot(-1) + foot(1)

# ---------------------------------------------------------------- tail

ctrl = [(0, 12, 6), (0, 24, 3.4), (0, 33, 3.4), (0, 40, 7), (1, 44, 15),
        (2.5, 45, 25), (3.5, 43, 34), (3.5, 38.5, 40)]
path = catmull_rom(ctrl, 8)
radii = np.linspace(3.6, 2.6, len(path))
tail = tube(path, radii)
end, prev = path[-1], path[-3]
direction = (end - prev) / np.linalg.norm(end - prev)
tail_tip = place(Manifold.batch_hull([ellipsoid((4.8, 4.8, 5.5)), Manifold.sphere(0.8, 24).translate([0, 0, 9.0])]),
                 end + direction * 1.5, direction)

# ---------------------------------------------------------------- assemble colour parts (no overlaps)

bed = lambda m: m.trim_by_plane([0, 0, 1], 0.0)  # keep z >= 0

black = bed(union(pupils))
red = bed(jewels)
gold = bed(crown - red)
eye_white = union(whites) - black
brown = bed(union([*patches, beak, mohawk, feet, tail_tip]) - black - red - gold - eye_white)
white = bed(union([body, arms, tail, eye_white]) - nostrils - black - red - gold - brown)

PARTS = [  # (name, colour, manifold)
    ("body_white", "#F4F1EAFF", white),
    ("chocolate", "#5A3220FF", brown),
    ("pupils_black", "#1A1A1AFF", black),
    ("crown_gold", "#E3B11BFF", gold),
    ("jewels_red", "#C8102EFF", red),
]

# ---------------------------------------------------------------- output


def mesh_arrays(m):
    mesh = m.simplify(0.001).to_mesh()  # drop sliver edges left by the booleans
    return np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)


def write_3mf(path, parts, title):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<model unit="millimeter" xml:lang="ja-JP" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">',
             f' <metadata name="Title">{title}</metadata>',
             ' <resources>',
             '  <m:colorgroup id="1">']
    lines += [f'   <m:color color="{c}"/>' for _, c, _ in parts]
    lines.append('  </m:colorgroup>')
    for i, (name, _, m) in enumerate(parts):
        v, t = mesh_arrays(m)
        lines.append(f'  <object id="{i + 2}" name="{name}" type="model" pid="1" pindex="{i}"><mesh><vertices>')
        lines += [f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in v]
        lines.append('</vertices><triangles>')
        lines += [f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in t]
        lines.append('</triangles></mesh></object>')
    asm = len(parts) + 2
    lines.append(f'  <object id="{asm}" name="{title}" type="model"><components>')
    lines += [f'   <component objectid="{i + 2}"/>' for i in range(len(parts))]
    lines += ['  </components></object>', ' </resources>',
              f' <build><item objectid="{asm}" transform="1 0 0 0 1 0 0 0 1 128 128 0"/></build>', '</model>']
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                     '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Target="/3D/3dmodel.model" Id="rel-{uuid.uuid4().hex[:8]}" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", "\n".join(lines))


def render_preview(path, parts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    light = np.array([-0.4, -0.8, 0.6])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(12, 6), dpi=110)
    views = [(15, -90, "front"), (20, -45, "3/4"), (12, 0, "side")]
    for k, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        tris, cols = [], []
        for _, colour, m in parts:
            v, t = mesh_arrays(m)
            tri = v[t]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            # light follows the camera so every view is lit from the front
            cam = np.array([math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
                            math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
                            math.sin(math.radians(elev))])
            l = cam + np.array([0, 0, 0.6]) + np.cross([0, 0, 1], cam) * 0.4
            l /= np.linalg.norm(l)
            shade = 0.35 + 0.65 * np.clip(nrm @ l, 0, 1)
            base = np.array(matplotlib.colors.to_rgb(colour[:7]))
            tris.append(tri)
            cols.append(base * shade[:, None])
        fc = np.concatenate(cols)
        ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolors=fc, edgecolors=fc, linewidths=0.1))
        ax.set_xlim(-35, 35); ax.set_ylim(-38, 50); ax.set_zlim(0, 64)
        ax.set_box_aspect((70, 88, 64))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(label)
    fig.suptitle("Big Momonja", fontsize=14)
    plt.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for name, _, m in PARTS:
        lo, hi = np.array(m.bounding_box()[:3]), np.array(m.bounding_box()[3:])
        print(f"{name:13s} tris={m.num_tri():6d} vol={m.volume():9.1f} mm3 genus={m.genus()} "
              f"bbox={np.round(lo, 1)}..{np.round(hi, 1)}")
    allm = union([body, arms, tail, beak, mohawk, feet, tail_tip, crown, jewels, *patches, *whites, *pupils])
    print("overall bbox:", np.round(allm.bounding_box(), 1), "connected pieces:", len(allm.decompose()))
    write_3mf(os.path.join(here, "..", "ビッグももんじゃ.3mf"), PARTS, "ビッグももんじゃ")
    render_preview(os.path.join(here, "preview.png"), PARTS)
