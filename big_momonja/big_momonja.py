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

BC = np.array([0.0, 0.0, 29.0])   # body centre
BR = np.array([34.0, 31.0, 33.0])  # body radii (big and round: the body dominates the figure)
HEAD_TOP = BC[2] + BR[2]


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
    Manifold.batch_hull([ellipsoid((6.0, 6.0, 6.5), (s * 30.5, -3, 33)),
                         ellipsoid((5.3, 5.3, 5.3), (s * 34.0, -11, 19))])
    for s in (-1, 1))

# ---------------------------------------------------------------- face

patches, whites, pupils = [], [], []
for s in (-1, 1):
    p, n = surface(s * 22, 17)
    # tanuki-like ring around the eye
    patches.append(place(ellipsoid((9.2, 10.4, 2.8)), p - n * 1.4, n, s * 12))
    # slanted (tsurime) white of the eye
    whites.append(place(ellipsoid((6.4, 4.6, 2.2)), p + n * 1.3, n, s * 22))
    # dot pupil, nudged towards the nose
    q, m = surface(s * 20.5, 17)
    pupils.append(place(ellipsoid((2.1, 2.3, 2.1)), q + m * 3.0, m))

# duck bill: wide flat upper bill and a smaller lower bill
upper_bill = Manifold.batch_hull([ellipsoid((10.5, 4.5, 5.0), (0, -28.5, 26.0)),
                                  ellipsoid((11.5, 3.5, 3.3), (0, -41.0, 24.5))])
lower_bill = Manifold.batch_hull([ellipsoid((7.7, 4.4, 3.3), (0, -28.0, 16.6)),
                                  ellipsoid((8.6, 3.3, 2.4), (0, -38.5, 17.6))])
nostrils = union(ellipsoid((1.3, 1.8, 1.5), (s * 3.5, -38.0, 28.4)) for s in (-1, 1))
beak = (upper_bill + lower_bill) - nostrils

# ---------------------------------------------------------------- crown + mohawk

CC = np.array([0.0, -1.0, HEAD_TOP - 4.0])  # crown base centre (sunk into the head)
CR_OUT, CR_IN, CR_H = 11.5, 9.8, 10.5
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
    tuft((2.6, 3.9, 11.0), (0, -5.5, CC[2] + 8.0), -20),
    tuft((3.0, 4.6, 14.0), (0, -0.5, CC[2] + 10.0), 2),
    tuft((2.6, 3.9, 11.0), (0, 4.5, CC[2] + 8.0), 26),
    tuft((2.3, 3.4, 7.0), (0, 14.0, 58.5), 45),
    tuft((2.1, 3.0, 5.5), (0, 21.0, 53.0), 62),
    tuft((1.9, 2.6, 4.5), (0, 26.5, 46.0), 75),
])

# ---------------------------------------------------------------- feet


def foot(s):
    heel = (s * 13.0, -12.0)
    toes = []
    for k, spread in enumerate((-24, 0, 24)):
        a = math.radians(spread + s * 12)  # toes fan out, foot turned slightly outwards
        toes.append((heel[0] + 21 * math.sin(a), heel[1] - 21 * math.cos(a)))
    shape = Manifold.batch_hull([ellipsoid((6.3, 6.3, 4.0), (*heel, 2.0))]
                                + [ellipsoid((3.2, 3.2, 2.7), (x, y, 1.5)) for x, y in toes])
    # scallop the web between the toes
    for (x0, y0), (x1, y1) in zip(toes, toes[1:]):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = mx - heel[0], my - heel[1]
        L = math.hypot(dx, dy)
        shape -= Manifold.cylinder(10, 3.0, 3.0, 32).translate([mx + dx / L * 1.8, my + dy / L * 1.8, -2])
    return shape


feet = foot(-1) + foot(1)

# ---------------------------------------------------------------- tail

ctrl = [(0, 16, 6), (0, 28, 3.6), (0, 38, 3.6), (0, 46, 8), (1.2, 50, 17),
        (3, 51.5, 28), (4, 50, 39), (4, 44.5, 48)]
path = catmull_rom(ctrl, 8)
radii = np.linspace(4.1, 3.0, len(path))
tail = tube(path, radii)
end, prev = path[-1], path[-3]
direction = (end - prev) / np.linalg.norm(end - prev)
tail_tip = place(Manifold.batch_hull([ellipsoid((5.5, 5.5, 6.3)), Manifold.sphere(0.9, 24).translate([0, 0, 10.3])]),
                 end + direction * 1.5, direction)

# ---------------------------------------------------------------- assemble colour parts (no overlaps)

bed = lambda m: m.trim_by_plane([0, 0, 1], 0.0)  # keep z >= 0

black = bed(union(pupils))
red = bed(jewels)
gold = bed(crown - red)
eye_white = union(whites) - black
brown = bed(union([*patches, beak, mohawk, feet, tail_tip]) - black - red - gold - eye_white)
white = bed(union([body, arms, tail, eye_white]) - nostrils - black - red - gold - brown)



def clean(m):
    # drop zero-volume slivers the booleans can leave where several parts meet
    return Manifold.compose([p for p in m.decompose() if p.volume() > 0.5])


PARTS = [  # (name, colour, manifold)
    ("body_white", "#F4F1EAFF", clean(white)),
    ("chocolate", "#5A3220FF", clean(brown)),
    ("pupils_black", "#1A1A1AFF", clean(black)),
    ("crown_gold", "#E3B11BFF", clean(gold)),
    ("jewels_red", "#C8102EFF", clean(red)),
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


def render_preview(path, parts, bbox):
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
        lo, hi = np.array(bbox[:3]), np.array(bbox[3:])
        lo[2] = 0
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo))
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
    render_preview(os.path.join(here, "preview.png"), PARTS, allm.bounding_box())
