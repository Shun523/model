"""Chibi cat-ear maid figure generator.

Design taken from the reference illustration:
  - long straight dark-brown hair with blunt bangs and side locks
  - brown cat ears with pink insides, brown cat tail
  - white frilly maid headdress on a black band, black ribbon bows
  - purple eyes, glaring brows, small pouting mouth
  - black choker with a gold bell
  - black dress with puffed sleeves and corset, white blouse, white frilled apron

Run:  python3 catmaid.py
Writes ../ネコ耳メイド.3mf (one object, one part per colour) and preview.png.
Units: millimetres. The face points to -Y, the bed is z = 0.
"""
import math
import os
import uuid
import zipfile

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

SEG = 96

# ---------------------------------------------------------------- helpers


def ellipsoid(r, c=(0, 0, 0), seg=SEG):
    return Manifold.sphere(1.0, seg).scale(list(r)).translate(list(c))


def union(parts):
    parts = list(parts)
    return Manifold.batch_boolean(parts, OpType.Add) if parts else Manifold()


def hull(parts):
    return Manifold.batch_hull(list(parts))


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


def place(m, pos, normal, spin_deg=0.0, up=(0, 0, 1)):
    """Orient local z of `m` along `normal` (local y towards `up`), spin about it, move to `pos`."""
    m = m.rotate([0, 0, spin_deg])
    mat = np.column_stack([frame(normal, up), np.asarray(pos, float)])
    return m.transform(mat.tolist())


def tube(points, radii, seg=40):
    balls = [Manifold.sphere(r, seg).translate(list(p)) for p, r in zip(points, radii)]
    return union(hull([a, b]) for a, b in zip(balls, balls[1:]))


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


def torus(R, r, c=(0, 0, 0)):
    return Manifold.revolve(CrossSection.circle(r, 32).translate([R, 0]), SEG).translate(list(c))


def bow(pos, normal, scale=1.0):
    """Ribbon bow facing along `normal`: two loops, a knot and two hanging tails."""
    parts = []
    for s in (-1, 1):
        parts.append(ellipsoid((3.4, 2.3, 1.2), (0, 0, 0)).rotate([0, 0, s * 15]).translate([s * 3.0, 0.4, 0]))
        parts.append(ellipsoid((1.0, 3.0, 0.7), (0, 0, 0)).rotate([0, 0, s * 18]).translate([s * 1.6, -3.4, -0.2]))
    parts.append(ellipsoid((1.4, 1.6, 1.4), (0, 0, 0.4)))
    return place(union(parts).scale([scale] * 3), pos, normal)


# ---------------------------------------------------------------- head

HC = np.array([0.0, 0.0, 69.0])   # head centre
HR = np.array([19.5, 18.0, 18.5])  # head radii


def head_pt(az_deg, el_deg):
    """Point and outward normal on the head. az=0 is the face centre, + is the model's left (+x)."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    n = d / HR
    return HC + HR * d, n / np.linalg.norm(n)


def head_front_y(x, z):
    k = 1 - (x / HR[0]) ** 2 - ((z - HC[2]) / HR[2]) ** 2
    return HC[1] - HR[1] * math.sqrt(max(k, 0.02))


head = ellipsoid(HR, HC)
neck = Manifold.cylinder(10, 3.8, 3.8, 48).translate([0, 0.5, 44])

# face ------------------------------------------------------------
sclera, irises, pupils, highlights, lashes, brows = [], [], [], [], [], []
for s in (-1, 1):
    p, n = head_pt(s * 27, -12)
    sclera.append(place(ellipsoid((5.6, 4.2, 1.0)), p + n * 0.2, n))
    iris = place(ellipsoid((3.9, 4.6, 1.3)), p + n * 0.3, n)
    irises.append(iris ^ place(ellipsoid((5.6, 4.2, 3.0)), p, n))
    pupils.append(place(ellipsoid((1.5, 2.3, 1.0)), p + n * 1.05, n))
    highlights.append(place(Manifold.sphere(0.9, 32).translate([s * 1.4, 1.6, 1.4]), p, n))
    # heavy upper lid, outer end raised -> a glaring look
    lashes.append(place(ellipsoid((6.4, 1.3, 1.5)).rotate([0, 0, s * 10]).translate([0, 3.7, 0.6]), p, n))
    q, m = head_pt(s * 24, 7)
    brows.append(place(ellipsoid((4.6, 0.9, 1.0)), q + m * 0.3, m, s * 22))  # inner end low = frown

mouth_pts = []
for az, el in ((-3.5, -33.5), (0, -32), (3.5, -33.5)):  # middle raised: a small pout
    p, n = head_pt(az, el)
    mouth_pts.append(p + n * 0.15)
mouth = tube(catmull_rom(mouth_pts, 6), [0.55] * 13)

# ---------------------------------------------------------------- hair

cap = ellipsoid((20.8, 19.6, 20.0), HC + [0, 1.4, 1.0]) - Manifold.cube([60, 40, 40]).translate([-30, -44, 37])



def strand(x, zb, depth=0.0):
    """One bang strand hanging over the forehead, tapering to a point at height zb."""
    zm = (81 + zb) / 2
    top = (x * 0.95, head_front_y(x, 81) - 1.6 + depth, 81)
    mid = (x, head_front_y(x, zm) - 1.4 + depth, zm)
    tip = (x, head_front_y(x, zb) - 1.1 + depth, zb)
    return hull([ellipsoid((2.8, 1.4, 2.2), top), ellipsoid((2.0, 1.0, 1.5), mid), ellipsoid((0.8, 0.7, 0.8), tip)])


BANG_X = (-16.5, -12, -8, -4, 0, 4, 8, 12, 16.5)
BANG_Z = (63, 70.5, 72.5, 71.5, 69.5, 71.5, 72.5, 70.5, 63)
bangs = [strand(x, z) for x, z in zip(BANG_X, BANG_Z)]
# a second, slightly recessed layer fills the gaps between the front strands
bangs += [strand((x0 + x1) / 2, max(z0, z1) + 1.0, depth=0.5)
          for x0, x1, z0, z1 in zip(BANG_X, BANG_X[1:], BANG_Z, BANG_Z[1:])]

# long straight back hair: a bundle of strands ending in uneven points
back_hair = union(hull([ellipsoid((7.5, 8.5, 13), (x * 0.75, 9, 65)),
                        ellipsoid((3.2, 3.0, 2.5), (x, 12.5 + 0.02 * x * x, zb))])
                  for x, zb in ((-15, 31), (-10, 27.5), (-5, 29.5), (0, 26.5), (5, 29.5), (10, 27.5), (15, 31)))

side_locks = []
for s in (-1, 1):
    path = catmull_rom([(s * 16.5, -6, 70), (s * 15.5, -7.5, 58), (s * 12.5, -7.5, 49),
                        (s * 10, -6.5, 40), (s * 9.5, -5.5, 35)], 6)
    side_locks.append(tube(path, np.linspace(3.6, 1.6, len(path))))

ears_outer, ears_inner = [], []
for s in (-1, 1):
    outer = hull([ellipsoid((6.8, 2.6, 1.2)), Manifold.sphere(0.9, 24).translate([0, 0.8, 13])])
    inner = hull([ellipsoid((4.6, 1.0, 1.0), (0, -2.0, 1.8)), Manifold.sphere(0.6, 24).translate([0, -0.7, 10.5])])
    for lst, m in ((ears_outer, outer), (ears_inner, inner)):
        lst.append(m.rotate([0, s * 30, 0]).translate([s * 10.5, 2.0, 84]))

tail_path = catmull_rom([(-3, 13, 20), (-6, 24, 21), (-14, 29, 28), (-21, 28, 38),
                         (-24, 24, 48), (-22, 20, 57), (-17, 19, 62)], 8)
tail = tube(tail_path, np.linspace(3.4, 3.8, len(tail_path)))

# ---------------------------------------------------------------- headdress

frill, band_pts = [], []
for i, a in enumerate(np.linspace(-75, 75, 21)):
    ar = math.radians(a)
    radial = np.array([math.sin(ar), 0, math.cos(ar)])
    pos = HC + [21.3 * math.sin(ar), -3.5, 1.0 + 20.6 * math.cos(ar)]
    r = (2.6, 2.2, 3.0 if i % 2 else 2.6)  # alternate sizes for a ruffled edge
    frill.append(place(ellipsoid(r), pos, radial, up=(0, -1, 0)))
for a in np.linspace(-82, 82, 24):
    ar = math.radians(a)
    band_pts.append(HC + [20.4 * math.sin(ar), -1.6, 1.0 + 19.7 * math.cos(ar)])
headband = tube(band_pts, [1.3] * len(band_pts), 24)
head_bows = union(bow((s * 19.0, -7.0, 76), (s * 0.75, -0.66, 0), 0.9) for s in (-1, 1))

# ---------------------------------------------------------------- body

SKIRT = [(0, 0), (23, 0), (23.6, 1.5), (23.2, 4), (21.5, 10), (19, 16), (16, 22), (13.5, 27), (12, 30),
         (11.2, 32), (0, 32)]
skirt_r = lambda z: float(np.interp(z, [p[1] for p in SKIRT[1:-1]], [p[0] for p in SKIRT[1:-1]]))
skirt = Manifold.revolve(CrossSection([np.array(SKIRT, float)]), SEG)

petticoat = union(ellipsoid((3.0, 3.0, 2.6), (23.3 * math.cos(a), 23.3 * math.sin(a), 1.8))
                  for a in np.linspace(0, 2 * math.pi, 30, endpoint=False))

# apron: a shell over the front of the skirt
zs = np.linspace(5, 31, 14)
apron_profile = [(skirt_r(z) + 0.9, z) for z in zs] + [(skirt_r(z) - 1.5, z) for z in zs[::-1]]
wedge = CrossSection([np.array([(0, 0)] + [(60 * math.sin(math.radians(a)), -60 * math.cos(math.radians(a)))
                                          for a in np.linspace(-50, 50, 11)], float)]).extrude(40)
apron = Manifold.revolve(CrossSection([np.array(apron_profile)]), SEG) ^ wedge
apron_hem = union(place(ellipsoid((2.2, 1.8, 1.6)),
                        ((skirt_r(5) + 0.9) * math.sin(math.radians(a)), -(skirt_r(5) + 0.9) * math.cos(math.radians(a)), 5),
                        (math.sin(math.radians(a)), -math.cos(math.radians(a)), -0.3))
                  for a in np.linspace(-48, 48, 13))
a30 = math.radians(28)
apron_bow = bow(((skirt_r(19) + 1.4) * math.sin(a30), -(skirt_r(19) + 1.4) * math.cos(a30), 19),
                (math.sin(a30), -math.cos(a30), 0.35), 1.0)

corset = hull([ellipsoid((11.5, 9.5, 2.0), (0, 0, 31.5)), ellipsoid((10.2, 8.2, 2.0), (0, 0, 41))])
buttons = union(Manifold.sphere(1.1, 24).translate([s * 5.5, -7.9, z]) for s in (-1, 1) for z in (34, 38))
blouse = hull([ellipsoid((10.6, 8.6, 2.0), (0, 0, 41)), ellipsoid((12.0, 7.6, 2.5), (0, 0, 46.5))])
chest_bow = bow((0, -8.6, 43.5), (0, -1, 0.15), 0.85)

sleeves = union(ellipsoid((6.0, 6.0, 5.5), (s * 13, 0.5, 44.5)) for s in (-1, 1))
cuffs = union(torus(4.2, 1.4, (s * 14, 1, 39.8)) for s in (-1, 1))
arms = union(tube([(s * 14, 1, 40), (s * 9, 8, 28)], [2.6, 2.4]) for s in (-1, 1))  # hands behind the back

choker = torus(4.1, 1.1, (0, 0.5, 49.5))
bell = ellipsoid((1.9, 1.9, 1.9), (0, -5.4, 47.3)) - Manifold.cube([0.6, 3, 1.2]).translate([-0.3, -8, 45.6])

# ---------------------------------------------------------------- assemble colour parts (no overlaps)

COLOURS = {
    "skin": "#F7DCC8FF",
    "hair_brown": "#3B2420FF",
    "dress_black": "#1C1C1EFF",
    "frill_white": "#F7F5F0FF",
    "eye_purple": "#9B4FC8FF",
    "bell_gold": "#E0B020FF",
}

PRIORITY = [  # earlier entries win where volumes overlap
    ("frill_white", union(highlights)),
    ("dress_black", union(pupils)),
    ("eye_purple", union(irises)),
    ("dress_black", union([*lashes, mouth])),
    ("frill_white", union(sclera)),
    ("bell_gold", bell + buttons),
    ("dress_black", union([head_bows, chest_bow, apron_bow, choker, headband])),
    ("skin", union(ears_inner)),
    ("frill_white", union(frill)),
    ("hair_brown", union([*brows, cap, *bangs, back_hair, *side_locks, *ears_outer, tail])),
    ("frill_white", union([petticoat, apron, apron_hem, blouse, cuffs])),
    ("dress_black", union([skirt, corset, sleeves])),
    ("skin", union([head, neck, arms])),
]


def clean(m):
    return Manifold.compose([p for p in m.decompose() if p.volume() > 0.5])


def build_parts():
    occupied = Manifold()
    pieces = {k: [] for k in COLOURS}
    for key, m in PRIORITY:
        m = m.trim_by_plane([0, 0, 1], 0.0)
        pieces[key].append(m - occupied)
        occupied = occupied + m
    return [(k, COLOURS[k], clean(union(v))) for k, v in pieces.items()], occupied


PARTS, ALL = build_parts()

# ---------------------------------------------------------------- output


def mesh_arrays(m):
    mesh = m.as_original().simplify(0.001).to_mesh()  # as_original lets simplify merge across source meshes
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
        lines += [f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in v]
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

    fig = plt.figure(figsize=(13, 7), dpi=110)
    views = [(8, -90, "front"), (15, -55, "3/4"), (10, 90, "back")]
    for k, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        cam = np.array([math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
                        math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
                        math.sin(math.radians(elev))])
        light = cam + np.array([0, 0, 0.6]) + np.cross([0, 0, 1], cam) * 0.4
        light /= np.linalg.norm(light)
        tris, cols = [], []
        for _, colour, m in parts:
            v, t = mesh_arrays(m)
            tri = v[t]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            shade = 0.4 + 0.6 * np.clip(nrm @ light, 0, 1)
            base = np.array(matplotlib.colors.to_rgb(colour[:7]))
            tris.append(tri)
            cols.append(np.clip(base * shade[:, None] + 0.04, 0, 1))
        fc = np.concatenate(cols)
        ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolors=fc, edgecolors=fc, linewidths=0.1))
        lo, hi = np.array(bbox[:3]), np.array(bbox[3:])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(label)
    plt.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for name, _, m in PARTS:
        print(f"{name:12s} tris={m.num_tri():6d} vol={m.volume():9.1f} mm3 pieces={len(m.decompose())}")
    print("overall bbox:", np.round(ALL.bounding_box(), 1), "connected pieces:", len(ALL.decompose()))
    write_3mf(os.path.join(here, "..", "ネコ耳メイド.3mf"), PARTS, "ネコ耳メイド")
    render_preview(os.path.join(here, "preview.png"), PARTS, ALL.bounding_box())
