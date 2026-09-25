"""Export Track.blend to the USD + JSON files the simulator loads.

Run headless from the Sim/ folder (or use scripts/export_track.sh):

    blender -b assets/blender/Track.blend --python assets/blender/export_track.py -- --out assets/export

What is in Track.blend
    Track       Bezier curve + geometry nodes: resamples `TrackCurve`, offsets it to both sides and
                instances the `Cylinder` object on the points, then REALIZES the instances. A
                plain export would therefore merge all ~310 track objects into one mesh.
    MainWalls   Same idea, but extrudes a 0.4 m line profile along both offset curves (+ solidify,
                bevel). Collision only: the sim hides it from every camera and ends the run on
                contact, so touching it means the algorithm let the car leave the corridor.
    TrackCurve  The centre line (closed Bezier). Used for spawning, progress and lap timing.

What this writes (Z up, Y forward - Blender's own frame, so no axis conversion happens)
    track_objects.usdc   /Track/Cylinder_###   one Mesh prim per track object, origin at the base
                         centre, so the simulator can label each one and produce a bounding box
                         per object from the renderer's instance segmentation (no YOLO needed).
    walls.usdc           /Walls/Walls          the realized wall mesh.
    track.json           per-object position / size / side / arc length, the centre line and a
                         few corridor statistics. The simulator reads object geometry from here.

The split works on the evaluated (realized) mesh: every track object is one connected component,
so each loose part becomes its own object. Nothing in Track.blend is modified or saved.
"""
import argparse
import datetime
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

TRACK_OBJ, WALLS_OBJ, CENTERLINE_OBJ = "Track", "MainWalls", "TrackCurve"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="export_track.py")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
    ap.add_argument("--prefix", default="Object", help="prim name prefix for the track objects")
    return ap.parse_args(argv)


def evaluated_mesh(obj):
    """World-space copy of an object's fully evaluated mesh (modifiers + geometry nodes)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    me.transform(obj.matrix_world)
    return me


def loose_parts(me):
    """Split a mesh into connected components -> [(verts [Vector], faces [[local idx]], material idx)]."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    comp_of = [-1] * len(bm.verts)
    n = 0
    for v in bm.verts:
        if comp_of[v.index] != -1:
            continue
        stack = [v]
        comp_of[v.index] = n
        while stack:
            x = stack.pop()
            for e in x.link_edges:
                y = e.other_vert(x)
                if comp_of[y.index] == -1:
                    comp_of[y.index] = n
                    stack.append(y)
        n += 1
    parts = [{"verts": [], "remap": {}, "faces": [], "mat": 0} for _ in range(n)]
    for v in bm.verts:
        p = parts[comp_of[v.index]]
        p["remap"][v.index] = len(p["verts"])
        p["verts"].append(v.co.copy())
    for f in bm.faces:
        p = parts[comp_of[f.verts[0].index]]
        p["faces"].append([p["remap"][v.index] for v in f.verts])
        p["mat"] = f.material_index
    bm.free()
    return parts


def centerline(obj):
    """Ordered XY points of the evaluated centre-line curve and whether it is closed."""
    me = evaluated_mesh(obj)
    adj = {i: [] for i in range(len(me.vertices))}
    for e in me.edges:
        a, b = e.vertices
        adj[a].append(b)
        adj[b].append(a)
    start = next((i for i, nb in adj.items() if len(nb) == 1), 0)   # open curve: start at an end
    order, prev, cur = [start], None, start
    while True:
        nxt = [j for j in adj[cur] if j != prev]
        if not nxt or nxt[0] == start:
            break
        prev, cur = cur, nxt[0]
        order.append(cur)
    pts = [(me.vertices[i].co.x, me.vertices[i].co.y) for i in order]
    closed = all(len(nb) == 2 for nb in adj.values())
    bpy.data.meshes.remove(me)
    return pts, closed


def project(pts, closed, x, y):
    """(arc length s, signed lateral offset, +left of the travel direction) of point (x, y)."""
    best, s_acc, res = None, 0.0, (0.0, 0.0)
    n = len(pts)
    for i in range(n if closed else n - 1):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        L = math.sqrt(L2)
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2)) if L2 > 0 else 0.0
        px, py = ax + t * dx, ay + t * dy
        d2 = (x - px) ** 2 + (y - py) ** 2
        if best is None or d2 < best:
            best = d2
            cross = dx * (y - ay) - dy * (x - ax)
            res = (s_acc + t * L, math.copysign(math.sqrt(d2), cross))
        s_acc += L
    return res


def polyline_length(pts, closed):
    n = len(pts)
    return sum(math.dist(pts[i], pts[(i + 1) % n]) for i in range(n if closed else n - 1))


def export_usd(filepath, objects, root, materials):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.wm.usd_export(
        filepath=filepath,
        check_existing=False,
        selected_objects_only=True,
        export_animation=False,
        export_materials=materials,
        generate_preview_surface=materials,
        export_textures_mode="KEEP",
        export_lights=False,
        export_cameras=False,
        export_curves=False,
        export_uvmaps=False,
        use_instancing=False,
        evaluation_mode="RENDER",
        convert_orientation=True,               # explicit: Z up, Y forward (identity from Blender)
        export_global_forward_selection="Y",
        export_global_up_selection="Z",
        convert_scene_units="METERS",
        root_prim_path=root,
        merge_parent_xform=True,                # /Track/Cylinder_000 is the Mesh itself
        export_custom_properties=True,
        xform_op_mode="TRS",
    )


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    for name in (TRACK_OBJ, WALLS_OBJ, CENTERLINE_OBJ):
        if name not in bpy.data.objects:
            raise SystemExit(f"[export] '{name}' not found in {bpy.data.filepath}")
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    cl, closed = centerline(bpy.data.objects[CENTERLINE_OBJ])
    cl_len = polyline_length(cl, closed)
    print(f"[export] centre line: {len(cl)} points, {'closed' if closed else 'open'}, {cl_len:.2f} m")

    # All exported objects live in a throw-away collection; the .blend is never saved.
    col = bpy.data.collections.new("EXPORT")
    bpy.context.scene.collection.children.link(col)

    # ---- track objects: one object per connected component -------------------------------
    src = bpy.data.objects[TRACK_OBJ]
    track_me = evaluated_mesh(src)
    parts = loose_parts(track_me)
    mats = list(track_me.materials)
    info = []
    for p in parts:
        vs = p["verts"]
        mn = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
        mx = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
        base = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
        s, lat = project(cl, closed, base.x, base.y)
        info.append({"part": p, "base": base, "radius": max(mx.x - mn.x, mx.y - mn.y) / 2,
                     "height": mx.z - mn.z, "s": s, "lateral": lat})
    info.sort(key=lambda d: (d["s"], d["lateral"]))

    objects, records = [], []
    for i, d in enumerate(info):
        name = f"{args.prefix}_{i:03d}"
        p = d["part"]
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v - d["base"]) for v in p["verts"]], [], p["faces"])
        me.validate()
        if mats:
            me.materials.append(mats[min(p["mat"], len(mats) - 1)])
        for poly in me.polygons:
            poly.use_smooth = True
        ob = bpy.data.objects.new(name, me)
        ob.location = d["base"]
        side = "left" if d["lateral"] > 0 else "right"
        ob["side"] = side
        ob["track_id"] = i
        col.objects.link(ob)
        objects.append(ob)
        records.append({
            "id": i, "name": name, "prim": f"/Track/{name}",
            "x": round(d["base"].x, 5), "y": round(d["base"].y, 5), "z0": round(d["base"].z, 5),
            "radius": round(d["radius"], 5), "height": round(d["height"], 5),
            "side": side, "s": round(d["s"], 4), "lateral": round(d["lateral"], 4),
        })
    bpy.data.meshes.remove(track_me)
    print(f"[export] track objects: {len(records)} "
          f"({sum(r['side'] == 'left' for r in records)} left / {sum(r['side'] == 'right' for r in records)} right), "
          f"radius {records[0]['radius']:.4f} m, height {records[0]['height']:.4f} m")

    # ---- walls: one realized mesh ------------------------------------------------------------
    walls_me = evaluated_mesh(bpy.data.objects[WALLS_OBJ])
    walls_me.name = "Walls"
    walls_ob = bpy.data.objects.new("Walls", walls_me)
    col.objects.link(walls_ob)
    wz = [v.co.z for v in walls_me.vertices]
    wall_lat = {"left": [], "right": []}
    for v in list(walls_me.vertices)[:: max(1, len(walls_me.vertices) // 1500)]:
        _, lat = project(cl, closed, v.co.x, v.co.y)
        wall_lat["left" if lat > 0 else "right"].append(abs(lat))
    print(f"[export] walls: {len(walls_me.vertices)} verts, height {max(wz) - min(wz):.3f} m")

    # ---- write -------------------------------------------------------------------------------
    export_usd(os.path.join(out, "track_objects.usdc"), objects, "/Track", materials=True)
    export_usd(os.path.join(out, "walls.usdc"), [walls_ob], "/Walls", materials=False)

    def stats(xs):
        return {"min": round(min(xs), 4), "max": round(max(xs), 4), "mean": round(sum(xs) / len(xs), 4)} if xs else None

    obj_lat = {side: [abs(r["lateral"]) for r in records if r["side"] == side] for side in ("left", "right")}
    by_side = {side: sorted(r["s"] for r in records if r["side"] == side) for side in ("left", "right")}
    spacing = {side: stats([b - a for a, b in zip(v, v[1:])]) for side, v in by_side.items()}
    manifest = {
        "source": os.path.basename(bpy.data.filepath),
        "exported": datetime.datetime.now().isoformat(timespec="seconds"),
        "blender": bpy.app.version_string,
        "frame": {"up": "Z", "forward": "Y", "units": "m"},
        "files": {"objects": "track_objects.usdc", "walls": "walls.usdc"},
        "object_class": "apple",
        "objects": records,
        "walls": {"prim": "/Walls", "height": round(max(wz) - min(wz), 4),
                  "lateral": {k: stats(v) for k, v in wall_lat.items()}},
        "centerline": {"closed": closed, "length_m": round(cl_len, 4),
                       "points": [[round(x, 5), round(y, 5)] for x, y in cl]},
        "stats": {"object_lateral": {k: stats(v) for k, v in obj_lat.items()},
                  "object_spacing_along_track": spacing},
    }
    with open(os.path.join(out, "track.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[export] object lateral offset: {manifest['stats']['object_lateral']}")
    print(f"[export] wall lateral offset:   {manifest['walls']['lateral']}")
    print(f"[export] object spacing:        {spacing}")
    print(f"[export] wrote {out}/track_objects.usdc, walls.usdc, track.json")


if __name__ == "__main__":
    main()
