"""g1.py S mod out.png yaw pitch half_h "Clip:f,f,f;Clip:f,f" -> frames with a ground grid at y=0 (camera fixed on the root)"""
import sys, numpy as np
S = sys.argv[1]; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L, render as R
from animsheet import setup, B, CL
mod = sys.argv[2]; root, sk, m, tex = setup(mod); yaw = float(sys.argv[4]); pitch = float(sys.argv[5]); HH = float(sys.argv[6])
parts = [(i0//3, ic//3, tex.get(n), (200, 200, 200)) for n, vs, vc, i0, ic in m.subs]
cl = dict(CL); ims = []; labels = []
for spec in sys.argv[7].split(';'):
    name, fr = spec.split(':'); rel = cl[name]
    if mod == 'new': rel = rel.replace('skin20/animations/talon_skin20_spell3_long_healthbar_fixed', 'skin12/animations/talon_spell3_long').replace('skin20/animations/talon_skin20_spell3_medium_healthbar_fixed', 'skin12/animations/talon_spell3_medium')
    a = L.load_anm(root + B + rel)
    frs = [int(x) for x in fr.split(',')] if fr != '*' else list(np.linspace(0, a.nframes - 1, 6).round().astype(int))
    for f in frs:
        f = min(f, a.nframes - 1); w = L.pose_world(sk, a, f); p, n = L.deform(m, sk, w); c = w[sk.index['root'], :3, 3] if len(sys.argv) < 9 else np.array([float(v) for v in sys.argv[8].split(',')])
        g = []
        for k in np.arange(-150, 151, 25):
            g += [[c[0] + k, 0, c[2] - 150], [c[0] + k, 0, c[2] + 150], [c[0] - 150, 0, c[2] + k], [c[0] + 150, 0, c[2] + k]]
        # hide what is below the ground: clip verts is not possible cheaply, so draw the grid on top
        ims.append(R.render(p, n, m.uv, m.idx, parts, size=(380, 420), yaw=yaw, pitch=pitch, center=(c[0], HH - 25 + max(0.0, c[1] - 110.0), c[2]), half_h=HH, lines=g)); labels.append(f'{name} f{f}/{a.nframes}')
R.sheet(ims, labels, cols=6).save(sys.argv[3]); print('saved', sys.argv[3])
