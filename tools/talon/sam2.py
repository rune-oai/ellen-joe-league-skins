"""sam2.py S out.png anim yaw nframes -> the Samira-mod Ellen mesh playing an animation of the downloaded Samira GLB"""
import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L, render as R
from glbanim import Glb
D = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'
sk = L.load_skl(D + 'samira_skin30.skl'); m = L.load_skn(D + 'samira_skin30.skn'); g = Glb(S + '/dl/samira30/model.glb')
# rest pose of the GLB vs the mod skeleton
gi = {n: i for i, n in enumerate(g.names)}; bw = np.linalg.inv(sk.ibind); gb = np.linalg.inv(g.ibm)
for n in ('Root', 'Head', 'L_Hand', 'R_Foot', 'Sword'):
    print(n, 'skl bind pos', np.round(bw[sk.index[n]][:3, 3], 2), 'glb bind pos', np.round(gb[gi[n]][:3, 3], 3), 'skl lt', np.round(sk.lt[sk.index[n]], 2), 'glb rest t', np.round(g.rest(g.joints[gi[n]])[0], 3))
name = sys.argv[3]; dur = g.duration(name); fps = 30.0; times = np.arange(int(round(dur * fps)) + 1) / fps; smp = g.sample(name, times)
print(name, 'duration', round(dur, 3), 'frames', len(times), 'root parent chain:', [g.nodes[p].get('name') for p in [g.parent.get(g.joints[gi['Root']])] if p is not None], [ (g.nodes[p].get('scale'), g.nodes[p].get('rotation')) for p in [g.parent.get(g.joints[gi['Root']])] if p is not None])
if len(sys.argv) > 2 and sys.argv[2] != '-':
    names = [n for n in sk.names if n in smp]; a = L.Anim(); a.hashes = np.array([L.elf(n) for n in names], np.uint32); a.sig = 'r3d2anmd'; a.has = np.ones((len(names), 3), bool)
    a.T = np.stack([smp[n][0] for n in names], 1); a.R = np.stack([smp[n][1] for n in names], 1); a.S = np.stack([smp[n][2] for n in names], 1); a.nframes = len(times); a.fps = fps
    import pickle; pickle.dump((names, a.T, a.R, a.S, fps), open(S + f'/dl/samira30_{name.lower()}.pkl', 'wb'))
    tex = {}
    for n, *_ in m.subs: tex[n] = None
    parts = [(i0 // 3, ic // 3, tex.get(n), (150, 150, 160) if n != 'sword' else (220, 80, 60)) for n, vs, vc, i0, ic in m.subs]
    nf = int(sys.argv[5]); ims = []; labels = []
    for f in np.linspace(0, len(times) - 1, nf).round().astype(int):
        w = L.pose_world(sk, a, f); p, nn = L.deform(m, sk, w); c = w[sk.index['Root'], :3, 3]
        gl = []
        for k in np.arange(-150, 151, 25): gl += [[c[0] + k, 0, c[2] - 150], [c[0] + k, 0, c[2] + 150], [c[0] - 150, 0, c[2] + k], [c[0] + 150, 0, c[2] + k]]
        ims.append(R.render(p, nn, m.uv, m.idx, parts, size=(300, 340), yaw=float(sys.argv[4]), pitch=10, center=(c[0], 110, c[2]), half_h=140, lines=gl)); labels.append(f'{name} f{f}/{len(times)}')
    R.sheet(ims, labels, cols=8).save(sys.argv[2]); print('saved', sys.argv[2])
