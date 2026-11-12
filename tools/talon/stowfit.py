"""stowfit.py S BUILDDIR [search] -> the recall while the scissors are stowed on her back: per key how many scissors vertices sit
inside the body / inside the tail, and how far the right hand is from the grip. With 'search': grid over a constant offset of the
stowed scissors in the chest's frame (roll about the back's normal, side / up / back shift) and the offsets that clear body and tail."""
import sys, os, itertools, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
from scipy.spatial import cKDTree
B = 'assets/ice/wild/bum/characters/talon/skins/'
RIG = S + '/' + sys.argv[2] + '/wad/' + B
sk = L.load_skl(RIG + 'skin12/talon_skin12.skl'); m = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
inf = np.array(sk.influences)[m.infl]; own = inf[np.arange(len(inf)), m.w.argmax(1)]
bw = np.linalg.inv(sk.ibind); JW = sk.index['weapon']; JC = sk.index['chest']; JRH = sk.index['r_hand']; TJ = [sk.index[f'tail{i}'] for i in range(1, 17)]
ws = np.where(own == JW)[0]; body = np.where((own != JW) & ~np.isin(own, TJ))[0]; tail = np.where(np.isin(own, TJ))[0]
rng = np.random.default_rng(0); wsub = ws[rng.choice(len(ws), 700, replace=False)]
def ortho(M): u, _, vt = np.linalg.svd(M[:3, :3]); return u @ vt
RcB = ortho(bw[JC]); a = L.load_anm(RIG + 'skin12/animations/talon_recall.anm')
def inside(X, tr, P, N): d, k = tr.query(X); return (((X - P[k]) * N[k]).sum(1) < 0) & (d < 14)
K0, K1 = 4, 112; rows = {}
for f in range(K0, K1 + 1):
    W = L.pose_world(sk, a, f); p, n = L.deform(m, sk, W); C = ortho(W[JC]) @ RcB.T
    rows[f] = (p, n, C, cKDTree(p[body]), cKDTree(p[tail]), np.linalg.norm(p[ws] - W[JRH][:3, 3], axis=1).min())
    X = p[ws]; print(f'key {f:3d}: inside body {int(inside(X, rows[f][3], p[body], n[body]).sum()):4d}  inside tail {int(inside(X, rows[f][4], p[tail], n[tail]).sum()):4d}  of {len(ws)}; right hand to scissors {rows[f][5]:5.1f}')
if len(sys.argv) > 3:
    keys = [f for f in range(12, 100, 4)]; res = []
    for th, dx, dy, dz in itertools.product(range(-30, 31, 6), range(-18, 19, 6), (0, 10, 20), range(0, 25, 4)):
        t = np.radians(th); Rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]); nb = []; nt = []
        for f in keys:
            p, n, C, tb, tt, _ = rows[f]; X0 = p[wsub]; pc = p[ws].mean(0); X = pc + (X0 - pc) @ (C @ Rz @ C.T).T + C @ np.array([dx, dy, -dz], float)
            nb.append(int(inside(X, tb, p[body], n[body]).sum())); nt.append(int(inside(X, tt, p[tail], n[tail]).sum()))
        res.append((max(nb) + max(nt), sum(nb) + sum(nt), abs(th) / 3 + abs(dx) / 3 + dy / 5 + dz / 4, th, dx, dy, dz, max(nb), max(nt)))
    res.sort()
    for r in res[:25]: print('worst key %3d  total %5d  size %5.1f  roll %4d side %4d up %3d back %3d   (worst body %d, worst tail %d; of 700 sampled)' % r)
