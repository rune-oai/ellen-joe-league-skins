"""tailpen.py S BUILDDIR clip [k0 k1 [step]] -> per key: how deep scissors vertices sit inside the tail (tail modelled as a tube
around its joint chain; radius per joint = median distance of the vertices it owns to the chain) and where the blade crosses"""
import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
B = 'assets/ice/wild/bum/characters/talon/skins/'
RIG = S + '/' + sys.argv[2] + '/wad/' + B
sk = L.load_skl(RIG + 'skin12/talon_skin12.skl'); m = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
inf = np.array(sk.influences)[m.infl]; own = inf[np.arange(len(inf)), m.w.argmax(1)]
bw = np.linalg.inv(sk.ibind); TJ = [sk.index[f'tail{i}'] for i in range(1, 17)]; C0 = np.array([bw[j][:3, 3] for j in TJ])
def chain_dist(X, C):
    best = np.full(len(X), 1e9); seg = np.zeros(len(X), int); tt = np.zeros(len(X))
    for i in range(len(C) - 1):
        d = C[i + 1] - C[i]; ln = np.linalg.norm(d); d = d / ln; t = np.clip((X - C[i]) @ d, 0, ln); dist = np.linalg.norm(X - (C[i] + np.outer(t, d)), axis=1)
        k = dist < best; best[k] = dist[k]; seg[k] = i; tt[k] = t[k] / ln
    return best, seg, tt
rad = np.zeros(16)
for i, j in enumerate(TJ):
    v = m.pos[own == j]
    if len(v): rad[i] = np.median(chain_dist(v, C0)[0])
print('tail tube radius per joint:', ' '.join(f'{r:.1f}' for r in rad))
ws = np.where(own == sk.index['weapon'])[0]; P = np.c_[m.pos[ws], np.ones(len(ws))]; JW = sk.index['weapon']; JH = sk.index['hip']
a = L.load_anm(RIG + 'skin12/animations/' + sys.argv[3]); k0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0; k1 = int(sys.argv[5]) if len(sys.argv) > 5 else a.nframes - 1
st = int(sys.argv[6]) if len(sys.argv) > 6 else 1; worst = []
for f in range(k0, k1 + 1, st):
    W = L.pose_world(sk, a, f); X = (P @ (W[JW] @ sk.ibind[JW]).T)[:, :3]; C = np.array([W[j][:3, 3] for j in TJ])
    dist, seg, tt = chain_dist(X, C); r = rad[seg] * (1 - tt) + rad[np.minimum(seg + 1, 15)] * tt; pen = r - dist; k = pen.argmax(); worst.append(pen[k])
    u, _, vt = np.linalg.svd(W[JH][:3, :3]); Rh = u @ vt; hb = np.linalg.svd(bw[JH][:3, :3]); Rb = hb[0] @ hb[2]
    loc = Rb.T @ Rh.T @ (X[k] - W[TJ[0]][:3, 3])      # deepest scissors vertex relative to the tail root, pelvis bind axes (x side, y up, z forward)
    print(f'key {f:3d}: deepest {pen[k]:6.1f} u inside at tail{seg[k] + 1} (+{tt[k]:.1f}); {int((pen > 0).sum()):4d} of {len(X)} scissors vertices inside; that vertex from the tail root: side {loc[0]:6.1f} up {loc[1]:6.1f} fwd {loc[2]:6.1f}')
worst = np.array(worst); print(f'keys with scissors inside the tail: {int((worst > 0).sum())} of {len(worst)}; deepest {worst.max():.1f}')
