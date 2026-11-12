"""tailgap.py S BUILDDIR clip -> tail-scissors gap per stored key (and the stretches where it is under 3 units)"""
import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
B = 'assets/ice/wild/bum/characters/talon/skins/'
RIG = S + '/' + sys.argv[2] + '/wad/' + B
sk = L.load_skl(RIG + 'skin12/talon_skin12.skl'); m = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
def subset(joints, n=None):
    inf = np.array(sk.influences)[m.infl]; wsum = sum(((inf == sk.index[j]) * m.w).sum(1) for j in joints); sel = np.where(wsum > 0.5)[0]
    if n and len(sel) > n: sel = sel[np.random.default_rng(0).choice(len(sel), n, replace=False)]
    return inf[sel], m.w[sel], np.c_[m.pos[sel], np.ones(len(sel))]
def pts(sub, M):
    inf, w, P = sub; out = np.zeros((len(P), 3))
    for k in range(4): out += w[:, k, None] * np.einsum('nij,nj->ni', M[inf[:, k], :3, :], P)
    return out
TAIL = subset([f'tail{i}' for i in range(1, 17)], 900); SW = subset(['weapon'], 1200)
a = L.load_anm(RIG + 'skin12/animations/' + sys.argv[3]); g = []
for f in range(a.nframes):
    W = L.pose_world(sk, a, f); M = W @ sk.ibind; A = pts(TAIL, M); Bp = pts(SW, M); g.append(np.sqrt(((A[:, None] - Bp[None]) ** 2).sum(-1).min()))
g = np.array(g); print(' '.join(f'{x:.0f}' for x in g))
low = g < 3; print('keys under 3 units:', int(low.sum()), 'of', len(g), [int(i) for i in np.where(low)[0]])
