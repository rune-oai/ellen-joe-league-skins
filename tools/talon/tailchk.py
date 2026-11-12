"""tailchk.py S BUILDDIR [clip,clip] -> per clip: how far the tail tip travels (in the pelvis' frame, % of tail length) and the
smallest gap between tail and scissors over the interpolated motion (4 sub-samples per key interval)"""
import sys, os, glob, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
B = 'assets/ice/wild/bum/characters/talon/skins/'
RIG = S + '/' + sys.argv[2] + '/wad/' + B
sk = L.load_skl(RIG + 'skin12/talon_skin12.skl'); m = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
SUB = 4
def subset(joints, n=None):
    inf = np.array(sk.influences)[m.infl]; wsum = sum(((inf == sk.index[j]) * m.w).sum(1) for j in joints); sel = np.where(wsum > 0.5)[0]
    if n and len(sel) > n: sel = sel[np.random.default_rng(0).choice(len(sel), n, replace=False)]
    return inf[sel], m.w[sel], np.c_[m.pos[sel], np.ones(len(sel))]
def pts(sub, M):
    inf, w, P = sub; out = np.zeros((len(P), 3))
    for k in range(4): out += w[:, k, None] * np.einsum('nij,nj->ni', M[inf[:, k], :3, :], P)
    return out
TAIL = subset([f'tail{i}' for i in range(1, 17)], 500); SW = subset(['weapon'], 700)
JT1 = sk.index['tail1']; JT16 = sk.index['tail16']; JH = sk.index['hip']
bw = np.linalg.inv(sk.ibind); tl = float(np.linalg.norm(np.diff(np.array([bw[sk.index[f'tail{i}']][:3, 3] for i in range(1, 17)]), axis=0), axis=1).sum())
def upsample(a, K):
    F = a.nframes; i = np.arange(K * (F - 1) + 1); f0 = np.minimum(i // K, F - 2); x = (i - f0 * K) / K; b = L.Anim(); b.hashes = a.hashes; b.sig = a.sig; b.has = a.has
    b.T = (1 - x[:, None, None]) * a.T[f0] + x[:, None, None] * a.T[f0 + 1]; b.S = (1 - x[:, None, None]) * a.S[f0] + x[:, None, None] * a.S[f0 + 1]
    b.R = L.slerp(a.R[f0], a.R[f0 + 1], np.repeat(x[:, None], a.R.shape[1], 1)); return b, len(i)
print(f'tail length {tl:.1f} units')
print(f'{"clip":38s} keys  tip travel side/vert (% of tail length)  max tip speed (lengths/s)   tail-scissors gap min (at key)')
for p in sorted(glob.glob(RIG + 'skin12/animations/*.anm')):
    nm = os.path.basename(p)[:-4]
    if only and not any(o in nm for o in only): continue
    a = L.load_anm(p); b, n = upsample(a, SUB); tip = np.zeros((n, 3)); gap = np.zeros(n)
    for f in range(n):
        W = L.pose_world(sk, b, f); M = W @ sk.ibind
        u, _, vt = np.linalg.svd(W[JH][:3, :3]); Rh = u @ vt
        tip[f] = Rh.T @ (W[JT16][:3, 3] - W[JT1][:3, 3])
        A = pts(TAIL, M); Bp = pts(SW, M); d2 = ((A[:, None] - Bp[None]) ** 2).sum(-1); gap[f] = np.sqrt(d2.min())
    hb = np.linalg.svd(bw[JH][:3, :3]); Rb = hb[0] @ hb[2]; tb = tip @ Rb      # bind axes of the pelvis: x side, y up, z forward
    sp = np.linalg.norm(np.diff(tip, axis=0), axis=1) * a.fps * SUB / tl
    print(f'{nm:38s} {a.nframes:4d}  {np.ptp(tb[:, 0]) / tl * 100:5.1f} / {np.ptp(tb[:, 1]) / tl * 100:5.1f}                         {sp.max():5.2f}                   {gap.min():6.1f} (f{gap.argmin() / SUB:.1f})')
