import sys, os, pickle, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
from glbanim import Glb
D = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'; B = 'assets/ice/wild/bum/characters/talon/skins/'
skS = L.load_skl(D + 'samira_skin30.skl'); mS = L.load_skn(D + 'samira_skin30.skn')
RIG = S + '/build22/wad/' + B; skN = L.load_skl(RIG + 'skin12/talon_skin12.skl'); mN = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
print('uv max diff', np.abs(mS.uv - mN.uv).max(), 'idx equal', np.array_equal(mS.idx, mN.idx))
def sim(P, Q):
    """similarity Q ~ s R P + t"""
    pc = P.mean(0); qc = Q.mean(0); A = P - pc; Bq = Q - qc; U, sg, Vt = np.linalg.svd(Bq.T @ A); d = np.sign(np.linalg.det(U @ Vt)); Dm = np.diag([1, 1, d]); R = U @ Dm @ Vt
    s = (sg * np.diag(Dm)).sum() / (A ** 2).sum(); t = qc - s * R @ pc; res = np.linalg.norm(Q - (s * P @ R.T + t), axis=1); return s, R, t, res
for n, vs, vc, i0, ic in mS.subs:
    s, R, t, res = sim(mS.pos[vs:vs + vc], mN.pos[vs:vs + vc]); print(f'{n:22s} scale {s:.4f} rot deg {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):6.1f} resid mean {res.mean():.2f} max {res.max():.2f}')
infS = np.array(skS.influences)[mS.infl]
for jn in ('Head', 'L_Foot', 'R_KneeLower', 'L_Elbow', 'R_Elbow', 'Spine1', 'Pelvis'):
    sel = np.where(((infS == skS.index[jn]) * mS.w).sum(1) > 0.9)[0]
    if len(sel) < 10: print(jn, 'few'); continue
    s, R, t, res = sim(mS.pos[sel], mN.pos[sel]); print(f'{jn:12s} n {len(sel):5d} scale {s:.4f} rot deg {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):6.1f} t {np.round(t, 1)} resid max {res.max():.2f}')
g = Glb(S + '/dl/samira30/model.glb')
for an in ('Recall', 'Recall_Winddown'):
    dur = g.duration(an); times = np.arange(int(round(dur * 30)) + 1) / 30; smp = g.sample(an, times)
    print(an, 'dur', round(dur, 3)); sc = {n: (v[2].min(), v[2].max()) for n, v in smp.items() if abs(v[2] - 1).max() > 0.02}; print('  scaled joints:', {k: (round(a, 2), round(b, 2)) for k, (a, b) in sc.items()})
    for n in ('Root', 'Sword', 'Sword_Back'):
        T = smp[n][0]; print('  ', n, 'local T at f0/60/120/150/180/240/250/260/last', np.round(T[[0, min(60, len(T) - 1), min(120, len(T) - 1), min(150, len(T) - 1), min(180, len(T) - 1), min(240, len(T) - 1), min(250, len(T) - 1), min(260, len(T) - 1), -1]], 0).tolist())
a = L.load_anm(S + '/build22/wad/' + B + 'skin12/animations/talon_recall.anm'); print('fps', repr(a.fps), 'frames', a.nframes)
