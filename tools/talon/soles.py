"""soles.py S BUILDDIR clipfile -> lowest sole point per key interval (8 sub-samples)"""
import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
B = 'assets/ice/wild/bum/characters/talon/skins/'; RIG = S + '/' + sys.argv[2] + '/wad/' + B; sk = L.load_skl(RIG + 'skin12/talon_skin12.skl'); m = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
a = L.load_anm(RIG + 'skin12/animations/' + sys.argv[3]); inf = np.array(sk.influences)[m.infl]; sel = np.where(sum(((inf == sk.index[n]) * m.w).sum(1) for n in ('l_foot', 'r_foot', 'l_toe', 'r_toe')) > 0.5)[0]
sub = 8; F = a.nframes; i = np.arange(sub * (F - 1) + 1); f0 = np.minimum(i // sub, F - 2); x = (i - f0 * sub) / sub; b = L.Anim(); b.hashes = a.hashes; b.sig = a.sig; b.has = a.has
b.T = (1 - x[:, None, None]) * a.T[f0] + x[:, None, None] * a.T[f0 + 1]; b.S = (1 - x[:, None, None]) * a.S[f0] + x[:, None, None] * a.S[f0 + 1]; b.R = L.slerp(a.R[f0], a.R[f0 + 1], np.repeat(x[:, None], a.R.shape[1], 1))
P = np.c_[m.pos[sel], np.ones(len(sel))]; y = []
for f in range(len(i)):
    M = L.pose_world(sk, b, f) @ sk.ibind; v = np.zeros(len(sel))
    for k in range(4): v += m.w[sel, k] * np.einsum('nj,nj->n', M[inf[sel, k], 1, :], P)
    y.append(v.min())
y = np.array(y); per = np.array([y[k * sub:(k + 1) * sub + 1].min() for k in range(F - 1)]); atk = y[::sub]
print('intervals below -1.5:', [(int(k), round(float(per[k]), 2), 'at keys', round(float(atk[k]), 2), round(float(atk[k + 1]), 2)) for k in np.where(per < -1.5)[0]])
