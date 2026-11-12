import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
B = 'assets/ice/wild/bum/characters/talon/skins/'; RIG = S + '/' + sys.argv[2] + '/wad/' + B; sk = L.load_skl(RIG + 'skin12/talon_skin12.skl'); m = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
a = L.load_anm(RIG + 'skin12/animations/talon_recall.anm'); sw = next((vs, vc) for n, vs, vc, i0, ic in m.subs if n == 'sword'); JW = sk.index['weapon']
V = (np.c_[m.pos[sw[0]:sw[0] + sw[1]], np.ones(sw[1])] @ sk.ibind[JW].T); sub = 8; F = a.nframes
i = np.arange(sub * (F - 1) + 1); f0 = np.minimum(i // sub, F - 2); x = (i - f0 * sub) / sub; b = L.Anim(); b.hashes = a.hashes; b.sig = a.sig; b.has = a.has
b.T = (1 - x[:, None, None]) * a.T[f0] + x[:, None, None] * a.T[f0 + 1]; b.S = (1 - x[:, None, None]) * a.S[f0] + x[:, None, None] * a.S[f0 + 1]; b.R = L.slerp(a.R[f0], a.R[f0 + 1], np.repeat(x[:, None], a.R.shape[1], 1))
y = np.array([(V @ L.pose_world(sk, b, f)[JW].T)[:, 1].min() for f in range(int(sys.argv[3]) * sub, int(sys.argv[4]) * sub + 1)])
k = int(sys.argv[3]) + np.arange(len(y)) / sub; u = y < 0; ch = np.where(np.diff(u.astype(int)) != 0)[0]
print('ground crossings at keys', [round(float(k[c + 1]), 2) for c in ch]); print('min y per key:', [(int(k[j]), round(float(y[j:j + sub].min()), 1)) for j in range(0, len(y) - 1, sub)])
