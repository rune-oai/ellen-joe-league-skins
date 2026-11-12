import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
from glbanim import Glb
D = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'; B = 'assets/ice/wild/bum/characters/talon/skins/'
skS = L.load_skl(D + 'samira_skin30.skl'); mS = L.load_skn(D + 'samira_skin30.skn'); g = Glb(S + '/dl/samira30/model.glb')
RIG = S + '/' + sys.argv[2] + '/wad/' + B; skN = L.load_skl(RIG + 'skin12/talon_skin12.skl'); mN = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
a = L.load_anm(RIG + 'skin12/animations/talon_recall.anm')
def ups(a, sub):
    F = len(a.T); i = np.arange(sub * (F - 1) + 1); f0 = np.minimum(i // sub, F - 2); x = (i - f0 * sub) / sub; b = L.Anim(); b.hashes = a.hashes; b.sig = 'r3d2anmd'; b.has = np.ones((len(a.hashes), 3), bool)
    b.T = (1 - x[:, None, None]) * a.T[f0] + x[:, None, None] * a.T[f0 + 1]; b.S = (1 - x[:, None, None]) * a.S[f0] + x[:, None, None] * a.S[f0 + 1]; b.R = L.slerp(a.R[f0], a.R[f0 + 1], np.repeat(x[:, None], a.R.shape[1], 1)); return b
def wob(sk, a, jn, pts, k0, k1, sub=8, label=''):
    b = ups(a, sub); J = sk.index[jn]; P = np.array([(L.pose_world(sk, b, f)[J] @ pts.T).T[:, :3] for f in range(k0 * sub, k1 * sub + 1)])
    dev = np.array([[np.linalg.norm(P[k * sub + s_] - ((1 - s_ / sub) * P[k * sub] + s_ / sub * P[(k + 1) * sub]), axis=1).max() for s_ in range(1, sub)] for k in range(k1 - k0)]).max(1)
    kp = P[::sub]; step = np.linalg.norm(np.diff(kp, axis=0), axis=2).max(1)
    print(f'{label}: between-key deviation mean {dev.mean():.2f} p90 {np.percentile(dev, 90):.2f} p99 {np.percentile(dev, 99):.2f} max {dev.max():.2f} (key {k0 + dev.argmax()}); intervals > 2 u: {(dev > 2).sum()} of {len(dev)}; key-to-key motion of the planted blade: mean {step.mean():.2f} max {step.max():.2f} (key {k0 + step.argmax()})')
    return dev
# source: Samira at 30 fps, sword verts extremes in Sword joint space
times = np.arange(270) / 30.0; smp = g.sample('Recall', times); nm = [n for n in skS.names if n in smp]
aS = L.Anim(); aS.hashes = np.array([L.elf(n) for n in nm], np.uint32); aS.sig = 'r3d2anmd'; aS.has = np.ones((len(nm), 3), bool); aS.T = np.stack([smp[n][0] for n in nm], 1); aS.R = np.stack([smp[n][1] for n in nm], 1); aS.S = np.stack([smp[n][2] for n in nm], 1)
ptsS = np.array([[0, 0, 60, 1.0], [0, 0, -140, 1.0]]); wob(skS, aS, 'Sword', ptsS, 166, 244, label='Samira source (30 fps, Sword under Chest), units x0.744 = ' ) 
sw = next((vs, vc) for n, vs, vc, i0, ic in mN.subs if n == 'sword'); V = (np.c_[mN.pos[sw[0]:sw[0] + sw[1]], np.ones(sw[1])] @ skN.ibind[skN.index['weapon']].T)
ext = V[[np.argmax(np.linalg.norm(V[:, :3], axis=1)), np.argmin(np.linalg.norm(V[:, :3], axis=1))]]; ext[:, 3] = 1
d = wob(skN, a, 'weapon', ext, 160, 236, label='this build (weapon under r_forearm)')
print('worst intervals:', [(160 + int(i), round(float(d[i]), 1)) for i in np.argsort(-d)[:14]])
