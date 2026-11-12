"""sam7.py S BUILDDIR -> how closely the recall of this build follows the Samira-mod pose (per region), and how still the planted scissors stand"""
import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
from glbanim import Glb
D = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'; B = 'assets/ice/wild/bum/characters/talon/skins/'
skS = L.load_skl(D + 'samira_skin30.skl'); mS = L.load_skn(D + 'samira_skin30.skn'); g = Glb(S + '/dl/samira30/model.glb')
RIG = S + '/' + sys.argv[2] + '/wad/' + B; skN = L.load_skl(RIG + 'skin12/talon_skin12.skl'); mN = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
a = L.load_anm(RIG + 'skin12/animations/talon_recall.anm'); F = a.nframes; times = np.minimum(np.arange(F) / a.fps, 244 / 30.0); smp = g.sample('Recall', times); nm = [n for n in skS.names if n in smp]
aS = L.Anim(); aS.hashes = np.array([L.elf(n) for n in nm], np.uint32); aS.sig = 'r3d2anmd'; aS.has = np.ones((len(nm), 3), bool); aS.T = np.stack([smp[n][0] for n in nm], 1); aS.R = np.stack([smp[n][1] for n in nm], 1); aS.S = np.stack([smp[n][2] for n in nm], 1)
infN = np.array(skN.influences)[mN.infl]; infS = np.array(skS.influences)[mS.infl]; domN = infN[np.arange(len(mN.pos)), mN.w.argmax(1)]; domS = infS[np.arange(len(mS.pos)), mS.w.argmax(1)]
E = []; fr = list(range(12, 236, 4)); roots = []
for f in fr:
    vs = L.deform(mS, skS, L.pose_world(skS, aS, f))[0] * 0.744; W = L.pose_world(skN, a, f); vn = L.deform(mN, skN, W)[0]
    d = vn - vs; d -= np.median(d, 0)      # (ground lift / perch shift are deliberate whole-body offsets)
    E.append(np.linalg.norm(d, axis=1))
E = np.array(E); print('per Talon joint region: mean / max error (units), worst key')
for j in np.unique(domN):
    sel = domN == j; e = E[:, sel]
    if e.max() > 6: print(f'  {skN.names[j]:14s} n {sel.sum():5d}  mean {e.mean():5.2f}  max {e.max():6.2f}  key {fr[int(e.max(1).argmax())]}')
print('per Samira joint region (max > 10):')
for j in np.unique(domS):
    sel = domS == j; e = E[:, sel]
    if e.max() > 10: print(f'  {skS.names[j]:14s} n {sel.sum():5d}  mean {e.mean():5.2f}  max {e.max():6.2f}  key {fr[int(e.max(1).argmax())]}')
# planted scissors: world motion of two points on them between keys (8 sub-samples), keys 165..214
JW = skN.index['weapon']; sub = 8; k0, k1 = 166, 236; pts = np.array([[0, 0, 0, 1.0], [60, 0, 0, 1.0], [0, 60, 0, 1.0], [0, 0, 60, 1.0]])
i = np.arange(sub * (F - 1) + 1); f0 = np.minimum(i // sub, F - 2); x = (i - f0 * sub) / sub; b = L.Anim(); b.hashes = a.hashes; b.sig = a.sig; b.has = a.has
b.T = (1 - x[:, None, None]) * a.T[f0] + x[:, None, None] * a.T[f0 + 1]; b.S = (1 - x[:, None, None]) * a.S[f0] + x[:, None, None] * a.S[f0 + 1]; b.R = L.slerp(a.R[f0], a.R[f0 + 1], np.repeat(x[:, None], a.R.shape[1], 1))
P = np.array([(L.pose_world(skN, b, f)[JW] @ pts.T).T[:, :3] for f in range(k0 * sub, k1 * sub + 1)])
keyP = P[::sub]; print('planted scissors keys', k0, '-', k1, ': drift across keys (max dist from mean)', np.round(np.linalg.norm(keyP - keyP.mean(0), axis=2).max(), 2))
dev = [np.linalg.norm(P[k * sub + s_] - ((1 - s_ / sub) * P[k * sub] + s_ / sub * P[(k + 1) * sub]), axis=1).max() for k in range(k1 - k0) for s_ in range(1, sub)]
print('   wobble between keys: max', round(max(dev), 2), 'mean', round(float(np.mean(dev)), 3), 'worst key', k0 + int(np.argmax(dev)) // (sub - 1))
