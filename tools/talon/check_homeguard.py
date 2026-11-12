"""Check homeguard endpoints and quarter-key tail/scissors clearance from built files."""
import sys, copy
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
import lolio as L
root = Path(sys.argv[1]) / "wad/assets/ice/wild/bum/characters/talon/skins/skin12"
sk = L.load_skl(str(root / "talon_skin12.skl")); m = L.load_skn(str(root / "ellen/ellen_talon.skn"))
def load(n): return L.load_anm(str(root / ("animations/talon_skin12_" + n + ".anm")))
run, entry, sprint = map(load, ("run", "homeguard_in", "homeguard"))
def mesh(a, f): return L.deform(m, sk, L.pose_world(sk, a, f))[0]
for label, a, f, b, g in (("run -> entry",run,0,entry,0),("entry -> sprint",entry,-1,sprint,0)):
    d = np.linalg.norm(mesh(a,f)-mesh(b,g), axis=1).max()
    print(label, "max vertex displacement", d); assert d < 1e-3
inf = np.array(sk.influences)[m.infl]; own = inf[np.arange(len(inf)), m.w.argmax(1)]
tail = np.isin(own, [sk.index[f"tail{i}"] for i in range(1,17)]); weapon = own == sk.index["weapon"]
a = copy.deepcopy(entry); a.T = entry.T[:1].copy(); a.R = entry.R[:1].copy(); a.S = entry.S[:1].copy()
gaps = []; heights = []; jumps = []
for f in range(entry.nframes-1):
    for u in (0,.25,.5,.75,1):
        a.T[0] = (1-u)*entry.T[f]+u*entry.T[f+1]; a.S[0] = (1-u)*entry.S[f]+u*entry.S[f+1]
        a.R[0] = L.slerp(entry.R[f],entry.R[f+1], np.full(len(entry.hashes),u))
        p = mesh(a,0); gaps.append(cKDTree(p[weapon]).query(p[tail])[0].min()); heights.append(p[tail,1].min())
        if u == 0: jumps.append(p[tail])
print("entry minimum tail/scissors nearest-vertex distance", min(gaps), "minimum tail height",min(heights))
print("max consecutive key tail vertex movement", np.linalg.norm(np.diff(jumps,axis=0),axis=-1).max())
assert min(gaps) > 10 and min(heights) > 0
