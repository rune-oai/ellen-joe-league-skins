import copy
"""build_anim.py S -> retargets every animation the skin uses onto the new (Ellen-proportioned) skeleton and repairs
the broken clips of the base mod. Writes S/build/wad/.../animations/*.anm and S/build/anim_report.txt"""
import os, sys, pickle, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L

B = 'assets/ice/wild/bum/characters/talon/skins/'
SRC = S + '/w/aemeath_talon/' + B
OUT = sys.argv[2] if len(sys.argv) > 2 else 'build'      # output folder under S (rig.pkl is always read from S/build)
DST = S + '/' + OUT + '/wad/' + B; os.makedirs(S + '/' + OUT, exist_ok=True)
os.makedirs(DST + 'skin12/animations', exist_ok=True)
rig = pickle.load(open(os.environ.get('LOL_RIG', S + '/build') + '/rig.pkl', 'rb'))
names = rig['names']; idx = {n: i for i, n in enumerate(names)}
old_names = rig['old_names']; old_lt = rig['old_lt']; old_lr = rig['old_lr']; lt = rig['lt']; lr = rig['lr']
H = {n: L.elf(n) for n in names}
SWING_LIMIT = 70.0      # right wrist: max bend away from the forearm axis (deg) while following the weapon joint
TWIST_LIMIT = 75.0      # right wrist: max roll about the forearm axis (deg)
A = 'skin12/animations/'
# output file (relative to B)  <-  source file
PLAIN = ['talon_skin12_idle1', 'talon_skin12_run', 'talon_skin12_attack1', 'talon_skin12_attack2', 'talon_skin12_crit', 'talon_skin12_spell1',
         'talon_skin12_spell1_leap', 'talon_skin12_spell2', 'talon_skin12_spell3_jump', 'talon_skin12_spell3_longtorun', 'talon_skin12_spell3_medium_torun',
         'talon_skin12_spell4', 'talon_skin12_channel', 'talon_skin12_channel_windup', 'talon_skin12_homeguard', 'talon_skin12_homeguard_in',
         'talon_skin12_dance', 'talon_skin12_taunt', 'talon_skin12_joke', 'talon_skin12_laugh', 'talon_skin12_death']
rep = open(S + '/' + OUT + '/anim_report.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); rep.write(s + '\n')

def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0); bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz], -1)
def qconj(a): return a * np.array([-1, -1, -1, 1.0])
def qang(a, b): return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))

class Clip:
    def __init__(self, a):
        self.hashes = a.hashes.copy(); self.T = a.T.copy(); self.R = a.R.copy(); self.S = a.S.copy(); self.fps = a.fps; self.F = a.nframes
        self.R /= np.linalg.norm(self.R, axis=-1, keepdims=True)
        self.t = {int(h): i for i, h in enumerate(self.hashes)}
    def tr(self, name): return self.t.get(H[name] if name in H else L.elf(name))

def load(rel):
    a = L.load_anm(SRC + rel)
    assert a.sig == 'r3d2anmd', rel
    return Clip(a)


# ---------- offline evaluation helpers (ground contact) ----------
RIG = os.environ.get('LOL_RIG', S + '/build') + '/wad/' + B      # skeleton + mesh from build_rig.py
skN = L.load_skl(RIG + 'skin12/talon_skin12.skl'); mN = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
skA = L.load_skl(SRC + 'skin12/talon_skin12.skl'); mA = L.load_skn(SRC + 'skin12/talon_skin12.skn')
def as_anim(hashes, T, R, S_):
    a = L.Anim(); a.hashes = hashes; a.T = T; a.R = R; a.S = S_; a.sig = 'r3d2anmd'; a.has = np.ones((len(hashes), 3), bool); return a
def subset(m, sk, joints):
    inf = np.array(sk.influences)[m.infl]; wsum = sum(((inf == sk.index[n]) * m.w).sum(1) for n in joints); sel = np.where(wsum > 0.5)[0]
    return inf[sel], m.w[sel], np.c_[m.pos[sel], np.ones(len(sel))]
def ymin(sub, M):
    inf, w, P = sub; y = np.zeros(len(P))
    for k in range(4): y += w[:, k] * np.einsum('nj,nj->n', M[inf[:, k], 1, :], P)
    return float(y.min())
FEET_N = subset(mN, skN, ['l_foot', 'r_foot', 'l_toe', 'r_toe']); FEET_A = subset(mA, skA, ['l_foot', 'r_foot', 'l_toe', 'r_toe'])
JW = skN.index['weapon']; JWP = int(skN.parents[JW]); JROOT = skN.index['root']
_sw = next((vs, vc) for n, vs, vc, i0, ic in mN.subs if n == 'sword'); _sel = np.arange(_sw[0], _sw[0] + _sw[1])
assert (np.array(skN.influences)[mN.infl[_sel, 0]] == JW).all() and np.allclose(mN.w[_sel, 0], 1.0), 'scissors must be rigid on the weapon joint'
_vl = (np.c_[mN.pos[_sel], np.ones(len(_sel))] @ skN.ibind[JW].T)[:, :3]          # scissors verts in weapon-joint space
TIP_L = _vl[np.argmax(np.linalg.norm(_vl, axis=1))]; TIP_L = TIP_L / np.linalg.norm(TIP_L)
_rng = np.random.default_rng(0); _ext = np.unique(np.r_[_vl.argmin(0), _vl.argmax(0), np.argsort(-np.linalg.norm(_vl, axis=1))[:150]])
SW_L = _vl[np.unique(np.r_[_ext, _rng.choice(len(_vl), min(900, len(_vl)), replace=False)])]
def _desc(j):
    out = {j}
    for k in range(len(skN.names)):
        p = k
        while p >= 0:
            if p == j: out.add(k); break
            p = int(skN.parents[p])
    return out
_hj = _desc(skN.index['r_hand']); _infN = np.array(skN.influences)[mN.infl]
_hv = np.where((np.isin(_infN, list(_hj)) * mN.w).sum(1) > 0.5)[0]; _hv = _hv[~np.isin(_hv, _sel)]
_dmin = np.array([np.linalg.norm(mN.pos[_sel] - mN.pos[v], axis=1).min() for v in _hv])
GRIP_P = np.c_[mN.pos[_hv[_dmin < 0.75]], np.ones(int((_dmin < 0.75).sum()))]; assert len(GRIP_P) > 20, len(GRIP_P)   # where the fist touches the handle (bind pose)
TAILJ = [skN.index[f'tail{i}'] for i in range(1, 17)]; JT1 = TAILJ[0]; JT16 = TAILJ[-1]; JHIP = skN.index['hip']; assert int(skN.parents[JT1]) == JHIP
TAIL_SUB = subset(mN, skN, [f'tail{i}' for i in range(1, 17)]); TAIL_H = L.elf('tail1')
TAIL_ISJ = np.isin(np.arange(len(skN.names)), TAILJ)
def rot_about(ax, deg, pivot):
    h = np.radians(deg) / 2; M = np.eye(4); M[:3, :3] = L.quat_to_mat(np.r_[ax * np.sin(h), np.cos(h)]); M[:3, 3] = pivot - M[:3, :3] @ pivot; return M
def tail_ymin(Mskin, ax, deg, pivot):
    M = Mskin.copy(); M[TAIL_ISJ] = rot_about(ax, deg, pivot) @ Mskin[TAIL_ISJ]; return ymin(TAIL_SUB, M)
def rotq_about(q, pivot):
    M = np.eye(4); M[:3, :3] = L.quat_to_mat(q / np.linalg.norm(q)); M[:3, 3] = pivot - M[:3, :3] @ pivot; return M
def tail_ymin_m(Mskin, M4):
    M = Mskin.copy(); M[TAIL_ISJ] = M4 @ Mskin[TAIL_ISJ]; return ymin(TAIL_SUB, M)
SLACK = 5.0             # the scissors may turn at most this far (deg) inside the fist beyond the wrist limits
SOLE_FLOOR = -0.5       # soles never rest lower than this, even where the base rig sinks into the ground
TAIL_MARGIN = 1.5       # tail clearance at the keys (leaves room for the interpolated motion between them)
SUB = 8                 # sub-frame samples per key interval when checking the interpolated motion
ENTRY_LIFT = {'talon_skin12_spell3_medium_torun': (1, 5, 1.3)}      # clip: (first key, end key, units) - room for the cross-fade from the frozen vault pose
MARGIN_ITERS = 8        # passes of the between-key margin search in emit()
GROUND_MARGIN = 0.5     # scissors stay this far above the ground plane
EIN_GM = 2.5      # scissors margin on the keys eased out of the idle stance
GUARD_MAX = 60.0        # max corrective lift of the scissors (deg)
PINNED = ['C_BUFFBONE_GLB_HEAD_LOC', 'C_BUFFBONE_GLB_CHEST_LOC', 'L_BUFFBONE_GLB_HAND_LOC', 'R_BUFFBONE_GLB_HAND_LOC']
PLANTED = ('talon_skin12_attack1', 'talon_skin12_spell1')      # first basic attack / Q melee: a wide planted lunge
RUNLIKE = ('talon_skin12_run', 'talon_skin12_homeguard', 'talon_skin12_homeguard_in')   # base mod hovers above the ground in these
BODY = ['root', 'hip', 'spine', 'chest', 'neck', 'head', 'l_shoulder', 'r_shoulder', 'l_uparm', 'r_uparm', 'l_forearm', 'r_forearm', 'l_thigh', 'r_thigh', 'l_knee', 'r_knee', 'l_foot', 'r_foot']
# ---------- live tail ----------
# The tail used to be a rigid stick (only tail1 keyed) left to the engine's ConformToPath modifier, which looked stiff in game
# and could push it through the scissors. Every tail joint is now keyed in every clip: a travelling swish wave (lateral and
# vertical, growing toward the tip), a gentle resting curve, and follow-through that lags behind the pelvis' own motion.
TAILH = np.array([L.elf(f'tail{i}') for i in range(1, 17)], np.uint32)
def _qm(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float)); ax, ay, az, aw = np.moveaxis(a, -1, 0); bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)
def _qc(q): return np.asarray(q, float) * np.array([-1, -1, -1, 1.0])
_QB = {}
for _j in range(len(skN.names)):      # bind world rotation of every joint (parents precede children)
    _p = int(skN.parents[_j]); _q = skN.lr[_j] / np.linalg.norm(skN.lr[_j]); _QB[_j] = _q if _p < 0 else _qm(_QB[_p], _q)
TAIL_QB = np.array([_QB[j] for j in TAILJ]); TAIL_LR = np.array([skN.lr[j] / np.linalg.norm(skN.lr[j]) for j in TAILJ]); TAIL_LT = np.array([skN.lt[j] for j in TAILJ])
_tp = np.array([np.linalg.inv(skN.ibind[j])[:3, 3] for j in TAILJ]); _cl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(_tp, axis=0), axis=1))]
TAIL_S = _cl / _cl[-1]; TAIL_E = 0.35 + 0.65 * TAIL_S; HIP_RB = L.quat_to_mat(_QB[JHIP])
assert abs(_tp[-1][2] - _tp[0][2]) > 0.9 * _cl[-1] and _tp[-1][2] < _tp[0][2], 'tail is expected to run straight back (-Z) in the bind pose'
TAIL_REST = np.where(TAIL_S < 0.3, -2.0, np.where(TAIL_S > 0.5, 2.6, 0.0))      # resting curve (deg per joint): droops a little, tip curls up
TAIL_LAG = 4.5; TAIL_VREF = 180.0; TAIL_WREF = 300.0; TAIL_DELAY = 0.14          # follow-through: max deg per joint, units/s and deg/s that half-saturate it, lag of the tip (s)
def tail_cfg(name):
    # A / A2: primary / secondary sideways wave (deg per joint at the tip), Ap: vertical wave, hz: primary rate, cyc: primary cycles per loop
    if 'homeguard' in name: return dict(A=5.5, hz=0.8, A2=0.0, Ap=0.0, cyc=1)                # sprint: one smooth lateral sweep per stride, no vertical ripple
    if name.endswith('_run'): return dict(A=6.0, hz=0.8, A2=0.0, Ap=0.0, cyc=1)              # one swish per stride cycle
    if any(k in name for k in ('idle', 'channel', 'recall', 'dance', 'taunt', 'joke', 'laugh')): return dict(A=6.0, hz=0.45, A2=0.0, Ap=0.25, cyc=0)   # lazy swish with a quicker ripple on top
    return dict(A=5.0, hz=1.1, A2=1.8, Ap=2.0, cyc=0)                                        # attacks, spells, death
def _delayed(x, d, loop):
    """x (F,) sampled d keys earlier"""
    F = len(x); f = np.arange(F) - d
    if loop and F > 2: return np.interp(np.mod(f, F - 1), np.arange(F), x)
    return np.interp(np.clip(f, 0, F - 1), np.arange(F), x)
def tail_tracks(name, WN, lift, fps, loop):
    F = len(WN); cfg = tail_cfg(name); t = np.arange(F) / fps; D = max((F - 1) / fps, 1e-6); u = np.arange(F) / max(F - 1, 1); s = TAIL_S[None]; e = TAIL_E[None]
    if loop: n1 = cfg['cyc'] or max(1, int(round(D * cfg['hz']))); ph1 = 2 * np.pi * n1 * u; ph2 = 2 * np.pi * (3 * n1) * u
    else: ph1 = 2 * np.pi * cfg['hz'] * t; ph2 = 3.0 * ph1
    amp = np.ones(F)
    if 'death' in name: amp = 1 - sstep((u - 0.3) / 0.5)       # the tail comes to rest with her
    yaw = amp[:, None] * (cfg['A'] * e * np.sin(ph1[:, None] - 2 * np.pi * 0.8 * s) + cfg['A2'] * e * np.sin(ph2[:, None] - 2 * np.pi * 1.3 * s + 1.0))
    pit = TAIL_REST[None] + amp[:, None] * cfg['Ap'] * e * np.sin(ph2[:, None] - 2 * np.pi * 0.9 * s + 2.0)
    if F > 2:
        hipR = np.array([W[JHIP][:3, :3] for W in WN]); p1 = np.array([W[JT1][:3, 3] for W in WN]); p1[:, 1] += lift
        if loop: vel = (np.roll(p1[:-1], -1, 0) - np.roll(p1[:-1], 1, 0)) / 2 * fps; vel = np.r_[vel, vel[:1]]
        else: vel = np.gradient(p1, axis=0) * fps
        Rrel = np.einsum('fij,fkj->fik', np.roll(hipR, -1, 0), hipR); w = 0.5 * np.stack([Rrel[:, 2, 1] - Rrel[:, 1, 2], Rrel[:, 0, 2] - Rrel[:, 2, 0], Rrel[:, 1, 0] - Rrel[:, 0, 1]], 1)
        w = np.degrees(np.arcsin(np.clip(np.linalg.norm(w, axis=1), 0, 1)))[:, None] * w / np.maximum(np.linalg.norm(w, axis=1), 1e-9)[:, None] * fps
        w[-1] = w[0] if loop else w[-2]
        vb = np.einsum('ij,fkj,fk->fi', HIP_RB, hipR, vel); wb = np.einsum('ij,fkj,fk->fi', HIP_RB, hipR, w)      # in the pelvis' bind frame: x lateral, y up, z forward
        sy_ = smooth(vb[:, 0] / TAIL_VREF - wb[:, 1] / TAIL_WREF, 1.5, loop); sp_ = smooth(-vb[:, 1] / TAIL_VREF - wb[:, 0] / TAIL_WREF, 1.5, loop)
        for i in range(16):
            d = TAIL_DELAY * TAIL_S[i] * fps
            if not any(name == n for n in RUNLIKE):
                yaw[:, i] += TAIL_E[i] * TAIL_LAG * np.tanh(_delayed(sy_, d, loop)); pit[:, i] += TAIL_E[i] * TAIL_LAG * np.tanh(_delayed(sp_, d, loop))
    hy = np.radians(yaw) / 2; hp = np.radians(pit) / 2; z = np.zeros_like(hy)
    dq = _qm(np.stack([z, np.sin(hy), z, np.cos(hy)], -1), np.stack([np.sin(hp), z, z, np.cos(hp)], -1))      # yaw about bind Y, pitch about bind X
    Rl = _qm(TAIL_LR[None], _qm(_qc(TAIL_QB)[None], _qm(dq, TAIL_QB[None]))); Rl /= np.linalg.norm(Rl, axis=-1, keepdims=True)
    if loop: Rl[-1] = Rl[0]
    return np.repeat(TAIL_LT[None], F, 0), Rl

def smooth(x, sigma=1.5, wrap=False):
    """gaussian over time; wrap=True for looping clips (last key == first key) so the loop seam stays seamless"""
    r = int(np.ceil(3 * sigma)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    if wrap and len(x) > 2: y = np.convolve(np.pad(x[:-1], r, mode='wrap'), k, mode='valid'); return np.r_[y, y[0]]
    return np.convolve(np.pad(x, r, mode='edge'), k, mode='valid')
def dilate(x, r=2, wrap=False):
    if wrap and len(x) > 2: xp = np.pad(x[:-1], r, mode='wrap'); y = np.max(np.stack([xp[i:i + len(x) - 1] for i in range(2 * r + 1)]), 0); return np.r_[y, y[0]]
    xp = np.pad(x, r, mode='edge'); return np.max(np.stack([xp[i:i + len(x)] for i in range(2 * r + 1)]), 0)
def sstep(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def destance(c, crouch, idle):
    """The base mod replaced Talon's crouched idle with an upright one, but most action clips still begin and end in the old
    crouch. Cross-fade those leading / trailing crouch frames to the upright idle pose so she never dips back into it."""
    ix = [c.tr(n) for n in BODY]; ic = [crouch.tr(n) for n in BODY]
    dC = qang(c.R[:, ix], crouch.R[0, ic][None]).mean(1); F = c.F; note = []; wI = np.zeros(F)
    c.orig = (c.T.copy(), c.R.copy(), c.S.copy())
    if dC[0] < 10:
        n_out = int(np.argmax(dC > 15)) if (dC > 15).any() else F
        Ns = int(min(max(n_out + 4, 6), F // 3))
        for f in range(Ns): wI[f] = 1 - sstep(f / Ns); c.T[f], c.R[f], c.S[f] = blend_pose(idle, 0, c, f, 1 - wI[f])
        note.append(f'start f0-{Ns - 1}')
    if dC[-1] < 10:
        n_back = int(np.argmax(dC[::-1] > 15)) if (dC > 15).any() else F
        s0 = max(F - 1 - (n_back + 4), F // 2); ramp = max(1, min(10, F - 1 - s0))
        for f in range(s0 + 1, F): wI[f] = sstep((f - s0) / ramp); c.T[f], c.R[f], c.S[f] = blend_pose(idle, 0, c, f, 1 - wI[f])
        note.append(f'end f{s0 + 1}-{F - 1}')
    c.wI = wI
    return ('crouch>upright ' + ', '.join(note)) if note else ''

def bind_frames(c):
    """frames whose limbs sit in the old bind (T) pose: the tell-tale of the stray reference frames in the base mod"""
    bad = np.ones(c.F, bool)
    for n in ['l_uparm', 'r_uparm', 'l_forearm', 'l_thigh', 'r_thigh', 'l_knee', 'spine', 'chest']:
        t = c.tr(n); j = old_names.index(n)
        bad &= qang(c.R[:, t], old_lr[j][None]) < 1.0
    return np.where(bad)[0]

def blend_pose(c0, f0, c1, f1, w):
    """pose (T,R,S per track of c1) interpolated from clip c0 frame f0 to clip c1 frame f1"""
    T = c1.T[f1].copy(); R = c1.R[f1].copy(); S_ = c1.S[f1].copy()
    for h, t1 in c1.t.items():
        t0 = c0.t.get(h)
        if t0 is None: continue
        T[t1] = (1 - w) * c0.T[f0, t0] + w * c1.T[f1, t1]; S_[t1] = (1 - w) * c0.S[f0, t0] + w * c1.S[f1, t1]
        R[t1] = L.slerp(c0.R[f0, t0], c1.R[f1, t1], w)
    return T, R, S_

def upsample(T, R, S_, K):
    """K-1 extra frames per key interval, interpolated the way the engine does it (lerp T/S, shortest-arc slerp R)"""
    F = len(T)
    if K == 1 or F < 2: return T.copy(), R.copy(), S_.copy()
    i = np.arange(K * (F - 1) + 1); f0 = np.minimum(i // K, F - 2); a = (i - f0 * K) / K; a3 = a[:, None, None]
    Ro = L.slerp(R[f0], R[f0 + 1], np.repeat(a[:, None], R.shape[1], 1))
    return (1 - a3) * T[f0] + a3 * T[f0 + 1], Ro, (1 - a3) * S_[f0] + a3 * S_[f0 + 1]

def clearance(hashes, T, R, S_, sub, detail=False):
    """lowest point of soles / scissors / tail and the widest hand-vs-scissors grip gap over the interpolated motion"""
    Tu, Ru, Su = upsample(T, R, S_, sub); a = as_anim(hashes, Tu, Ru, Su); gap = 0.0; D = np.zeros((len(Tu), 3))
    for f in range(len(Tu)):
        W = L.pose_world(skN, a, f); M = W @ skN.ibind
        D[f] = ymin(FEET_N, M), float((SW_L @ W[JW][:3, :3].T)[:, 1].min() + W[JW][1, 3]), ymin(TAIL_SUB, M)
        Mh = M[skN.index['r_hand']]; Mw = M[JW]; gap = max(gap, float(np.linalg.norm((GRIP_P @ Mh.T - GRIP_P @ Mw.T)[:, :3], axis=1).max()))
    fy, sy, ty = D.min(0)
    if detail:      # lowest point per key interval (soles, scissors, tail)
        n = len(T) - 1; return fy, sy, ty, gap, np.array([D[i * sub:(i + 1) * sub + 1].min(0) for i in range(n)]) if n > 0 else D[:1]
    return fy, sy, ty, gap

def retarget(c, label, K=1, xm=None):
    """re-proportion translations for the new bind, lock the right hand to the weapon joint, write-ready arrays.
    K > 1 resamples the clip at K times its key rate so the ground corrections also hold between the original keys."""
    T = c.T.copy(); R = c.R.copy(); S_ = c.S.copy()
    missing = [n for n in old_names if c.tr(n) is None]
    for j, n in enumerate(old_names):
        t = c.tr(n)
        if t is None: continue
        T[:, t] += lt[idx[n]] - old_lt[j]
    tw, th = c.tr('weapon'), c.tr('r_hand'); free = getattr(c, 'free', False)      # free: the scissors leave the hand (recall), see samira_recall()
    assert tw is not None and th is not None, label
    T[:, th] = lt[idx['r_hand']]; T[:, tw] = lt[idx['weapon']]
    # Attachment points on the head, chest and hands: the base mod's own clips carry garbage on these tracks (scale 0 / -1,
    # single-frame 170 deg flips, offsets of 7-19 units), which would throw attached effects around. They are rigid
    # attachment points, so pin them to their bind transform under the parent joint.
    for n in PINNED:
        t = c.tr(n)
        if t is not None: T[:, t] = lt[idx[n]]; R[:, t] = lr[idx[n]]; S_[:, t] = 1.0
    # The dash clips stretch a leg up to 2.45x for a few take-off / landing frames (thigh scale, counter-scaled foot). On
    # Ellen's bare legs that reads as a rubber limb, so every joint keeps unit scale.
    stretch = float(np.abs(S_ - 1).max()); S_[:] = 1.0
    # The right hand grips the scissors. Talon's blade turns freely against his forearm (up to 140 deg), which no wrist can
    # follow, so the rotation of the weapon joint is split into swing (bending away from the forearm) and twist (roll about
    # it), both limited to what a wrist can do; the scissors themselves may turn only SLACK degrees further inside the fist.
    qb = lr[idx['r_hand']][None]; axf = lt[idx['r_hand']] / np.linalg.norm(lt[idx['r_hand']])
    def split(q):
        Dq = qmul(q, qconj(qb)); Dq[Dq[:, 3] < 0] *= -1
        twq = np.c_[np.outer(Dq[:, :3] @ axf, axf), Dq[:, 3]]; nn = np.linalg.norm(twq, axis=1, keepdims=True)
        twq = np.where(nn > 1e-9, twq / np.maximum(nn, 1e-9), np.array([0, 0, 0, 1.0])); swq = qmul(Dq, qconj(twq))
        ident = np.tile(np.array([0, 0, 0, 1.0]), (len(q), 1)); sa = qang(swq, ident); ta = qang(twq, ident)
        swc = L.slerp(ident, swq, np.minimum(1.0, SWING_LIMIT / np.maximum(sa, 1e-6))); twc = L.slerp(ident, twq, np.minimum(1.0, TWIST_LIMIT / np.maximum(ta, 1e-6)))
        H = qmul(swc, twc); E = qmul(Dq, qconj(H)); ea = qang(E, ident)
        return H, E, ea, sa, ta, ident
    mx = (0.0, 0.0); nover = 0
    if not free:
        H, E, ea, sa, ta, ident = split(R[:, tw].copy())
        Ec = L.slerp(ident, E, np.minimum(1.0, SLACK / np.maximum(ea, 1e-6)))
        R[:, tw] = qmul(Ec, qmul(H, np.repeat(qb, len(H), 0))); R[:, tw] /= np.linalg.norm(R[:, tw], axis=1, keepdims=True)
        mx = (float(sa.max()), float(ta.max())); nover = int(((sa > SWING_LIMIT) | (ta > TWIST_LIMIT)).sum())
    # --- resample (see K) together with the base-rig reference
    F0 = len(T); bT, bR, bS = getattr(c, 'orig', (c.T, c.R, c.S)); yref = getattr(c, 'yref', np.full(F0, np.nan)); wI = getattr(c, 'wI', np.zeros(F0))
    T, R, S_ = upsample(T, R, S_, K); bT, bR, bS = upsample(bT, bR, bS, K); F = len(T); x = np.arange(F) / K; k0 = np.arange(F0)
    wI = np.interp(x, k0, wI); ynan = np.interp(x, k0, np.isnan(yref).astype(float)) > 0; yref = np.where(ynan, np.nan, np.interp(x, k0, np.nan_to_num(yref)))
    sg = 1.5 * K; loop = F0 > 2 and float(qang(c.R[0], c.R[-1]).max()) < 0.5 and float(np.abs(c.T[0] - c.T[-1]).max()) < 0.05      # looping clip
    # --- ground contact. (a) feet: Ellen's ankle is lower and she has no toe joint, so heel-up stances would push her toes
    #     into the floor; lift the root so her lowest sole point matches the base rig's in every frame (and never sinks).
    aN = as_anim(c.hashes, T, R, S_); aA = as_anim(c.hashes, bT, bR, bS)
    WN = [L.pose_world(skN, aN, f) for f in range(F)]
    yN = np.array([ymin(FEET_N, WN[f] @ skN.ibind) for f in range(F)]); yA = np.array([ymin(FEET_A, L.pose_world(skA, aA, f) @ skA.ibind) for f in range(F)])
    yA = np.where(np.isnan(yref), yA, yref)
    yA = (1 - wI) * yA + wI * IDLE_Y      # cross-faded frames: blend the two grounded references
    wX = np.interp(x, k0, c.wX) if hasattr(c, 'wX') else np.zeros(F)
    wE = np.interp(x, k0, c.wE) if hasattr(c, 'wE') else np.zeros(F)
    if hasattr(c, 'wX'): yA = (1 - wX) * yA + wX * np.interp(x, k0, c.yX)      # frames eased from / into a neighbouring clip's pose
    planted = any(label.endswith(r + '.anm') for r in PLANTED)
    if planted: yA = np.minimum(yA, 0.0)      # the base holds this lunge 12-25 units in the air with the hips at standing height: stand it on the ground
    raw = np.clip(np.maximum(yA, SOLE_FLOOR) - yN, -30.0 if planted else -1.0 - 9.0 * wX, 10.0)       # (eased keys follow the neighbouring clip's final height, which may sit lower: run)
    lift = np.maximum(smooth(raw, sg, loop), raw - 0.4) if F > 2 else raw
    if any(label.endswith(r + '.anm') for r in RUNLIKE): lift = lift + max(-10.0, min(0.0, SOLE_FLOOR - float((yN + lift)[wX < 0.5].min())))      # (the keys eased into the next clip take that clip's height)
    lift = (1 - wX) * lift + wX * raw      # keys taken from a neighbouring clip sit exactly at that clip's height
    if xm is None: xm = np.zeros((F, 3))      # extra per-key margins (soles, scissors, tail) asked for by emit() for the motion between keys
    lift = lift + xm[:, 0]
    T[:, c.tr('root'), 1] += lift
    if free:      # the scissors follow their own world path (c.wW); keyed under the forearm, so re-express it against the lifted arm
        assert K == 1
        for f in range(F):
            Wp = WN[f][JWP].copy(); Wp[1, 3] += lift[f]; Wt = c.wW[f].copy(); Wt[1, 3] += c.wU[f] * lift[f]; Ml = np.linalg.inv(Wp) @ Wt; T[f, tw] = Ml[:3, 3]; R[f, tw] = L.mat_to_quat(Ml[:3, :3]); WN[f][JW] = WN[f][JWP] @ Ml
    # --- live tail tracks (appended after the base tracks); the correction below then turns the whole animated tail at tail1
    tT, tR = tail_tracks(os.path.basename(label)[:-4], WN, lift, c.fps * K, loop); n0 = T.shape[1]
    T = np.concatenate([T, tT], 1); R = np.concatenate([R, tR], 1); S_ = np.concatenate([S_, np.ones((F, 16, 3))], 1)
    for f in range(F):
        for i, j in enumerate(TAILJ):
            Ml = np.eye(4); Ml[:3, :3] = L.quat_to_mat(tR[f, i]); Ml[:3, 3] = tT[f, i]; WN[f][j] = WN[f][int(skN.parents[j])] @ Ml
    # --- (c) tail: its joints are not animated (the engine's ConformToPath modifier sways them), so its rest direction just
    #     follows the pelvis: straight up when she lies face down, through the floor in the landing roll. Add a track for the
    #     tail root that keeps the tail from pointing steeply upward (the lower the pelvis, the flatter) and above the ground.
    Cq = np.tile(np.array([0, 0, 0, 1.0]), (F, 1)); guardT = np.zeros(F, bool); RpT = np.zeros((F, 3, 3)); prev_h = None; cand = np.arange(2.0, 151.0, 2.0)
    up = np.array([0, 1.0, 0]); P1 = np.zeros((F, 3)); Dn = np.zeros((F, 3)); MSK = []; phis = np.zeros(F)
    def horiz(d, f, p1, prev):
        """horizontal heading of the tail; when it is (nearly) vertical fall back to 'away from the feet'"""
        h = d - d[1] * up; hn = np.linalg.norm(h)
        fb = (WN[f][skN.index['l_foot']][:3, 3] + WN[f][skN.index['r_foot']][:3, 3]) / 2 + np.array([0, lift[f], 0]) - p1; fb[1] = 0
        fb = fb / np.linalg.norm(fb) if np.linalg.norm(fb) > 1e-3 else (prev if prev is not None else np.array([0, 0, -1.0]))
        wgt = float(sstep((hn - 0.15) / 0.25)); hd = wgt * (h / max(hn, 1e-9)) + (1 - wgt) * fb; return hd / max(np.linalg.norm(hd), 1e-9), hn
    def qax(ax, deg): hh = np.radians(deg) / 2; return np.r_[ax * np.sin(hh), np.cos(hh)]
    def tail_y(f, q): return tail_ymin_m(MSK[f], rotq_about(q, P1[f]))
    for f in range(F):
        W = WN[f]; lv = np.array([0, lift[f], 0])
        p1 = W[JT1][:3, 3] + lv; d = W[JT16][:3, 3] + lv - p1; d /= np.linalg.norm(d); hip_y = W[JHIP][1, 3] + lift[f]; P1[f] = p1; Dn[f] = d
        u, _, vt = np.linalg.svd(W[JHIP][:3, :3]); RpT[f] = u @ vt
        hd, hn = horiz(d, f, p1, prev_h); prev_h = hd
        ax = np.cross(hd, up); ax /= np.linalg.norm(ax)                                   # +angle about ax raises the tail, -angle lowers it
        elev = np.degrees(np.arcsin(np.clip(d[1], -1, 1))); emax = 10.0 + (2.0 if 'homeguard' in label else 18.0) * float(np.clip((hip_y - 30.0) / 50.0, 0, 1))
        if hn < 0.15 and d[1] > 0: elev = 90.0 + np.degrees(np.arccos(np.clip(d @ up, -1, 1))) * (1 if d @ hd < 0 else -1)   # past vertical, away from hd
        if 'idle' in label or 'channel' in label: emax = 12.0
        capw = 1.0 - float(sstep((hip_y - 125.0) / 40.0))        # on the ground a heavy tail never sticks up like a flag (the run leans the pelvis forward); in the air it follows the pelvis
        ph = -max(0.0, elev - emax) * capw
        if any(label.endswith(n + '.anm') for n in RUNLIKE):
            ph = -8.0 - elev      # keep the tail low and steady in world pitch while the running pelvis rocks
        Msk = W @ skN.ibind; Msk[:, 1, 3] += lift[f]; MSK.append(Msk)
        if tail_ymin(Msk, ax, ph, p1) < TAIL_MARGIN + xm[f, 2]:
            ys = np.array([tail_ymin(Msk, ax, ph + a, p1) for a in cand]); ok = np.where(ys >= TAIL_MARGIN + xm[f, 2])[0]
            ph += cand[ok[0]] if len(ok) else cand[int(np.argmax(ys))]
        phis[f] = ph; Cq[f] = qax(ax, ph)
    tailfix = (float(phis.min()), float(phis.max()))
    if F > 2:
        # smooth the (world-space) correction over time so the tail never snaps from one side to the other in a single frame,
        # then re-check ground clearance on the smoothed result
        for f in range(1, F):
            if Cq[f] @ Cq[f - 1] < 0: Cq[f] *= -1
        Cq0 = Cq.copy(); Cq = np.stack([smooth(Cq[:, i], sg, loop) for i in range(4)], 1); Cq /= np.linalg.norm(Cq, axis=1, keepdims=True)
        if any(label.endswith(n + '.anm') for n in RUNLIKE): Cq = Cq0.copy()      # exact pitch cancellation: smoothing its inverse lets pelvis key jitter leak back in
        if wX.max() > 0: Cq = L.slerp(Cq, Cq0, wX ** 6)      # a key taken from a neighbouring clip keeps that clip's own (unsmoothed) correction
        prev_h = None; extra = np.zeros(F); AX2 = np.zeros((F, 3))
        for f in range(F):
            d = L.quat_to_mat(Cq[f]) @ Dn[f]; hd, hn = horiz(d, f, P1[f], prev_h); prev_h = hd; ax = np.cross(hd, up); AX2[f] = ax / np.linalg.norm(ax)
            if tail_y(f, Cq[f]) < TAIL_MARGIN + xm[f, 2]:
                ys = np.array([tail_y(f, qmul(qax(AX2[f], a)[None], Cq[f][None])[0]) for a in cand]); ok = np.where(ys >= TAIL_MARGIN + xm[f, 2])[0]
                extra[f] = cand[ok[0]] if len(ok) else cand[int(np.argmax(ys))]; guardT[f] = True
        if extra.max() > 0:
            ex = np.maximum(extra, smooth(dilate(extra, 2 * K, loop), sg, loop))
            for f in range(F): Cq[f] = qmul(qax(AX2[f], ex[f])[None], Cq[f][None])[0]
    Cl = np.c_[np.einsum('fji,fj->fi', RpT, Cq[:, :3]), Cq[:, 3]]                        # correction expressed in the pelvis' space
    tailR = _qm(Cl, R[:, n0]); tailR /= np.linalg.norm(tailR, axis=1, keepdims=True); R[:, n0] = tailR
    # --- (b) scissors: they are far longer than Talon's arm blade, so wherever they would cut into the floor rotate them
    #     (about the wrist, in the blade's own vertical plane) just enough to clear it.
    phi_req = np.zeros(F); axes = np.zeros((F, 3)); Rp = np.zeros((F, 3, 3)); angs = np.radians(np.arange(1, GUARD_MAX + 1)); prev_ax = None
    for f in range(0 if not free else F, F):
        Mw = WN[f][JW]; pv = Mw[:3, 3] + np.array([0, lift[f], 0]); V = SW_L @ Mw[:3, :3].T; d = Mw[:3, :3] @ TIP_L
        u, _, vt = np.linalg.svd(WN[f][JWP][:3, :3]); Rp[f] = u @ vt
        ax = np.cross(d, [0, 1.0, 0])
        if np.linalg.norm(ax) < 0.2:                       # blade (nearly) vertical: tip it away from the body instead
            o = pv - WN[f][JROOT][:3, 3]; o[1] = 0; o = o / max(np.linalg.norm(o), 1e-6); ax = np.cross(d, o)
            if prev_ax is not None and np.linalg.norm(ax) < 0.2: ax = prev_ax
        ax = ax / max(np.linalg.norm(ax), 1e-9); axes[f] = ax; prev_ax = ax
        gm = GROUND_MARGIN + (1.0 if 0 < wX[f] < 1 else 0.0)      # eased keys sweep the blade fast: more room for the motion between keys
        if 0 < wE[f] < 1: gm = GROUND_MARGIN + EIN_GM      # keys eased out of the idle stance are also what the engine cross-fades into: more room
        gm += xm[f, 1]
        if (V[:, 1] + pv[1]).min() >= gm: continue
        cx = np.cross(ax, V)[:, 1]; dt = V @ ax
        ys = np.cos(angs)[:, None] * V[None, :, 1] + np.sin(angs)[:, None] * cx[None] + (1 - np.cos(angs))[:, None] * ax[1] * dt[None] + pv[1]
        mins = ys.min(1); ok = np.where(mins >= gm)[0]
        phi_req[f] = np.degrees(angs[ok[0]] if len(ok) else angs[np.argmax(mins)])
    guard = (float(phi_req.max()), int((phi_req > 0).sum()))
    if phi_req.max() > 0:
        phi = np.maximum(phi_req, (1 - wX ** 6) * smooth(dilate(phi_req, 2 * K, loop), sg, loop)) if F > 2 else phi_req
        axl = np.einsum('fji,fj->fi', Rp, axes); h = np.radians(phi) / 2          # axis in the forearm's space
        R[:, tw] = qmul(np.c_[axl * np.sin(h)[:, None], np.cos(h)], R[:, tw]); R[:, tw] /= np.linalg.norm(R[:, tw], axis=1, keepdims=True)
    # the hand follows the (now final) scissors; the ground guard may have turned them past the wrist limits, in which case
    # the wrist gives way rather than the grip (the guard only ever acts for a few frames and by a few degrees)
    if not free:
        H, E, ea, sa2, ta2, ident = split(R[:, tw].copy())
        Hq = qmul(L.slerp(ident, E, np.maximum(0.0, 1.0 - SLACK / np.maximum(ea, 1e-6))), H)
        R[:, th] = qmul(Hq, np.repeat(qb, F, 0)); R[:, th] /= np.linalg.norm(R[:, th], axis=1, keepdims=True); S_[:, th] = 1.0
    return T, R, S_, missing, mx, nover, (float(lift.min()), float(lift.max())), guard, tailfix, stretch

def write(rel_out, c, T, R, S_, hashes, K=1):
    # keep quaternion sign continuity per track (cosmetic; the engine slerps on the shortest arc anyway)
    for f in range(1, len(R)):
        flip = (R[f] * R[f - 1]).sum(-1) < 0; R[f][flip] *= -1
    n = L.write_anm_v4(DST + rel_out, hashes, T, R, S_, 1.0 / (c.fps * K))
    b = L.load_anm(DST + rel_out)      # read-back check
    assert b.nframes == len(T) and np.array_equal(b.hashes, hashes)
    assert np.abs(b.T - T).max() < 1e-3 and np.abs(b.S - S_).max() < 1e-4 and qang(b.R, R).max() < 0.05, rel_out
    return n

idle = load(A + 'talon_skin12_idle1.anm')

def port_wip_idle(c):
    """The work-in-progress mod restyled the idle (left hand on the hip, relaxed shoulders). Its arm joints are bound differently,
    so carry the pose over as mesh-space rotations: D = W_wip * Bind_wip^-1 per joint, re-applied on the new bind."""
    WIP = S + '/w/ellen_talon/' + B
    skE = L.load_skl(WIP + 'skin12/talon_skin12.skl'); aE = L.load_anm(WIP + 'skin12/animations/talon_skin12_idle1.anm')
    assert aE.nframes == c.F
    rot = lambda M: (lambda u, s_, vt: u @ vt)(*np.linalg.svd(M[:3, :3]))
    JS = ['l_shoulder', 'l_uparm', 'l_forearm', 'l_hand', 'r_shoulder', 'r_uparm', 'r_forearm']
    aO = as_anim(c.hashes, c.T, c.R, c.S); mx = {}
    for f in range(c.F):
        wE = L.pose_world(skE, aE, f); wO = L.pose_world(skA, aO, f); Wn = {}
        for n in JS:
            D = rot(wE[skE.index[n]]) @ rot(skE.world[skE.index[n]]).T
            Wn[n] = D @ rot(skN.world[skN.index[n]])
            pn = skN.names[skN.parents[skN.index[n]]]; Wp = Wn[pn] if pn in Wn else rot(wO[skA.index[pn]])
            q = L.mat_to_quat(Wp.T @ Wn[n]); t = c.tr(n); mx[n] = max(mx.get(n, 0), float(qang(q, c.R[f, t]))); c.R[f, t] = q
    return ', '.join(f'{n} {v:.0f}' for n, v in mx.items())
_idle_note = 'WIP idle pose ported (deg changed: ' + port_wip_idle(idle) + ')'
STANCE = ['root', 'hip', 'spine', 'chest', 'neck', 'head', 'l_thigh', 'l_knee', 'l_foot', 'l_toe', 'r_thigh', 'r_knee', 'r_foot', 'r_toe']
def upright(c):
    """The base channel clips (also the material for the rebuilt recall) hold a crouch as deep as the old ready stance.
    Keep their arm work, but stand on the idle's legs and torso."""
    for n in STANCE:
        t = c.tr(n); ti = idle.tr(n); c.T[:, t] = idle.T[0, ti]; c.R[:, t] = idle.R[0, ti]; c.S[:, t] = idle.S[0, ti]
    return 'crouch > idle stance (arms kept)'
def ease_in_idle(c, hold, n):
    """The engine cross-fades into a clip over 0.1 s (about 3 keys). The dash clips open in a deep push-off crouch, and
    blending that with an upright stance drives the feet / scissors through the floor. Open them on the idle stance for
    the length of that cross-fade and ease into the push-off inside the clip, where the ground corrections apply."""
    if not hasattr(c, 'orig'): c.orig = (c.T.copy(), c.R.copy(), c.S.copy()); c.wI = np.zeros(c.F)
    if not hasattr(c, 'wE'): c.wE = np.zeros(c.F)
    for f in range(hold + n):
        w = 1.0 if f < hold else 1.0 - sstep((f - hold + 1) / (n + 1))
        c.T[f], c.R[f], c.S[f] = blend_pose(idle, 0, c, f, 1 - w); c.wI[f] = max(c.wI[f], w); c.wE[f] = w
    return f'opens on the idle stance (f0-{hold + n - 1})'
def ease_out_idle(c, n):
    """End a one-shot clip on the idle stance. When it has played out, the engine cross-fades it into the idle (or the run); from a
    landing crouch or a lunge that fade passes the feet / scissors through the floor, from the idle stance it is a no-op."""
    if not hasattr(c, 'orig'): c.orig = (c.T.copy(), c.R.copy(), c.S.copy()); c.wI = np.zeros(c.F)
    if not hasattr(c, 'wE'): c.wE = np.zeros(c.F)
    for i in range(n):
        f = c.F - n + i; w = sstep((i + 1) / n)
        c.T[f], c.R[f], c.S[f] = blend_pose(idle, 0, c, f, 1 - w); c.wI[f] = max(c.wI[f], w); c.wE[f] = max(c.wE[f], w)
    return f'ends on the idle stance (f{c.F - n}-{c.F - 1})'
def _xref(c):
    if not hasattr(c, 'orig'): c.orig = (c.T.copy(), c.R.copy(), c.S.copy())
    if not hasattr(c, 'wX'): c.wX = np.zeros(c.F); c.yX = np.zeros(c.F); c.snaps = []
def ease_out_to(c, tgt, n, hold=1):
    """end the clip exactly on the first pose of the clip the graph hands over to (tgt), held for the last `hold` keys, so
    that hand-off blends two like poses whether the engine freezes this clip at its end or lets it play out during the blend"""
    _xref(c); src = load(A + tgt + '.anm'); wk = np.zeros(c.F)
    for i in range(n + hold - 1):
        f = c.F - (n + hold - 1) + i; w = sstep((i + 1) / n); c.T[f], c.R[f], c.S[f] = blend_pose(src, 0, c, f, 1 - w); c.wX[f] = w; c.yX[f] = FINAL[tgt][0]; wk[f] = w
    c.snaps.append((tgt, 0, wk))
    return f'ends on the first pose of {tgt[6:]} (f{c.F - (n + hold - 1)}-{c.F - 1})'
def ease_in_from(c, prev, hold, n):
    """open the clip exactly on the last pose of the clip the graph comes from"""
    _xref(c); src = load(A + prev + '.anm'); wk = np.zeros(c.F)
    for f in range(hold + n):
        w = 1.0 if f < hold else 1.0 - sstep((f - hold + 1) / (n + 1))
        c.T[f], c.R[f], c.S[f] = blend_pose(src, src.F - 1, c, f, 1 - w); c.wX[f] = w; c.yX[f] = FINAL[prev][1]; wk[f] = w
    c.snaps.append((prev, 1, wk))
    return f'opens on the last pose of {prev[6:]} (f0-{hold + n - 1})'
def retime(c, pts):
    """Re-key a clip in place: output key k shows the source pose at pts (piecewise linear key -> source key, interpolated the way
    the engine does it). The key count and the key rate stay as they are (the client plays re-rated clips in slow motion)."""
    src = np.interp(np.arange(c.F), [p[0] for p in pts], [p[1] for p in pts]); f0 = np.clip(np.floor(src).astype(int), 0, c.F - 2); a = src - f0; a3 = a[:, None, None]
    def rs(T, R, S_): return (1 - a3) * T[f0] + a3 * T[f0 + 1], L.slerp(R[f0], R[f0 + 1], np.repeat(a[:, None], R.shape[1], 1)), (1 - a3) * S_[f0] + a3 * S_[f0 + 1]
    c.T, c.R, c.S = rs(c.T, c.R, c.S)
    if hasattr(c, 'orig'): c.orig = rs(*c.orig)
    for k in ('wI', 'wE'):
        if hasattr(c, k): setattr(c, k, np.interp(src, np.arange(c.F), getattr(c, k)))
    return 'compact strike (key>source key: ' + ' '.join(f'{a_}>{b_:g}' for a_, b_ in pts) + ')'
# Q at melee range. The base clip is the first basic attack: a full turn with the blade swept low behind her, 10 keys held overhead
# and the cut only at key 24 (0.83 s). The spell itself is over after about a quarter of a second, so in game all one saw was that
# wide wind-up - with scissors twice as long as Talon's blade. Re-keyed as one short lunge: from the idle stance straight into the
# extended cut (reached on key 7 = 0.24 s; the clip's own hit effect fires on key 8), follow-through, then an unhurried recovery.
QUICK = {'talon_skin12_spell1': [(0, 23.5), (6, 23.8), (7, 24), (10, 26), (13, 28.5), (30, 41), (41, 41)]}
EASE_IN = {'talon_skin12_spell3_jump': (2, 5), 'talon_skin12_spell4': (2, 5), 'talon_skin12_attack1': (2, 3), 'talon_skin12_spell1': (2, 4),
           'talon_skin12_spell1_leap': (2, 2)}                      # clip: (keys held on idle, keys of ease)
EASE_END = {'talon_skin12_spell3_jump': 3, 'talon_skin12_attack1': 6, 'talon_skin12_spell1': 6}      # clip: keys of ease onto the idle stance at the end
EASE_OUT = {'talon_skin12_spell3_longtorun': ('talon_skin12_run', 5, 2), 'talon_skin12_spell3_medium_torun': ('talon_skin12_run', 5, 2),
            'talon_skin12_homeguard_in': ('talon_skin12_homeguard', 5, 2)}   # clip: (clip the graph hands over to, keys of ease, keys held)
EASE_FROM = {'talon_skin12_homeguard_in': ('talon_skin12_run', 1, 10),
             'talon_skin12_spell3_medium_torun': ('talon_spell3_medium', 1, 2),
             'talon_skin12_spell3_longtorun': ('talon_spell3_long', 1, 0)}     # clip: (clip the graph comes from, keys held, keys of ease)
CHANNELS = ('talon_skin12_channel', 'talon_skin12_channel_windup')
LOOPS = ('talon_skin12_idle1', 'talon_skin12_run', 'talon_skin12_homeguard', 'talon_skin12_channel', 'talon_skin12_channel_windup', 'talon_skin12_dance')   # mFlags & 2 in the animation graph
def close_loop(c):
    """Looping clips repeat their first key as the last one, but the base files only do so to within compression noise
    (run: 0.1 units), which also kept retarget() from treating them as loops. Make the last key an exact copy."""
    keep = np.array([i for i in range(len(c.hashes)) if i != c.tr('r_hand')])      # the right hand is rebuilt from the weapon joint in retarget()
    sr = float(qang(c.R[0, keep], c.R[-1, keep]).max()); st = float(np.abs(c.T[0] - c.T[-1]).max())
    if sr > 1.0 or st > 0.2: return f'loop seam left as authored ({sr:.2f} deg / {st:.3f} u)'
    c.T[-1] = c.T[0]; c.R[-1] = c.R[0]; c.S[-1] = c.S[0]
    if hasattr(c, 'orig'):
        for x in c.orig: x[-1] = x[0]
    return f'loop seam closed ({sr:.2f} deg / {st:.3f} u)'
# ---------- custom homeguard: a forward-leaning sprint built from the run cycle ----------
def _wq(c):
    """world rotation (F,4) of every skeleton joint for clip c (rotation tracks only)"""
    W = {}
    for j, n in enumerate(skN.names):
        t = c.tr(n); q = c.R[:, t] if t is not None else np.repeat((skN.lr[j] / np.linalg.norm(skN.lr[j]))[None], c.F, 0)
        pj = int(skN.parents[j]); W[j] = q if pj < 0 else _qm(W[pj], q)
    return W
def _qrot(q, v): return np.einsum('...ij,...j->...i', L.quat_to_mat(q), v)
def _between(a, b):
    """shortest rotation taking unit vector a to b"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); ax = np.cross(a, b); n = np.linalg.norm(ax)
    if n < 1e-9: return np.array([0, 0, 0, 1.0])
    h = np.arctan2(n, a @ b) / 2; return np.r_[ax / n * np.sin(h), np.cos(h)]
def _turn(c, name, Q, W):
    """rotate joint `name` (and everything below it) by the world rotation Q"""
    j = skN.index[name]; t = c.tr(name); Wp = W[int(skN.parents[j])]
    c.R[:, t] = _qm(_qc(Wp), _qm(Q, _qm(Wp, c.R[:, t]))); c.R[:, t] /= np.linalg.norm(c.R[:, t], axis=-1, keepdims=True)
def _axq(ax, deg): h = np.radians(deg) / 2; return np.r_[np.asarray(ax, float) / np.linalg.norm(ax) * np.sin(h), np.cos(h)]
def _mean_dir(v): m = v.mean(0); return m / np.linalg.norm(m)
def make_sprint(run):
    c = copy.deepcopy(run); up = np.array([0, 1.0, 0]); bw = np.linalg.inv(skN.ibind)
    fwd = (bw[skN.index['l_toe']][:3, 3] - bw[skN.index['l_foot']][:3, 3]) * np.array([1, 0, 1.0]); fwd /= np.linalg.norm(fwd)      # she faces along her toes
    hh = _mean_dir(_qrot(_wq(c)[skN.index['spine']], _qrot(_qc(_QB[skN.index['spine']]), up))); assert hh @ fwd > 0.2, 'the run cycle leans along the toes'
    lean = np.cross(up, fwd); left = bw[skN.index['l_uparm']][:3, 3] - bw[skN.index['r_uparm']][:3, 3]; left = lean * np.sign(left @ lean)
    # torso leans into the sprint, head stays on the horizon
    for name, deg in (('spine', 11.0), ('head', -9.0)): _turn(c, name, _axq(lean, deg), _wq(c))      # (the run cycle itself already leans about 35 deg)
    # Keep the authored bent-elbow run motion. Independently straightening both segments made the
    # bare left arm almost collinear (7-23 degrees of bend), losing its elbow silhouette.
    c.T[-1] = c.T[0]; c.R[-1] = c.R[0]; c.S[-1] = c.S[0]
    return c
def make_sprint_in(run, sprint, F):
    """run -> sprint over F keys, arriving on the first key of the sprint loop"""
    c = copy.deepcopy(run); n = run.F - 1; ph = [(f - (F - 1)) % n for f in range(F)]
    c.F = F; c.T = run.T[ph].copy(); c.R = run.R[ph].copy(); c.S = run.S[ph].copy()
    for f in range(F): c.T[f], c.R[f], c.S[f] = blend_pose(run, ph[f], sprint, ph[f], float(sstep(f / max(1, 0.75 * (F - 1)))))
    return c
crouch = load(A + 'talon_skin12_crit.anm')      # its first frame is the old crouched ready stance
IDLE_Y = ymin(FEET_A, L.pose_world(skA, as_anim(idle.hashes, idle.T, idle.R, idle.S), 0) @ skA.ibind)
log('clip / keys -> frames / fixes / weapon excursion vs forearm in the base clip / corrections / clearance of the interpolated motion')
out = {}; FINAL = {}; POSES = {}
def snap(c, T, R, S_, hashes, K):
    """The ground / tail / scissors corrections are computed per clip, so a pose eased to a neighbouring clip's pose can still
    end up a few units off that clip's final (corrected) pose. Pull the eased keys onto it exactly."""
    for tgt, which, wk in getattr(c, 'snaps', []):
        h2, P = POSES[tgt][0], POSES[tgt][1 + which]; mp = {int(h): i for i, h in enumerate(h2)}
        sel = np.array([i for i, h in enumerate(hashes) if int(h) in mp]); src = np.array([mp[int(hashes[i])] for i in sel])
        w = np.interp(np.arange(len(T)) / K, np.arange(c.F), wk); a = len(T) - 1 if which == 0 else 0; assert w[a] == 1.0
        dT = P[0][src] - T[a, sel]; dS = P[2][src] - S_[a, sel]; dR = qmul(P[1][src], qconj(R[a, sel])); ident = np.tile(np.array([0, 0, 0, 1.0]), (len(sel), 1))
        res = (float(np.abs(dT).max()), float(qang(dR, ident).max())); c.snapres = max(getattr(c, 'snapres', (0, 0)), res)
        for f in np.where(w > 0)[0]:      # the residual of the anchor key, faded out with the ease weights
            T[f, sel] += w[f] * dT; S_[f, sel] += w[f] * dS; R[f, sel] = qmul(L.slerp(ident, dR, np.full(len(sel), w[f])), R[f, sel])
def pin_idle(c, T, R, S_, hashes):
    """keys of a clip built outside the usual pipeline (recall) eased onto the final first idle pose with the weights c.wP"""
    if not hasattr(c, 'wP'): return
    h2, P = POSES['talon_skin12_idle1'][0], POSES['talon_skin12_idle1'][1]; mp = {int(h): i for i, h in enumerate(h2)}
    sel = np.array([i for i, h in enumerate(hashes) if int(h) in mp]); src = np.array([mp[int(hashes[i])] for i in sel]); assert len(sel) == len(hashes)
    for f in np.where(c.wP > 0)[0]:
        w = float(c.wP[f]); T[f, sel] = (1 - w) * T[f, sel] + w * P[0][src]; S_[f, sel] = (1 - w) * S_[f, sel] + w * P[2][src]; R[f, sel] = L.slerp(R[f, sel], P[1][src], np.full(len(sel), w))
def emit(rel_out, c, note=''):
    hashes = np.r_[c.hashes, TAILH].astype(np.uint32); assert not set(int(h) for h in TAILH) & set(int(h) for h in c.hashes)
    # The clip keeps its key count and rate (the client plays re-timed 58 / 116 fps clips in slow motion). Where the motion
    # the engine interpolates BETWEEN two keys dips into the ground, the two keys around it get a larger margin instead.
    K = 1; trail = []; xm = np.zeros((c.F, 3)); THR = np.array([SOLE_FLOOR - 1.5, 0.3, 0.3]); CAP = np.array([8.0, 45.0, 45.0]); best = None
    loop = any(rel_out.endswith(n + '.anm') for n in LOOPS)
    if getattr(c, 'free', False): THR[1] = -1e9; CAP[0] = 24.0      # the scissors are planted in the ground on purpose; keys eased from / onto the idle stance only get the un-eased share of a margin
    for n, (k0, k1, dy) in ENTRY_LIFT.items():
        if rel_out.endswith(n + '.anm'): xm[k0:k1, 0] = dy
    for it in range(MARGIN_ITERS):
        res = retarget(c, rel_out, K, xm.copy()); T, R, S_ = res[:3]
        snap(c, T, R, S_, hashes, K); pin_idle(c, T, R, S_, hashes)
        fy, sy, ty, gap, D = clearance(hashes, T, R, S_, SUB, True)
        trail.append(f'#{it}: soles {fy:.2f} scissors {sy:.2f} tail {ty:.2f}')
        bad = float(np.maximum(THR - np.array([fy, sy, ty]), 0).sum())
        if best is None or bad < best[0]: best = (bad, res, (fy, sy, ty, gap), xm.copy())
        if bad == 0: break
        need = np.maximum(THR[None] - D, 0); need = np.where(need > 0, need + 0.4, 0); inc = np.zeros_like(xm)
        inc[:-1] = need; inc[1:] = np.maximum(inc[1:], need)
        if loop: inc[0] = inc[-1] = np.maximum(inc[0], inc[-1])
        xm = np.minimum(xm + inc, CAP[None])
    bad, res, (fy, sy, ty, gap), xm = best
    T, R, S_, missing, mx, nover, lf, gd, tf, stretch = res
    if hasattr(c, 'snapres'): del c.snapres
    snap(c, T, R, S_, hashes, K)      # (pin_idle already ran on these arrays inside the loop)
    if xm.max() > 0: note += f' [between-key margins: soles {xm[:, 0].max():.1f} scissors {xm[:, 1].max():.1f} tail {xm[:, 2].max():.1f}]'
    if len(trail) > 1: print(rel_out[len(A):], ' | '.join(trail))
    if hasattr(c, 'snapres'): note += f' [hand-off residual removed: {c.snapres[0]:.2f} u / {c.snapres[1]:.1f} deg]'
    if hasattr(c, 'fxW'):
        # Seven unweighted, root-level knife anchors already exist on the upright rig. Reuse them
        # for the source recall's eyes and ghost guns; no weighted joint or skin bind is altered.
        a = as_anim(hashes,T,R,S_); hm = {int(h):i for i,h in enumerate(hashes)}
        for f in range(len(T)):
            W = L.pose_world(skN,a,f)
            for dst,ws in c.fxW.items():
                j=skN.index[dst]; ti=hm[L.elf(dst)]; assert skN.parents[j] < 0
                assert not ((_infN == j) * mN.w).any()
                X=ws[f].copy(); u,_,vt=np.linalg.svd(X[:3,:3]); X[:3,:3]=u@vt
                if dst in ('knife_04','knife_06'): X[:3,3] += W[skN.index['r_hand']][:3,3]-c.fxH['R_Hand'][f]
                if dst in ('knife_05','knife_07'): X[:3,3] += W[skN.index['l_hand']][:3,3]-c.fxH['L_Hand'][f]
                T[f,ti]=X[:3,3]; R[f,ti]=L.mat_to_quat(X[:3,:3]); S_[f,ti]=1.0
        ti=hm[L.elf('BUFFBONE_GLB_WEAPON_1')]
        T[:,ti]=_vl[np.argmax(np.linalg.norm(_vl,axis=1))]; R[:,ti]=L.mat_to_quat(c.fxTipR); S_[:,ti]=1.0
        note += ' [17 source recall particle events; eyes/guns/tip anchors keyed]'
    size = write(rel_out, c, T, R, S_, hashes, K)
    POSES[rel_out[len(A):-4]] = (hashes.copy(), (T[0].copy(), R[0].copy(), S_[0].copy()), (T[-1].copy(), R[-1].copy(), S_[-1].copy()))
    _a = as_anim(hashes, T, R, S_); FINAL[rel_out[len(A):-4]] = (ymin(FEET_N, L.pose_world(skN, _a, 0) @ skN.ibind), ymin(FEET_N, L.pose_world(skN, _a, len(T) - 1) @ skN.ibind))   # sole height of the first / last pose
    log(f'{rel_out[len(A):]:38s} {c.F:4d} keys @ {c.fps:4.1f} fps -> x{K} = {len(T):4d} frames   {note:52s} weapon swing {mx[0]:5.1f} twist {mx[1]:5.1f} ({nover:3d} keys limited)  '
        f'limb scale removed {stretch:4.2f}  root lift {lf[0]:5.1f}..{lf[1]:4.1f}  scissors lift max {gd[0]:4.1f} deg on {gd[1]:3d} fr  tail corr {tf[0]:6.1f}..{tf[1]:5.1f} deg  '
        f'| interpolated motion (x{SUB} sub-samples): soles min y {fy:5.2f}  scissors min y {sy:5.2f}  tail min y {ty:5.2f}  grip gap max {gap:4.2f}' + (f'  MISSING TRACKS {missing}' if missing else ''))
    out[rel_out] = size

# --- E wall-vault (long / medium): the base mod pointed these at clips made for another skeleton (body collapses).
#     Use the intact clips that ship unused in the same folder, with the health-bar anchor track of the "healthbar_fixed" versions.
for kind in ('long', 'medium'):
    c = load(A + f'talon_spell3_{kind}.anm'); hb = L.load_anm(SRC + f'skin20/animations/talon_skin20_spell3_{kind}_healthbar_fixed.anm')
    assert len(bind_frames(c)) == 0
    hname = 'C_Buffbone_Cstm_Healthbar_Loc'; t = c.tr(hname); hm = {int(h): i for i, h in enumerate(hb.hashes)}; t2 = hm[L.elf(hname)]
    assert hb.nframes == c.F
    c.fps = hb.fps      # preserve the 30 fps resource rate used by the upright base graph
    c.T[:, t] = hb.T[:, t2]; c.R[:, t] = hb.R[:, t2] / np.linalg.norm(hb.R[:, t2], axis=-1, keepdims=True); c.S[:, t] = hb.S[:, t2]
    emit(A + f'talon_spell3_{kind}.anm', c, 'intact body clip + fixed health-bar track; ' + ease_in_idle(c, 2, 5))

for nm in PLAIN:
    c = load(A + nm + '.anm'); bf = bind_frames(c); note = ''
    if nm == 'talon_skin12_homeguard':
        _run = load(A + 'talon_skin12_run.anm'); close_loop(_run); SPRINT = make_sprint(_run); c = copy.deepcopy(SPRINT); bf = []; note = 'custom sprint: run cycle, torso leaning in, free arm and tail streaming back'
    if nm == 'talon_skin12_homeguard_in':
        c = make_sprint_in(_run, SPRINT, c.F); bf = []; note = 'custom: run eased into the sprint'
    if len(bf):
        # stray bind-pose frames must be a contiguous prefix; rebuild them as an ease from the idle stance into the first real frame
        assert np.array_equal(bf, np.arange(len(bf))), (nm, bf)
        k = len(bf)
        for f in range(k):
            w = (f / k); w = w * w * (3 - 2 * w)
            c.T[f], c.R[f], c.S[f] = blend_pose(idle, 0, c, k, w)
        note = f'frames 0-{k-1} were T-pose -> eased idle>f{k}'
    if nm == 'talon_skin12_idle1': c = idle; note = _idle_note
    elif nm in CHANNELS: note = upright(c)
    else: note = (note + ' ' + destance(c, crouch, idle)).strip()
    if nm in QUICK: note = (note + ' ' + retime(c, QUICK[nm])).strip()
    if nm in EASE_IN: note = (note + ' ' + ease_in_idle(c, *EASE_IN[nm])).strip()
    if nm in EASE_END: note = (note + ' ' + ease_out_idle(c, EASE_END[nm])).strip()
    if nm in EASE_FROM: note = (note + ' ' + ease_in_from(c, *EASE_FROM[nm])).strip()
    if nm in EASE_OUT: note = (note + ' ' + ease_out_to(c, *EASE_OUT[nm])).strip()
    if nm in LOOPS: note = (note + ' ' + close_loop(c)).strip()
    emit(A + nm + '.anm', c, note)

# --- Recall: the recall of the Ellen Joe Samira skin. That mod plays Riot's Soul Fighter Samira recall on Samira's skeleton
#     (it ships no animation files of its own); the motion is read from a public export of that skin's animations. The Ellen mesh
#     of this rig is, vertex for vertex, the mesh of that mod (body scaled 0.744, arms re-posed, scissors scaled 0.615), so the
#     motion is carried over by fitting every joint of this rig to where its own vertices sit in the Samira-mod pose - no joint
#     naming or bind-pose guesswork. The scissors follow the sword of that choreography: on her back, swung, thrown, planted in
#     the ground, and she ends perched on them. (The shipped Talon recall targets a different rig: limbs fold, body 250 units up.)
def _kabsch(P, Q, w):
    pc = (w[:, None] * P).sum(0) / w.sum(); qc = (w[:, None] * Q).sum(0) / w.sum(); Hm = ((Q - qc) * w[:, None]).T @ (P - pc)
    U, sg, Vt = np.linalg.svd(Hm); Dm = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]); Rm = U @ Dm @ Vt; return Rm, qc - Rm @ pc
def _sim(P, Q):
    pc = P.mean(0); qc = Q.mean(0); U, sg, Vt = np.linalg.svd((Q - qc).T @ (P - pc)); Dm = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]); Rm = U @ Dm @ Vt
    sc = float((sg * np.diag(Dm)).sum() / ((P - pc) ** 2).sum()); return sc, Rm, qc - sc * Rm @ pc
def samira_recall(F, fps, base):
    from glbanim import Glb
    DS = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'; skS = L.load_skl(DS + 'samira_skin30.skl'); mS = L.load_skn(DS + 'samira_skin30.skn')
    g = Glb(S + '/dl/samira30/model.glb'); assert np.array_equal(mS.idx, mN.idx) and np.abs(mS.uv - mN.uv).max() < 1e-6, 'the two Ellen meshes must correspond vertex for vertex'
    T_END = 244 / 30.0      # she is perched on the scissors; after this Samira leaps out of the map (the Talon graph has no arrival clip for that)
    times = np.minimum(np.arange(F) / fps, T_END); smp = g.sample('Recall', times); nmS = [n for n in skS.names if n in smp]
    aS = as_anim(np.array([L.elf(n) for n in nmS], np.uint32), np.stack([smp[n][0] for n in nmS], 1), np.stack([smp[n][1] for n in nmS], 1), np.stack([smp[n][2] for n in nmS], 1))
    face = next(np.arange(vs, vs + vc) for n, vs, vc, i0, ic in mS.subs if n == 'Ellen_Face_D'); sb = _sim(mS.pos[face], mN.pos[face])[0]      # body scale
    sf, Rf, tf = _sim(mS.pos[_sel], mN.pos[_sel]); assert np.abs(sf * mS.pos[_sel] @ Rf.T + tf - mN.pos[_sel]).max() < 0.05                     # scissors: similarity between the two bind poses
    JS = skS.index['Sword']; JHS = skS.index['R_Buffbone_Glb_Hand_Loc']; infS = np.array(skS.influences)[mS.infl]
    infN = np.array(skN.influences)[mN.infl]; bw = np.linalg.inv(skN.ibind); P1 = np.c_[mN.pos, np.ones(len(mN.pos))]
    feet = np.where(sum(((infN == skN.index[n]) * mN.w).sum(1) for n in ('l_foot', 'r_foot', 'l_toe', 'r_toe')) > 0.5)[0]
    reg = {}
    for j, n in enumerate(skN.names):
        if j in TAILJ or j == JW: continue
        wj = ((infN == j) * mN.w).sum(1); sel = np.where(wj > 0.3)[0]
        if len(sel) >= 16: reg[j] = (sel, wj[sel])
    for j in reg: assert int(skN.parents[j]) < 0 or int(skN.parents[j]) in reg, skN.names[j]
    from recall_vfx import ANCHORS
    fxnames = {src:dst for src,dst in ANCHORS.items() if dst.startswith('knife_')}
    FX = {dst:[] for dst in fxnames.values()}; FXH = {side:[] for side in ('L_Hand','R_Hand')}
    VS = []; MS = []; HS = []
    for f in range(F):
        W = L.pose_world(skS, aS, f); VS.append(L.deform(mS, skS, W)[0]); MS.append(W[JS] @ skS.ibind[JS]); HS.append(W[JHS][:3, 3])
        for src,dst in fxnames.items(): FX[dst].append(W[skS.index[src]].copy())
        for side in FXH: FXH[side].append(W[skS.index[side]][:3,3].copy())
    VS = np.array(VS); MS = np.array(MS); HS = np.array(HS); tk = lambda fr: int(round(fr / 30.0 * fps))      # Samira frame (30 fps) -> key of this clip
    # where on the scissors her hand holds them (keys with the sword in the right hand), and the point that ends up in the ground
    held = np.arange(tk(111), tk(123) + 1); ah = np.array([np.linalg.inv(MS[f]) @ np.r_[HS[f], 1] for f in held])[:, :3]; assert ah.std(0).max() < 4.0, ah.std(0); a_hold = ah.mean(0)
    kp = tk(180); a_tip = mS.pos[_sel][np.argmin(VS[kp][_sel][:, 1])]; assert VS[kp][_sel][:, 1].min() < 0 < VS[kp][feet][:, 1].min() + 6
    u_tip = sstep((times * 30.0 - 158.0) / 4.0)      # anchor moves from the hand-hold to the tip while she slams them into the ground
    # She lands on the planted scissors. These scissors are 17 % shorter against her body than in the Samira mod, so the spot
    # under her feet sits lower and closer to the tip: shift the perched body by that much, faded in over the jump.
    pk = np.arange(tk(229), tk(244) + 1); dl = []
    for f in pk:
        lo = VS[f][feet][np.argsort(VS[f][feet][:, 1])[:30]].mean(0); sv = VS[f][_sel]; cpt = sv[np.argmin(np.linalg.norm(sv - lo, axis=1))]; dl.append(cpt - (MS[f] @ np.r_[a_tip, 1])[:3])
    d_perch = -(sb - sf) * np.mean(dl, 0); u_p = sstep((times * 30.0 - 216.0) / 10.0)
    VT = sb * VS + u_p[:, None, None] * d_perch[None, None]
    # scissors path in the world. Planted, the source blade still twitches from key to key (compression noise of the export: 3.7 u
    # on average, 23 u once); single keys up to 15 u off); it is a blade stuck in the ground, so that phase is median-filtered and low-passed (the slow sag under her weight stays).
    QS = np.array([L.mat_to_quat(MS[f][:3, :3]) for f in range(F)])
    for f in range(1, F):
        if QS[f] @ QS[f - 1] < 0: QS[f] *= -1
    AT = np.array([sb * (MS[f] @ np.r_[(1 - u_tip[f]) * a_hold + u_tip[f] * a_tip, 1])[:3] for f in range(F)])
    kq = tk(163); ws = sstep((np.arange(F) - kq) / 5.0); QSs = QS.copy(); ATs = AT.copy()
    def _med(x, r=6): xp = np.pad(x, r, mode='edge'); return np.median(np.stack([xp[i:i + len(x)] for i in range(2 * r + 1)]), 0)      # the noise comes as single-key spikes (15 u): median first
    QSs[kq:] = np.stack([smooth(_med(QS[kq:, i]), 3.0) for i in range(4)], 1); ATs[kq:] = np.stack([smooth(_med(AT[kq:, i]), 3.0) for i in range(3)], 1)
    QS = (1 - ws[:, None]) * QS + ws[:, None] * QSs; QS /= np.linalg.norm(QS, axis=1, keepdims=True); AT = (1 - ws[:, None]) * AT + ws[:, None] * ATs
    wW = np.zeros((F, 4, 4)); bWw = bw[JW]
    for f in range(F):
        a_S = (1 - u_tip[f]) * a_hold + u_tip[f] * a_tip; RS = L.quat_to_mat(QS[f]); Mw = np.eye(4); Mw[:3, :3] = RS @ Rf.T
        Mw[:3, 3] = AT[f] - RS @ (Rf.T @ tf + sf * a_S); wW[f] = Mw @ bWw
    held = np.linalg.norm(np.array([(MS[f] @ np.r_[a_hold, 1])[:3] for f in range(F)]) - HS, axis=1) < 12.0; u_h = np.clip(smooth(held.astype(float), 1.0), 0, 1)      # scissors in her right hand
    STOW = (24.0, 20.0, 8.0, 22.0)      # roll about the back's normal (deg), side, up, back (units); found with stowfit.py (smallest offset that keeps the blade out of body and tail)
    JCHEST = skN.index['chest']; _u, _, _vt = np.linalg.svd(bw[JCHEST][:3, :3]); RCB = _u @ _vt; c_sc = mN.pos[_sel].mean(0)
    k_grab = int(np.argmax(u_h > 0.5)); u_s = 1 - sstep((np.arange(F) - (k_grab - 11)) / 9.0); print('recall: scissors grabbed on key', k_grab)
    c = Clip.__new__(Clip); c.hashes = base.hashes.copy(); c.t = dict(base.t); c.fps = fps; c.F = F
    c.T = np.repeat(base.T[:1], F, 0); c.R = np.repeat(base.R[:1], F, 0); c.S = np.ones_like(c.T)
    err = []; nJ = len(skN.names); onm = {n: i for i, n in enumerate(old_names)}; JRH = skN.index['r_hand']; drop = np.zeros(F); reach = 0
    LEGS = [tuple(skN.index[x + n] for n in ('thigh', 'knee', 'foot')) for x in ('l_', 'r_')]; FV = [np.where(sum(((infN == skN.index[x + n]) * mN.w).sum(1) for n in ('foot', 'toe')) > 0.5)[0] for x in ('l_', 'r_')]
    def _swing(u, v):
        u = u / np.linalg.norm(u); v = v / np.linalg.norm(v); ax = np.cross(u, v); n_ = np.linalg.norm(ax)
        if n_ < 1e-9: return np.eye(3)
        return L.quat_to_mat(np.r_[ax / n_ * np.sin(np.arctan2(n_, u @ v) / 2), np.cos(np.arctan2(n_, u @ v) / 2)])
    DROP = None; npass = [0]
    for f in list(range(F)) + list(range(F)):      # pass 1: how far the pelvis has to come down per key; pass 2: solve with that curve smoothed over time
        npass[0] += 1
        if npass[0] == F + 1: DROP = np.maximum(drop, smooth(dilate(drop, 2), 2.0)); print('recall: pelvis lowered on', int((drop > 0).sum()), 'keys, max', drop.max(), ' top keys', np.argsort(-drop)[:12].tolist()); reach = 0
        fit = {j: _kabsch(mN.pos[sel], VT[f][sel], wj) for j, (sel, wj) in reg.items()}; Rw = {j: fit[j][0] @ bw[j][:3, :3] for j in reg}
        proot = fit[JROOT][0] @ bw[JROOT][:3, 3] + fit[JROOT][1]
        # Legs. Riot's clip stretches Samira's legs through translation tracks (5 units most of the time, 74 in the high kick); Ellen's
        # legs keep their length. Feet go where the source puts them by two-bone IK (thigh and shin keep the fitted twist, the knee
        # bends in the fitted plane); where a standing foot would be out of reach the pelvis comes down, in a kick the leg just ends extended.
        ph = {}; tgt = {}
        def hips(dy):
            pr = proot - np.array([0, dy, 0]); phip = pr + Rw[JROOT] @ skN.lt[JHIP]
            return pr, {jt: phip + Rw[JHIP] @ skN.lt[jt] for jt, jk, jf in LEGS}
        for (jt, jk, jf), fv in zip(LEGS, FV):
            tgt[jt] = (fit[jf][0] @ bw[jf][:3, 3] + fit[jf][1], fit[jk][0] @ bw[jk][:3, 3] + fit[jk][1], VT[f][fv][:, 1].min() < 4.0, np.linalg.norm(skN.lt[jk]), np.linalg.norm(skN.lt[jf]))
        if DROP is None:
            for dy in np.arange(0, 15.01, 0.25):
                pr, ph = hips(dy)
                if all((not tgt[jt][2]) or np.linalg.norm(tgt[jt][0] - ph[jt]) <= 0.998 * (tgt[jt][3] + tgt[jt][4]) for jt, jk, jf in LEGS): break
            drop[f] = dy; continue
        pr, ph = hips(DROP[f])
        for jt, jk, jf in LEGS:
            ank, knf, grd, l1, l2 = tgt[jt]; v = ank - ph[jt]; d0 = np.linalg.norm(v); d = float(np.clip(d0, abs(l1 - l2) + 0.5, 0.998 * (l1 + l2))); reach += d0 > d + 0.5; nrm = v / d0
            pole = (knf - ph[jt]) - ((knf - ph[jt]) @ nrm) * nrm; pole = pole / max(np.linalg.norm(pole), 1e-6); x = (l1 * l1 - l2 * l2 + d * d) / (2 * d); kn = ph[jt] + x * nrm + np.sqrt(max(l1 * l1 - x * x, 0.0)) * pole; an = ph[jt] + d * nrm
            Rw[jt] = _swing(Rw[jt] @ skN.lt[jk], kn - ph[jt]) @ Rw[jt]; Rw[jk] = _swing(Rw[jk] @ skN.lt[jf], an - kn) @ Rw[jk]
        Wn = np.zeros((nJ, 4, 4))
        for j in range(nJ):
            pj = int(skN.parents[j]); n = skN.names[j]; q = skN.lr[j] / np.linalg.norm(skN.lr[j]); tl = skN.lt[j].copy()
            if j in reg:
                if pj < 0: q = L.mat_to_quat(Rw[j]); tl = pr
                else: q = L.mat_to_quat(Wn[pj][:3, :3].T @ Rw[j])
                t = c.tr(n); c.R[f, t] = q; c.T[f, t] = tl - ((lt[idx[n]] - old_lt[onm[n]]) if n in onm else 0.0)      # (retarget() adds the re-proportioning back)
            Ml = L.trs(tl, q, np.ones(3)); Wn[j] = Ml if pj < 0 else Wn[pj] @ Ml
        # in her hand the scissors go with this rig's hand, which (rigid arm, lowered pelvis) can sit a little off the source's
        wW[f][:3, 3] += u_h[f] * (Wn[JRH][:3, 3] - (fit[JRH][0] @ bw[JRH][:3, 3] + fit[JRH][1])) - (1 - u_h[f]) * (1 - u_tip[f]) * np.array([0, DROP[f], 0])
        # On her back the source runs the blade through skirt, bow and tail root (Samira's sword sits flat on a slimmer back). While
        # they are stowed the scissors are carried off the back instead: a fixed offset in the chest's frame, gone before she grabs them.
        if u_s[f] > 0:
            _u, _, _vt = np.linalg.svd(Wn[JCHEST][:3, :3]); Cc = (_u @ _vt) @ RCB.T; th = np.radians(STOW[0]) * u_s[f]
            Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]); Rk = Cc @ Rz @ Cc.T; pc = (wW[f] @ skN.ibind[JW] @ np.r_[c_sc, 1])[:3]
            Mk = np.eye(4); Mk[:3, :3] = Rk; Mk[:3, 3] = pc - Rk @ pc + u_s[f] * (Cc @ np.array([STOW[1], STOW[2], -STOW[3]])); wW[f] = Mk @ wW[f]
        if f % 8 == 0:
            M = Wn @ skN.ibind; body = np.setdiff1d(np.arange(len(mN.pos)), np.r_[_sel, np.where(TAIL_ISJ[infN[:, 0]])[0]]); out = np.zeros((len(body), 3))
            for k in range(4): out += mN.w[body, k:k + 1] * np.einsum('nij,nj->ni', M[infN[body, k]][:, :3, :], P1[body])
            err.append(np.linalg.norm(out - VT[f][body], axis=1))
    for j in reg:
        n = skN.names[j]
        if n in onm and int(skN.parents[j]) >= 0: assert np.abs(skN.lt[j] - lt[idx[n]]).max() < 1e-3, n
    c.fxW = {dst:np.array(ws) for dst,ws in FX.items()}; c.fxH = {side:sb*np.array(ps)+u_p[:,None]*d_perch for side,ps in FXH.items()}
    for dst,ws in c.fxW.items(): ws[:,:3,3] = sb*ws[:,:3,3] + u_p[:,None]*d_perch
    c.fxTipR = bw[JW][:3,:3].T @ Rf @ np.linalg.inv(skS.ibind[skS.index['Sword_Tip2']])[:3,:3]
    c.wU = 1 - u_tip      # share of the ground lift the scissors take along (1 while they are on her / in her hand, 0 once they are in the ground)
    c.R /= np.linalg.norm(c.R, axis=-1, keepdims=True)
    c.yref = VT[:, feet, 1].min(1); c.free = True; c.wW = wW
    HOLD, EASE, K_END, OUT_N = 4, 7, tk(244), 13      # open on the idle stance (the engine cross-fades into the clip there); hop off the scissors back into it at the end
    k = np.arange(F); c.wP = np.where(k < HOLD, 1.0, 1.0 - sstep((k - HOLD + 1) / (EASE + 1.0))); c.wP = np.maximum(c.wP, sstep((k - K_END) / float(OUT_N)))
    err = np.concatenate(err)
    return c, (f'Samira-mod recall re-fitted to this rig: {len(reg)} joints fitted to their own vertices (body x{sb:.3f}, scissors x{sf:.3f}); skin vs the Samira-mod pose '
               f'mean {err.mean():.2f} / 99% {np.percentile(err, 99):.2f} / max {err.max():.2f} u (legs keep their length: {int(reach)} leg-keys end extended short of a stretched source leg, pelvis lowered up to {DROP.max():.1f} u for standing feet); scissors held {np.round(a_hold, 1).tolist()}, perch shift {np.round(d_perch, 1).tolist()}; '
               f'idle stance f0-{HOLD + EASE - 1} and f{K_END + 1}-{F - 1}')
ch = load(A + 'talon_skin12_channel.anm'); old = L.load_anm(SRC + A + 'talon_recall.anm'); _b = copy.deepcopy(ch)
for h, t1 in _b.t.items():      # tracks nothing is fitted to (root-level attachment points, unused Talon joints) keep the idle's first pose
    t0 = idle.t.get(h)
    if t0 is not None: _b.T[0, t1] = idle.T[0, t0]; _b.R[0, t1] = idle.R[0, t0]; _b.S[0, t1] = idle.S[0, t0]
rc, _rnote = samira_recall(old.nframes, old.fps, _b)
emit(A + 'talon_recall.anm', rc, _rnote)
pickle.dump(out, open(S + '/' + OUT + '/anims.pkl', 'wb'))
log('attachment points pinned to their bind transform in every clip (base tracks carry scale 0/-1 and rotation flips):', ', '.join(PINNED))
log('wrote', len(out), 'animations')
