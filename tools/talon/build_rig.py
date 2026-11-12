"""build_rig.py S  ->  builds new SKL + SKN for Ellen on the upright (Aemeath / Wild Rift) Talon rig.
Outputs to S/build/wad/... and S/build/rig.npz (bind data used by build_anim.py)."""
import sys, os, struct, pickle, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
from pyritofile import write_skn

B = 'assets/ice/wild/bum/characters/talon/skins/'
AEM = S + '/w/aemeath_talon/' + B + 'skin12/'
SAM = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'
OUT = S + '/build/wad/' + B + 'skin12/'
os.makedirs(OUT + 'ellen', exist_ok=True)

# ---------------- tunables ----------------
SCISSOR_REL = 0.826          # scissors size relative to body (same ratio the WIP used)
GRIP_Z = -63.0               # grip point along the long handle rod (raw scissors z)
GAMMA = 60.0                 # rod angle from hand-forward axis inside the fist (saber grip)
DELTA = 40.0                 # ulnar deviation of the right wrist baked into the bind
TAIL_N = 16
MCP_AT = 0.70                # knuckle line position between wrist and the first rigged finger joint
ROD_DEPTH = 0.25             # rod centre depth relative to the palm surface (raw units)
ROD_FWD = 0.3                # rod centre offset forward of the knuckle line under the middle finger
FINGER_HALF = 0.55           # finger half thickness
THUMB_TGT = (0.6, -0.8, 0.2) # thumb-tip target relative to the rod under the index finger (along f, along l, extra depth)
THUMB_FLEX = (28, 38)
TIP_LEN = 2.6                # length of the last finger segment (raw units)

def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
def rot_axis(axis, deg):
    a = unit(axis); t = np.radians(deg); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
def swing(a, b):
    a, b = unit(a), unit(b); c = np.cross(a, b); s = np.linalg.norm(c); d = a @ b
    if s < 1e-9: return np.eye(3)
    return rot_axis(c / s, np.degrees(np.arctan2(s, d)))
def about(p, R):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p - R @ p; return M

ak = L.load_skl(AEM + 'talon_skin12.skl')
sk = L.load_skl(SAM + 'samira_skin30.skl'); m = L.load_skn(SAM + 'samira_skin30.skn')
J = lambda n: sk.world[sk.index[n], :3, 3].copy()
T = lambda n: ak.world[ak.index[n], :3, 3].copy()
TR = lambda n: ak.world[ak.index[n], :3, :3].copy()
sub = {n: (vs, vc) for n, vs, vc, i0, ic in m.subs}
sinf = np.array(sk.influences)[m.infl]            # (V,4) samira joint ids

# ---------------- 1. right-hand grip pose baked in raw (Samira) space ----------------
w = J('R_Hand'); f = unit(J('R_Middle1') - w); l = J('R_Index1') - J('R_Pinky1'); l = unit(l - f * (l @ f)); n = np.cross(f, l)
curl = sum(unit(J(f'R_{x}2') - J(f'R_{x}1')) - unit(J(f'R_{x}1') - w) for x in ['Index', 'Middle', 'Ring', 'Pinky'])
palm = n * np.sign((J('R_Thumb1') - w) @ n)       # thumb base sits on the palm side
assert palm[0] < 0 and palm[1] < 0, 'right palm should face down/inward in the A-pose'
print('right hand frame f', f.round(2), 'l', l.round(2), 'palm', palm.round(2), 'curl.n', round(float(curl @ n), 3), 'thumb.palm', round(float((J('R_Thumb1') - w) @ palm), 2))
G = rot_axis(-n, DELTA)                              # ulnar deviation: f rotates toward -l
# --- scissors handle geometry
vs, vc = sub['sword']; SW = m.pos[vs:vs+vc]
capsel = (SW[:, 2] > -62) & (SW[:, 2] < -52); rod_xy = (SW[capsel][:, :2].min(0) + SW[capsel][:, :2].max(0)) / 2
rod_r = float(np.abs(SW[capsel][:, :2] - rod_xy).max()) * SCISSOR_REL
print('rod centre xy', rod_xy.round(2), 'radius (body raw units)', round(rod_r, 2))
# --- fist solver. The Samira rig only articulates the two outer finger joints (X1 = PIP, X2 = DIP); the knuckle (MCP) bend
# a fist needs is applied as a smooth extra bend of everything beyond the knuckle line. Angles are solved per finger in the
# (along-finger, toward-palm) plane so each finger wraps the handle rod that crosses the palm diagonally.
FING = ['Index', 'Middle', 'Ring', 'Pinky']
kc = unit(np.cross(f, palm))                         # curl axis: rotates f toward the palm
a0 = MCP_AT * ((J('R_Middle1') - w) @ f)             # knuckle line
mid_side = (J('R_Middle1') - w) @ l
hsel = np.where(((sinf == sk.index['R_Hand']) * m.w).sum(1) > 0.6)[0]
hv = m.pos[hsel] - w; along = hv @ f; side = hv @ l; up = hv @ palm
psel = (along > a0 - 4.0) & (along < a0 - 0.5) & (np.abs(side - mid_side) < 2.0)
palm_h = float(up[psel].max())
rho_c = palm_h + ROD_DEPTH                            # rod centre depth below the joint plane
RHO = rod_r + FINGER_HALF
gam = np.radians(GAMMA)
def rod_centre(s_lat): return np.array([a0 + ROD_FWD + (s_lat - mid_side) / np.tan(gam), rho_c])
def tangent_dir(Pt, C):
    v = C - Pt; d_ = np.linalg.norm(v); return np.arctan2(v[1], v[0]) - np.arcsin(min(1.0, RHO / d_))
MCP, PIP, DIP, A0 = {}, {}, {}, {}
def turn(Pt, C, t_prev, mx):
    d_ = (tangent_dir(Pt, C) - t_prev) % (2 * np.pi)
    return 0.0 if d_ > np.radians(300) else min(d_, np.radians(mx))
for x in FING:
    p1, p2 = J(f'R_{x}1'), J(f'R_{x}2'); s_x = (p1 - w) @ l; C = rod_centre(s_x)
    A0[x] = MCP_AT * ((p1 - w) @ f); L1 = (p1 - w) @ f - A0[x]; L2 = np.linalg.norm(p2 - p1); Pt = np.array([A0[x], 0.0])
    t1 = turn(Pt, C, 0.0, 90); Pt = Pt + L1 * np.array([np.cos(t1), np.sin(t1)])
    t2 = t1 + turn(Pt, C, t1, 105); Pt = Pt + L2 * np.array([np.cos(t2), np.sin(t2)])
    t3 = t2 + turn(Pt, C, t2, 80)
    MCP[x], PIP[x], DIP[x] = np.degrees(t1), np.degrees(t2 - t1), np.degrees(t3 - t2)
    print(f'  {x:6s} lateral {s_x:5.2f} rod centre a-a0 {C[0]-A0[x]:5.2f} depth {C[1]:.2f}  L1 {L1:.2f} L2 {L2:.2f}  MCP {MCP[x]:5.1f} PIP {PIP[x]:5.1f} DIP {DIP[x]:5.1f}')
# thumb: aim it across the handle toward the index finger, then flex the two outer joints around the rod
p1, p2, p3 = J('R_Thumb1'), J('R_Thumb2'), J('R_Thumb3')
s_i = (J('R_Index1') - w) @ l; Ci = rod_centre(s_i)
def hp(a_, s_, dpt): return w + a_ * f + s_ * l + dpt * palm
rod_pt = hp(Ci[0], s_i, Ci[1]); tgt = hp(Ci[0] + THUMB_TGT[0], s_i + THUMB_TGT[1], Ci[1] + RHO + THUMB_TGT[2])
Ra = swing(p3 - p1, tgt - p1)
kt = unit(np.cross(Ra @ (p2 - p1), rod_pt - p1))
# The source finger weights are far too soft to articulate with LBS, so the grip is posed geometrically: every right-hand
# vertex is assigned to the nearest finger/thumb bone segment and rotated about the joints with a short smooth blend.
def sstep(x_): x_ = np.clip(x_, 0, 1); return x_ * x_ * (3 - 2 * x_)
def seg_dist(Pp, A_, B_):
    ab = B_ - A_; t_ = np.clip((Pp - A_) @ ab / (ab @ ab), 0, 1); return np.linalg.norm(Pp - (A_ + np.outer(t_, ab)), axis=1), t_
def axis_angle(Rm):
    ang = np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))
    if ang < 1e-9: return np.array([1.0, 0, 0]), 0.0
    return unit([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]), np.degrees(ang)
hand_j = [sk.index['R_Hand']] + [sk.index[f'R_{x}{i}'] for x in FING for i in (1, 2)] + [sk.index[f'R_Thumb{i}'] for i in (1, 2, 3)]
hw_ = sum(((sinf == j) * m.w).sum(1) for j in hand_j)
rv = np.where(hw_ > 0)[0]; Pr = m.pos[rv].copy(); Nr = m.nrm[rv].copy()
segs = []   # (label, A, B)
for x in FING:
    p1, p2 = J(f'R_{x}1'), J(f'R_{x}2'); mc = w + (p1 - w) * MCP_AT
    segs += [('palm', w, mc), (x, mc, p1), (x, p1, p2), (x, p2, p2 + unit(p2 - p1) * TIP_LEN)]
t1_, t2_, t3_ = J('R_Thumb1'), J('R_Thumb2'), J('R_Thumb3')
segs += [('thenar', w, t1_), ('Thumb', t1_, t2_), ('Thumb', t2_, t3_), ('Thumb', t3_, t3_ + unit(t3_ - t2_) * TIP_LEN)]
dists = np.stack([seg_dist(Pr, A_, B_)[0] for _, A_, B_ in segs], 1); lab = np.array([segs[i][0] for i in dists.argmin(1)])
print('right-hand vertex labels:', {k_: int((lab == k_).sum()) for k_ in sorted(set(lab))})
Pq = Pr.copy(); Nq = Nr.copy(); BL = 0.6
def rot_pts(sel, centre, axis, degs):
    for ii in np.where(sel)[0]:
        if degs[ii] == 0: continue
        Rm = rot_axis(axis, degs[ii]); Pq[ii] = centre + Rm @ (Pq[ii] - centre); Nq[ii] = Rm @ Nq[ii]
a_ = (Pr - w) @ f; s_lat = (Pr - w) @ l
lat = np.array([(J(f'R_{x}1') - w) @ l for x in FING]); od = np.argsort(lat)
th = np.interp(s_lat, lat[od], np.array([MCP[x] for x in FING])[od]); a0v = np.interp(s_lat, lat[od], np.array([A0[x] for x in FING])[od])
for x in FING:
    p1, p2 = J(f'R_{x}1'), J(f'R_{x}2'); sel = lab == x; d2 = unit(p2 - p1)
    rot_pts(sel, p2, kc, DIP[x] * sstep(((Pr - p2) @ d2 + BL) / (2 * BL)))
    rot_pts(sel, p1, kc, PIP[x] * sstep(((Pr - p1) @ d2 + BL) / (2 * BL)))
    th[sel] = MCP[x]; a0v[sel] = A0[x]
nonthumb = (lab != 'Thumb') & (lab != 'thenar')
u = sstep((a_ - (a0v - 1.0)) / 2.0) * nonthumb
for ii in np.where(u > 0)[0]:
    c0 = w + a0v[ii] * f; Rm = rot_axis(kc, u[ii] * th[ii]); Pq[ii] = c0 + Rm @ (Pq[ii] - c0); Nq[ii] = Rm @ Nq[ii]
# thumb
selT = (lab == 'Thumb'); k_rest = Ra.T @ kt
rot_pts(selT, t3_, k_rest, THUMB_FLEX[1] * sstep(((Pr - t3_) @ unit(t3_ - t2_) + BL) / (2 * BL)))
rot_pts(selT, t2_, k_rest, THUMB_FLEX[0] * sstep(((Pr - t2_) @ unit(t3_ - t2_) + BL) / (2 * BL)))
ax_a, deg_a = axis_angle(Ra)
ua = np.where(selT, sstep(((Pr - t1_) @ unit(t2_ - t1_) + 1.0) / 2.0), 0.0)
then = lab == 'thenar'; ua = np.where(then, 0.6 * sstep((seg_dist(Pr, w, t1_)[1] - 0.45) / 0.55) * sstep((np.linalg.norm(Pr - t1_, axis=1) < 3.0) * 1.0), ua)
rot_pts(ua > 0, t1_, ax_a, deg_a * ua)
# wrist deviation, blended with the source wrist weights
for ii in range(len(rv)):
    h = hw_[rv[ii]]; Pq[ii] = h * (w + G @ (Pq[ii] - w)) + (1 - h) * Pq[ii]; Nq[ii] = h * (G @ Nq[ii]) + (1 - h) * Nq[ii]
pos = m.pos.copy(); nrm = m.nrm.copy(); pos[rv] = Pq; nrm[rv] = Nq
nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
D = {'R_Hand': about(w, G)}

# ---------------- 2. scissors placement (raw body space) ----------------
g0 = hp(a0 + ROD_FWD, mid_side, rho_c)                # point of the rod axis under the middle finger (un-posed hand frame)
g = (D['R_Hand'] @ np.r_[g0, 1])[:3]
d_rod = G @ unit(np.cos(gam) * f + np.sin(gam) * l)    # blade direction
e_edge = G @ unit(np.sin(gam) * f - np.cos(gam) * l)   # cutting-edge side
b_raw = unit([-1, -1, 0]); z_raw = np.array([0, 0, 1.0]); c_raw = np.cross(z_raw, b_raw)
Rsc = np.stack([e_edge, np.cross(d_rod, e_edge), d_rod], 1) @ np.stack([b_raw, c_raw, z_raw], 1).T
q = np.array([rod_xy[0], rod_xy[1], GRIP_Z])
pos[vs:vs+vc] = SCISSOR_REL * (SW - q) @ Rsc.T + g
nrm[vs:vs+vc] = m.nrm[vs:vs+vc] @ Rsc.T
fa = unit(J('R_Hand') - J('R_Elbow'))
print('palm surface %.2f  rod depth %.2f  blade vs forearm axis: %.1f deg' % (palm_h, rho_c, np.degrees(np.arccos(d_rod @ fa))))

# ---------------- 3. raw -> target space ----------------
body = np.ones(len(pos), bool); body[vs:vs+vc] = False
leg = lambda s_: np.linalg.norm(J(s_ + '_Hip') - J(s_ + '_KneeLower')) + np.linalg.norm(J(s_ + '_KneeLower') - J(s_ + '_Foot'))
tl = lambda s_: np.linalg.norm(T(s_ + '_thigh') - T(s_ + '_knee')) + np.linalg.norm(T(s_ + '_knee') - T(s_ + '_foot'))
sc = (tl('l') + tl('r')) / (leg('L') + leg('R'))
foot_sel = np.where(((sinf == sk.index['L_Foot']) | (sinf == sk.index['R_Foot'])).any(1) & body)[0]
ysole = m.pos[foot_sel, 1].min()
dz = (T('l_thigh')[2] + T('r_thigh')[2]) / 2 - sc * (J('L_Hip')[2] + J('R_Hip')[2]) / 2
Xf = lambda p: sc * (np.asarray(p, float) - np.array([0, ysole, 0])) + np.array([0, 0, dz])
E = lambda nme: Xf(J(nme))
LOWER = (T('l_thigh')[1] + T('r_thigh')[1]) / 2 - (E('L_Hip')[1] + E('R_Hip')[1]) / 2
print(f'scale {sc:.4f} ysole {ysole:.2f} dz {dz:.2f} LOWER {LOWER:.2f}  height {sc * (m.pos[body, 1].max() - ysole):.1f}')
pos = Xf(pos)

# ---------------- 4. new bind pose ----------------
names = list(ak.names); parents = list(ak.parents); NJ0 = len(names)
P = np.zeros((NJ0, 3)); Rw = np.zeros((NJ0, 3, 3)); A = np.zeros((NJ0, 3, 3))
def hand_frame(wr, mid, idx, pk):
    f_ = unit(mid - wr); l_ = idx - pk; l_ = unit(l_ - f_ * (l_ @ f_)); return np.stack([f_, l_, np.cross(f_, l_)], 1)
def tal_virtual(nm):   # right hand lacks index/pinky joints: mirror the left hand's local offsets (as the rig does for middle/thumb)
    return (ak.world[ak.index['r_hand']] @ np.r_[-ak.lt[ak.index[nm]], 1])[:3]
POS = {  # explicit world positions
    'root': T('root') - [0, LOWER, 0], 'hip': T('hip') - [0, LOWER, 0],
    'spine': E('Spine2'), 'chest': E('Chest'), 'neck': E('Neck'), 'head': E('Head'),
}
SW_RULE = {}   # joint -> (talon child joint name or direction, ellen direction)
for s_, S_ in (('l', 'L'), ('r', 'R')):
    POS[f'{s_}_shoulder'] = E(f'{S_}_Clavicle'); POS[f'{s_}_uparm'] = E(f'{S_}_Shoulder'); POS[f'{s_}_forearm'] = E(f'{S_}_Elbow'); POS[f'{s_}_hand'] = E(f'{S_}_Hand')
    POS[f'{s_}_thigh'] = E(f'{S_}_Hip'); POS[f'{s_}_knee'] = E(f'{S_}_KneeLower'); POS[f'{s_}_foot'] = E(f'{S_}_Foot')
    SW_RULE[f'{s_}_uparm'] = (T(f'{s_}_forearm') - T(f'{s_}_uparm'), E(f'{S_}_Elbow') - E(f'{S_}_Shoulder'))
    SW_RULE[f'{s_}_forearm'] = (T(f'{s_}_hand') - T(f'{s_}_forearm'), E(f'{S_}_Hand') - E(f'{S_}_Elbow'))
    SW_RULE[f'{s_}_thigh'] = (T(f'{s_}_knee') - T(f'{s_}_thigh'), E(f'{S_}_KneeLower') - E(f'{S_}_Hip'))
    SW_RULE[f'{s_}_knee'] = (T(f'{s_}_foot') - T(f'{s_}_knee'), E(f'{S_}_Foot') - E(f'{S_}_KneeLower'))
for fn, FN in (('index', 'Index'), ('middle', 'Middle'), ('ring', 'Ring'), ('pinky', 'Pinky'), ('thumb', 'Thumb')):
    POS[f'l_{fn}'] = E(f'L_{FN}1'); POS[f'l_{fn}_b'] = E(f'L_{FN}2')
    SW_RULE[f'l_{fn}'] = (T(f'l_{fn}_b') - T(f'l_{fn}'), E(f'L_{FN}2') - E(f'L_{FN}1'))
IDENT = {'root', 'hip', 'spine', 'chest', 'neck', 'head', 'l_shoulder', 'r_shoulder', 'l_foot', 'r_foot'}
FRAME = {'l_hand': hand_frame(E('L_Hand'), E('L_Middle1'), E('L_Index1'), E('L_Pinky1')) @ hand_frame(T('l_hand'), T('l_middle'), T('l_index'), T('l_pinky')).T}
SCALE_LOCAL = {}
for s_, S_ in (('l', 'L'), ('r', 'R')):
    ratio = np.linalg.norm(E(f'{S_}_Hand') - E(f'{S_}_Elbow')) / np.linalg.norm(T(f'{s_}_hand') - T(f'{s_}_forearm'))
    SCALE_LOCAL[f'{S_}_BUFFBONE_GLB_HAND_LOC'] = ratio
order = []
def visit(i):
    if i in order: return
    if parents[i] >= 0: visit(parents[i])
    order.append(i)
for i in range(NJ0): visit(i)
for i in order:
    nm = names[i]; p = parents[i]
    Ap = A[p] if p >= 0 else np.eye(3); Rp = Rw[p] if p >= 0 else np.eye(3); Pp = P[p] if p >= 0 else np.zeros(3)
    Bj = ak.world[i, :3, :3]
    if nm in IDENT: A[i] = np.eye(3)
    elif nm in SW_RULE: dT, dE = SW_RULE[nm]; A[i] = swing(Ap @ unit(dT), dE) @ Ap
    elif nm in FRAME: A[i] = FRAME[nm]
    else: A[i] = Ap
    Rw[i] = A[i] @ Bj
    if nm in POS: P[i] = POS[nm]
    elif nm in SCALE_LOCAL: P[i] = Pp + Rp @ (ak.lt[i] * SCALE_LOCAL[nm])
    elif nm == 'weapon': P[i] = POS['r_hand']
    elif nm == 'r_hand': pass
    else: P[i] = Pp + Rp @ ak.lt[i]
    if nm == 'r_hand':      # right hand is rigidly locked to the weapon joint (same orientation, same pivot)
        wi = ak.index['weapon']; fi = ak.index['r_forearm']
        Rw[i] = A[fi] @ ak.world[wi, :3, :3]; A[i] = Rw[i] @ Bj.T; P[i] = POS['r_hand']
idx = {nm: i for i, nm in enumerate(names)}
# sanity: weapon joint must come after r_forearm and share r_hand's bind rotation
assert np.allclose(Rw[idx['weapon']], Rw[idx['r_hand']], atol=1e-9), 'weapon/hand bind mismatch'
# head buffbone -> centre of Ellen's head
hsel2 = np.where(((sinf == sk.index['Head']) * m.w).sum(1) > 0.99)[0]
hc = (pos[hsel2].min(0) + pos[hsel2].max(0)) / 2; P[idx['C_BUFFBONE_GLB_HEAD_LOC']] = [0, hc[1], P[idx['head']][2] + 1.0]
# weapon buffbone -> on the scissors, at the blade base
P[idx['BUFFBONE_GLB_WEAPON_1']] = Xf(SCISSOR_REL * (np.array([rod_xy[0] - 6, rod_xy[1] - 6, 55.0]) - q) @ Rsc.T + g)
# tail chain (bind orientation copied from the Samira rig so the engine's path-conform modifier sees the same chain)
for k_ in range(1, TAIL_N + 1):
    names.append(f'tail{k_}'); parents.append(idx['hip'] if k_ == 1 else len(names) - 2)
    sj = sk.index[f'joint{k_}']
    P = np.vstack([P, Xf(sk.world[sj, :3, 3])]); Rw = np.concatenate([Rw, sk.world[sj, :3, :3][None]]); A = np.concatenate([A, np.eye(3)[None]])
idx = {nm: i for i, nm in enumerate(names)}; NJ = len(names); parents = np.array(parents)
for i in range(NJ):   # source quats are float32: project every bind rotation onto the nearest exact rotation
    assert np.abs(Rw[i] @ Rw[i].T - np.eye(3)).max() < 1e-4 and np.linalg.det(Rw[i]) > 0.999, names[i]
    U_, _, Vt_ = np.linalg.svd(Rw[i]); Rw[i] = U_ @ Vt_
lt = np.zeros((NJ, 3)); lr = np.zeros((NJ, 4)); it = np.zeros((NJ, 3)); ir = np.zeros((NJ, 4))
for i in range(NJ):
    p = parents[i]
    if p < 0: lt[i] = P[i]; lr[i] = L.mat_to_quat(Rw[i])
    else: lt[i] = Rw[p].T @ (P[i] - P[p]); lr[i] = L.mat_to_quat(Rw[p].T @ Rw[i])
    it[i] = -Rw[i].T @ P[i]; ir[i] = L.mat_to_quat(Rw[i].T)

# ---------------- 5. weights ----------------
MAP = {'Root': 'hip', 'Pelvis': 'hip', 'Spine1': 'root', 'Spine2': 'spine', 'Chest': 'chest', 'Neck': 'neck', 'Head': 'head', 'Sword': 'weapon'}
for s_, S_ in (('l', 'L'), ('r', 'R')):
    MAP.update({f'{S_}_Clavicle': f'{s_}_shoulder', f'{S_}_Shoulder': f'{s_}_uparm', f'{S_}_Elbow': f'{s_}_forearm', f'{S_}_Hand': f'{s_}_hand',
                f'{S_}_Hip': f'{s_}_thigh', f'{S_}_KneeLower': f'{s_}_knee', f'{S_}_Foot': f'{s_}_foot'})
for fn, FN in (('index', 'Index'), ('middle', 'Middle'), ('ring', 'Ring'), ('pinky', 'Pinky')):
    MAP[f'L_{FN}1'] = f'l_{fn}'; MAP[f'L_{FN}2'] = f'l_{fn}_b'; MAP[f'R_{FN}1'] = 'r_hand'; MAP[f'R_{FN}2'] = 'r_hand'
MAP.update({'L_Thumb1': 'l_thumb', 'L_Thumb2': 'l_thumb_b', 'L_Thumb3': 'l_thumb_b', 'R_Thumb1': 'r_hand', 'R_Thumb2': 'r_hand', 'R_Thumb3': 'r_hand'})
for k_ in range(1, TAIL_N + 1): MAP[f'joint{k_}'] = f'tail{k_}'
V = len(pos); Wd = np.zeros((V, NJ))
for k_ in range(4):
    for sj in np.unique(sinf[:, k_]):
        sel = (sinf[:, k_] == sj) & (m.w[:, k_] > 0)
        if not sel.any(): continue
        assert sk.names[sj] in MAP, 'unmapped weighted joint ' + sk.names[sj]
        Wd[sel, idx[MAP[sk.names[sj]]]] += m.w[sel, k_]
assert (Wd.sum(1) > 0.99).all()
top = np.argsort(-Wd, axis=1)[:, :4]; tw = np.take_along_axis(Wd, top, 1); tw /= tw.sum(1, keepdims=True); top[tw == 0] = 0
used = sorted(set(top[tw > 0].tolist())); print('joints carrying weight:', len(used), [names[i] for i in used])

# ---------------- 6. write SKL ----------------
def write_skl(path, names, parents, lt, lr, it, ir, ls=None, isc=None):
    n = len(names); jo = 64; io = jo + n * 100; fo = io + n * 8; no = fo + n * 2
    blob = bytearray(); offs = []
    for nm in names: offs.append(no + len(blob)); blob += nm.encode('ascii') + b'\0'
    out = bytearray(no) + blob
    struct.pack_into('<3I2HI6i5I', out, 0, len(out), 0x22FD4FC3, 0, 0, n, n, jo, io, fo, 0, 0, no, *([0xFFFFFFFF] * 5))
    hs = [L.elf(nm) for nm in names]
    for i in range(n):
        o = jo + i * 100
        struct.pack_into('<HhhHIf', out, o, 0, i, int(parents[i]), 0, hs[i], 2.1)
        struct.pack_into('<3f3f4f3f3f4f', out, o + 16, *lt[i], *((1, 1, 1) if ls is None else ls[i]), *lr[i], *it[i], *((1, 1, 1) if isc is None else isc[i]), *ir[i])
        struct.pack_into('<i', out, o + 96, offs[i] - (o + 96))
    for k_, i in enumerate(sorted(range(n), key=lambda i: hs[i])): struct.pack_into('<HHI', out, io + k_ * 8, i, 0, hs[i])
    for i in range(n): struct.pack_into('<H', out, fo + i * 2, i)
    open(path, 'wb').write(out)
# writer self-test: re-emit the Aemeath skeleton and compare with the original file
write_skl(S + '/out/_selftest.skl', ak.names, ak.parents, ak.lt, ak.lr, ak.it, ak.ir, ak.ls, ak.isc)
a_, b_ = open(AEM + 'talon_skin12.skl', 'rb').read(), open(S + '/out/_selftest.skl', 'rb').read()
diff = [i for i in range(len(a_)) if a_[i] != b_[i]]
io_ = 64 + 104 * 100
assert len(a_) == len(b_) and all(i < 4 or io_ <= i < io_ + 104 * 8 for i in diff), 'SKL writer mismatch'
print('SKL writer self-test OK (differs from original only in size field + hash-sorted index table: %d bytes)' % len(diff))
assert len(set(L.elf(nm) for nm in names)) == NJ, 'joint hash collision'
write_skl(OUT + 'talon_skin12.skl', names, parents, lt, lr, it, ir)

# ---------------- 7. write SKN ----------------
raw = m.raw
for i, v in enumerate(raw.vertices):
    v.position.x, v.position.y, v.position.z = map(float, pos[i]); v.normal.x, v.normal.y, v.normal.z = map(float, nrm[i])
    v.influences = tuple(int(x) for x in top[i]); v.weights = tuple(float(x) for x in tw[i])
write_skn(OUT + 'ellen/ellen_talon.skn', raw)
print('wrote', OUT + 'talon_skin12.skl', os.path.getsize(OUT + 'talon_skin12.skl'), OUT + 'ellen/ellen_talon.skn', os.path.getsize(OUT + 'ellen/ellen_talon.skn'))
print('mesh bbox', pos.min(0).round(1), pos.max(0).round(1), 'body bbox', pos[body].min(0).round(1), pos[body].max(0).round(1))
pickle.dump(dict(dbg=dict(d=d_rod, palm=G @ palm, g=Xf(g), f=G @ f, l=G @ l), names=names, parents=parents, lt=lt, lr=lr, P=P, Rw=Rw, old_names=list(ak.names), old_lt=ak.lt, old_lr=ak.lr, LOWER=LOWER, sc=sc), open(S + '/build/rig.pkl', 'wb'))
for nm in ['root', 'hip', 'spine', 'chest', 'neck', 'head', 'l_shoulder', 'l_uparm', 'l_forearm', 'l_hand', 'r_uparm', 'r_forearm', 'r_hand', 'weapon', 'l_thigh', 'l_knee', 'l_foot', 'l_toe', 'r_thigh', 'r_foot', 'tail1', 'tail16', 'C_BUFFBONE_GLB_HEAD_LOC', 'BUFFBONE_GLB_WEAPON_1']:
    i = idx[nm]; o = ak.index.get(nm)
    print(f'  {nm:26s} new {P[i].round(1)}  old {ak.world[o, :3, 3].round(1) if o is not None else "-"}  A-angle {np.degrees(np.arccos(np.clip((np.trace(A[i]) - 1) / 2, -1, 1))):.1f}')

# ---------------- grip diagnostics (target space) ----------------
gT = Xf(g); dT_ = d_rod; pT = G @ palm; eT = unit(np.cross(dT_, pT))
print('grip check: rod radius %.2f (target units)' % (rod_r * sc))
for x in FING + ['Thumb']:
    sel = np.zeros(len(pos), bool); sel[rv[lab == x]] = True
    rel = pos[sel] - gT; ax = rel @ dT_; perp = rel - np.outer(ax, dT_); rad = np.linalg.norm(perp, axis=1)
    ang = np.degrees(np.arctan2(perp @ eT, perp @ pT))
    print(f'  {x:6s} verts {sel.sum():3d} dist to rod axis min {rad.min():.2f} med {np.median(rad):.2f} max {rad.max():.2f}  angle around rod {ang.min():6.1f}..{ang.max():6.1f}  along rod {ax.min():5.1f}..{ax.max():5.1f}')
