"""validate.py S FANTOME -> unpacks the packaged mod into S/final and checks every file in it from scratch. Report: S/final/validation.txt"""
import sys, os, io, json, struct, zipfile, shutil, glob
import numpy as np
S = sys.argv[1]; FAN = sys.argv[2]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binutil import *
import lolio as L
from pyritofile import WAD
FIN = S + '/' + (sys.argv[3] if len(sys.argv) > 3 else 'final'); shutil.rmtree(FIN, ignore_errors=True); os.makedirs(FIN + '/w')
rep = open(FIN + '/validation.txt', 'w'); fails = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); rep.write(s + '\n')
def check(ok, msg):
    log(('  PASS  ' if ok else '  FAIL  ') + msg)
    if not ok: fails.append(msg)
with zipfile.ZipFile(FAN) as z:
    nl = z.namelist(); log('fantome entries:', nl); check(z.testzip() is None, 'zip CRCs')
    check(set(nl) == {'META/info.json', 'META/image.png', 'WAD/talon.wad.client'}, 'fantome layout META/info.json + META/image.png + WAD/talon.wad.client')
    info = json.loads(z.read('META/info.json')); log('info.json:', info)
    check(all(k in info for k in ('Name', 'Author', 'Version', 'Description')), 'info.json has Name/Author/Version/Description')
    check(z.read('META/image.png')[:8] == b'\x89PNG\r\n\x1a\n', 'image.png is a PNG')
    wadb = z.read('WAD/talon.wad.client'); open(FIN + '/talon.wad.client', 'wb').write(wadb)
wad = WAD(); wad.read(FIN + '/talon.wad.client'); chunks = {}
with wad.stream(FIN + '/talon.wad.client', 'rb') as bs:
    for c in wad.chunks: c.read_data(bs); chunks[c.hash] = c.data; assert c.data is not None and len(c.data) == c.decompressed_size
hs = [int(c.hash, 16) for c in wad.chunks]
log(f'\nWAD v{wad.version}: {len(chunks)} chunks, {len(wadb)} bytes')
check(hs == sorted(hs) and len(set(hs)) == len(hs), 'WAD table of contents sorted by hash, no duplicate entries')
check(all(c.offset + c.compressed_size <= len(wadb) for c in wad.chunks), 'all chunk data inside the file')
BINP = 'data/characters/talon/skins/skin0.bin'; check(ph(BINP) in chunks, 'skin bin present at ' + BINP)
b = BIN(); b.read('', raw=chunks[ph(BINP)]); check(b.write('', raw=True) == chunks[ph(BINP)], 'bin parses and re-serialises byte-identically')
tables = load_tables(S + '/hashes'); ent = {e.hash: e for e in b.entries}
nm = lambda h: tables['binhashes'].get(h) or tables['binentries'].get(h) or '0x' + h
log('bin entries by class:', {k: sum(1 for e in b.entries if tables['bintypes'].get(e.type) == k) for k in sorted({tables['bintypes'].get(e.type, e.type) for e in b.entries})}, 'links:', b.links)
# ---- references
refs = {}
for s_ in strings(b):
    l = s_.lower()
    if '/' in l and '.' in l.rsplit('/', 1)[-1] and (l.startswith('assets/') or l.startswith('data/')): refs[ph(l)] = l
game = set(line[:16] for line in open(S + '/hashes/hashes.game.txt', 'r', encoding='utf-8', errors='replace'))
miss = sorted(l for h, l in refs.items() if h not in chunks and h not in game)
log(f'asset paths in bin: {len(refs)}'); 
check(miss == ['assets/ice/wild/bum/sounds/wwise2016/vo/en_us/characters/talon/skins/base/talon_base_vo_audio.bnk'], f'every referenced asset is shipped (only the inherited VO _audio.bnk entry is absent, as in both source mods): {miss}')
extra = [h for h in chunks if h not in refs and h != ph(BINP)]; check(not extra, f'no unreferenced chunks in the WAD ({len(extra)})')
for h, l in refs.items():
    if h in chunks:
        p = FIN + '/w/' + l; os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'wb').write(chunks[h])
p = FIN + '/w/' + BINP; os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'wb').write(chunks[ph(BINP)])
# ---- every chunk has a sane signature for its extension
sig = {'.anm': (b'r3d2anmd', b'r3d2canm'), '.skn': (bytes.fromhex('33221100'),), '.tex': (b'TEX\0',), '.dds': (b'DDS ',), '.scb': (b'r3d2Mesh',), '.sco': (b'[Obj',), '.bnk': (b'BKHD',), '.wpk': (b'r3d2',), '.bin': (b'PROP', b'PTCH')}
bad = [l for h, l in refs.items() if h in chunks and os.path.splitext(l)[1] in sig and not chunks[h].startswith(sig[os.path.splitext(l)[1]])]
check(not bad, f'file signatures match extensions for all {sum(1 for h in refs if h in chunks)} shipped assets {bad}')
bad = []
for h, l in refs.items():
    if h in chunks and l.endswith('.tex'):
        w, hh, u1, fmt, u2, mips = struct.unpack_from('<HHBBBB', chunks[h], 4); bpb = {10: 8, 11: 8, 12: 16}.get(fmt)
        if fmt == 20: need = w * hh * 4
        elif bpb: need = max(1, (w + 3) // 4) * max(1, (hh + 3) // 4) * bpb
        else: bad.append((l, 'fmt', fmt)); continue
        if len(chunks[h]) - 12 < need: bad.append((l, len(chunks[h]), need))
check(not bad, f'all .tex files have a known format and enough pixel data {bad}')
# ---- skin entry
skin = ent[fh('Characters/Talon/Skins/Skin0')]; smp = get(skin.data, 'skinMeshProperties')
sklp = get(smp.data, 'skeleton').data.lower(); sknp = get(smp.data, 'simpleSkin').data.lower(); texp = get(smp.data, 'texture').data.lower()
check(all(ph(x) in chunks for x in (sklp, sknp, texp)), 'skeleton / simpleSkin / texture of skinMeshProperties are in the WAD')
sk = L.load_skl(FIN + '/w/' + sklp); m = L.load_skn(FIN + '/w/' + sknp)
raw = open(FIN + '/w/' + sklp, 'rb').read()
log(f'\nSKL: {len(sk.names)} joints, {len(sk.influences)} influences, file {len(raw)} bytes, size field {struct.unpack_from("<I", raw, 0)[0]}, magic {raw[4:8].hex()}')
check(struct.unpack_from('<I', raw, 0)[0] == len(raw) and raw[4:8].hex() == 'c34ffd22', 'SKL header size field and magic')
check(len(set(n.lower() for n in sk.names)) == len(sk.names) and len(set(int(h) for h in sk.hashes)) == len(sk.names), 'joint names and name hashes unique')
check(all(-1 <= p < len(sk.names) for p in sk.parents) and sum(1 for p in sk.parents if p < 0) >= 1, 'parent indices valid')
def depth(i, n=0):
    while sk.parents[i] >= 0: i = sk.parents[i]; n += 1; assert n < 200
    return n
check(all(depth(i) < 200 for i in range(len(sk.names))), 'no cycles in the hierarchy')
err = max(np.abs(sk.world[i] @ sk.ibind[i] - np.eye(4)).max() for i in range(len(sk.names))); check(err < 2e-3, f'inverse-bind matrices are the inverse of the bind pose (max error {err:.2e})')
dets = [np.linalg.det(sk.world[i][:3, :3]) for i in range(len(sk.names))]; check(min(dets) > 0.999 and max(dets) < 1.001, f'bind rotations are pure rotations, unit scale (det {min(dets):.5f}..{max(dets):.5f})')
A = L.load_skl(S + '/w/aemeath_talon/assets/ice/wild/bum/characters/talon/skins/skin12/talon_skin12.skl')
check(sk.names[:len(A.names)] == A.names and [int(p) for p in sk.parents[:len(A.names)]] == [int(p) for p in A.parents], 'first 104 joints = the base rig joints, same order and parents')
check(sk.names[len(A.names):] == [f'tail{i}' for i in range(1, 17)], 'plus tail1..tail16 appended')
# joint-index table sorted by hash (engine binary-searches it)
hdr = struct.unpack_from('<IIIHHIIiiiiii', raw, 0)
log('SKL header fields:', hdr)
# ---- SKN
log(f'\nSKN: {len(m.pos)} verts, {len(m.idx)} tris, submeshes {[(n, vc, ic // 3) for n, vs, vc, i0, ic in m.subs]}')
check(np.isfinite(m.pos).all() and np.isfinite(m.nrm).all() and np.isfinite(m.uv).all(), 'vertex data finite')
check(np.abs(m.w.sum(1) - 1).max() < 1e-3 and (m.w >= 0).all(), f'vertex weights non-negative and sum to 1 (max dev {np.abs(m.w.sum(1) - 1).max():.1e})')
check(m.infl.max() < len(sk.influences), f'influence indices within the skeleton influence list ({m.infl.max()} < {len(sk.influences)})')
check(m.idx.max() < len(m.pos) and m.idx.min() >= 0, 'triangle indices in range')
ok = True
for n, vs, vc, i0, ic in m.subs:
    tri = m.idx.reshape(-1)[i0:i0 + ic]; ok &= bool(tri.min() >= vs and tri.max() < vs + vc)
check(ok, 'each submesh only indexes its own vertex range')
nl_ = np.linalg.norm(m.nrm, axis=1); check(nl_.min() > 0.99 and nl_.max() < 1.01, 'normals unit length')
used = sorted({sk.names[sk.influences[i]] for i in np.unique(m.infl[m.w > 0])}); log('joints carrying weight:', len(used), used)
check(len(m.pos) <= 65535, 'vertex count fits 16-bit indices')
p0, _ = L.deform(m, sk, sk.world); check(np.abs(p0 - m.pos).max() < 1e-2, f'skinning the bind pose reproduces the mesh (max dev {np.abs(p0 - m.pos).max():.1e})')
log(f'mesh bbox {np.round(m.pos.min(0), 1)} .. {np.round(m.pos.max(0), 1)}')
ovr = {get(e_.data, 'submesh').data: get(e_.data, 'texture').data for e_ in get(smp.data, 'materialOverride').data}
check(set(ovr) <= {n for n, *_ in m.subs}, f'materialOverride submeshes exist in the SKN {sorted(ovr)}')
mat = get(smp.data, 'Material').data; check(mat in ent and tables['bintypes'].get(ent[mat].type) == 'StaticMaterialDef', f"skin Material link resolves to StaticMaterialDef '{get(ent[mat].data, 'name').data}'")
rp = get(smp.data, 'rigPoseModifierData'); bones = {fh(n): n for n in sk.names}
if rp is None: check(True, 'no rigPoseModifierData: the tail is keyed in every clip (nothing in the engine can bend it into the scissors)')
else: mod = rp.data[0]; check(tables['bintypes'].get(mod.hash_type) == 'ConformToPathRigPoseModifierData' and bones.get(get(mod.data, 'mStartingJointName').data) == 'tail1' and bones.get(get(mod.data, 'mEndingJointName').data) == 'tail16', 'tail modifier ConformToPath tail1 -> tail16 resolves to skeleton joints')
ti = [sk.index[f'tail{i}'] for i in range(1, 17)]; check(all(sk.parents[ti[i]] == ti[i - 1] for i in range(1, 16)) and sk.names[sk.parents[ti[0]]] == 'hip', 'tail joints form one chain hanging off hip')
check(fh(get(get(skin.data, 'healthBarData').data, 'attachToBone').data) in bones, 'health-bar bone exists')
# ---- bone / submesh names used anywhere in the bin
subm = {fh(n) for n, *_ in m.subs}; badr = []
for e in b.entries:
    for f in walk(e.data):
        fn = tables['binfields'].get(f.hash, '') if f.hash else ''
        for ty, v in values(f):
            if fn == 'mBoneName' and v not in bones: badr.append((fn, nm(v)))
            if fn in ('mShowSubmeshList', 'mHideSubmeshList') and v not in subm: badr.append((fn, nm(v)))
check(not badr, f'every animation-event bone / submesh exists on the model {badr}')
# ---- animation graph
g = ent[get(get(skin.data, 'skinAnimationProperties').data, 'animationGraphData').data]; clips = get(g.data, 'mClipDataMap'); keys = {k for k, p in clips.data}
log(f'\nanimation graph {nm(g.hash)}: {len(clips.data)} clips')
jh = {int(h): n for n, h in zip(sk.names, sk.hashes)}; rows = []; allok = True; dangling = []
# geometry used for the motion checks (independent of the build scripts): soles, scissors, tail, and the points where the
# right fist touches the scissors handle in the bind pose
SUBS = 16
infl = np.array(sk.influences)[m.infl]
def wsel(names): return np.where(sum(((infl == sk.index[n]) * m.w).sum(1) for n in names if n in sk.index) > 0.5)[0]
v_feet = wsel(['l_foot', 'r_foot', 'l_toe', 'r_toe']); v_tail = wsel([f'tail{i}' for i in range(1, 17)])
v_sw = next(np.arange(vs, vs + vc) for n, vs, vc, i0, ic in m.subs if n == 'sword')
def under(j, a):
    while j >= 0:
        if j == a: return True
        j = int(sk.parents[j])
    return False
hand_j = [j for j in range(len(sk.names)) if under(j, sk.index['r_hand'])]
v_hand = np.setdiff1d(np.where((np.isin(infl, hand_j) * m.w).sum(1) > 0.5)[0], v_sw)
_d = np.array([np.linalg.norm(m.pos[v_sw] - m.pos[v], axis=1).min() for v in v_hand]); grip = np.c_[m.pos[v_hand[_d < 0.75]], np.ones(int((_d < 0.75).sum()))]
def ysub(sel, M):
    P = np.c_[m.pos[sel], np.ones(len(sel))]; y = np.zeros(len(sel))
    for k in range(infl.shape[1]): y += m.w[sel, k] * np.einsum('nj,nj->n', M[infl[sel, k], 1, :], P)
    return float(y.min())
def between(a, K):
    """the clip as the engine plays it between stored frames: lerp T/S, shortest-arc slerp R"""
    F = a.nframes; i = np.arange(K * (F - 1) + 1); f0 = np.minimum(i // K, F - 2); t = (i - f0 * K) / K; t3 = t[:, None, None]
    b = L.Anim(); b.hashes = a.hashes; b.sig = a.sig; b.has = a.has
    b.T = (1 - t3) * a.T[f0] + t3 * a.T[f0 + 1]; b.S = (1 - t3) * a.S[f0] + t3 * a.S[f0 + 1]; b.R = L.slerp(a.R[f0], a.R[f0 + 1], np.repeat(t[:, None], a.R.shape[1], 1)); return b
evrows = []; motion_ok = True; grip_ok = True; scale_ok = True; hover_ok = True; mrows = []; v_body = np.setdiff1d(np.arange(len(m.pos)), np.r_[v_sw, v_tail])
# clips that leave the ground on purpose: the E dashes and their landings, the Q leap, the ult flourish, the death fall, and the
# run loops (flight phase in every stride). Every other clip is a standing action and must keep a foot on the ground.
# (Whole clips are exempt, so this test says nothing about the grounded parts of those clips. The recall jumps, flips and ends perched on the planted scissors.)
AIRBORNE = ('Spell3', 'Spell3_Long', 'Spell3_Medium', '0x86b44e3c', 'Spell1_Leap', 'Spell4', 'Death', 'Run_Normal', 'Run_Homeguard', 'Run_Homeguard_IN', '0x9260a8e8')
# The recall (0x9260a8e8) is the choreography of the Ellen Joe Samira skin: the scissors leave her hand (on her back, thrown, driven
# into the ground, she ends perched on them). It is the one clip whose scissors may be out of the fist and in the ground; it is gated on:
# fist closed on the handle on its first four keys and on its last one (what the cross-fades into and out of it see); at the stored
# keys the scissors are under the ground during ONE unbroken stretch only (planted once, pulled out once); never deeper than
# PLANT_DEPTH, also between keys; and while planted the blade stands still from key to key (<= PLANT_STEP units, after the impact).
# Between two keys the engine interpolates the scissors under her moving forearm, so a planted blade wobbles a little while she
# moves (most during the flip over it); that is reported, not gated - the Samira original keys its sword under her chest the same way.
RECALL_NAME = '0x9260a8e8'; PLANT_DEPTH = -12.0; PLANT_STEP = 3.0; STOW_MAX = 0.02; recall_rows = []
for k, p in clips.data:
    cls = tables['bintypes'].get(p.hash_type)
    for f in walk(p):
        fn = tables['binfields'].get(f.hash, '') if f.hash else ''
        if fn in ('mClipNameList', 'mTrueConditionClipName', 'mFalseConditionClipName'):
            for ty, v in values(f):
                if v not in keys: dangling.append((nm(k), nm(v)))
    if cls != 'AtomicClipData': continue
    path = get(get(p.data, 'mAnimationResourceData').data, 'mAnimationFilePath').data.lower()
    if ph(path) not in chunks: allok = False; log('   MISSING', nm(k), path); continue
    a = L.load_anm(FIN + '/w/' + path); unk = [hex(h) for h in a.hashes if int(h) not in jh]; untracked = [n for n, h in zip(sk.names, sk.hashes) if int(h) not in set(int(x) for x in a.hashes)]
    source_path = path.replace('assets/ice/wild/bum/characters/', S + '/w/aemeath_talon/assets/ice/wild/bum/characters/')
    for kind in ('long', 'medium'):
        if path.endswith('/talon_spell3_' + kind + '.anm'):
            source_path = S + '/w/aemeath_talon/assets/ice/wild/bum/characters/talon/skins/skin20/animations/talon_skin20_spell3_' + kind + '_healthbar_fixed.anm'
    source = L.load_anm(source_path)
    check(a.nframes == source.nframes and abs(a.fps - source.fps) < 1e-4, 'source key count/rate preserved: ' + nm(k))
    qn = np.linalg.norm(a.R, axis=-1); fin = np.isfinite(a.T).all() and np.isfinite(a.R).all() and np.isfinite(a.S).all()
    jump = float(np.degrees(2 * np.arccos(np.clip(np.abs((a.R[1:] * a.R[:-1]).sum(-1)), 0, 1))).max()) if a.nframes > 1 else 0
    good = (not unk) and fin and abs(qn - 1).max() < 1e-3 and a.sig == 'r3d2anmd' and a.version == 4 and untracked == [] and abs(a.fps - 29.0) < 1.5 and len(set(int(h) for h in a.hashes)) == len(a.hashes) and (a.S > 0.05).all()
    allok &= good; rows.append(f'   {"ok " if good else "BAD"} {nm(k):20s} {os.path.basename(path):40s} v{a.version} {a.nframes:4d} fr @{a.fps:5.1f} fps {len(a.hashes)} tracks, max per-frame joint rotation {jump:5.1f} deg, |T|max {np.abs(a.T).max():6.1f}' + (f' UNKNOWN TRACKS {unk}' if unk else ''))
    b = between(a, SUBS); fy = sy = ty = 1e9; gap = 0.0; hov = -1e9; jr = sk.index['r_hand']; jw = sk.index['weapon']; _gs = []; _ss = []
    for f in range(len(b.T)):
        M = L.pose_world(sk, b, f) @ sk.ibind; _f = ysub(v_feet, M)
        fy = min(fy, _f); hov = max(hov, _f); sy = min(sy, ysub(v_sw, M)); ty = min(ty, ysub(v_tail, M))
        _g = float(np.linalg.norm((grip @ M[jr].T - grip @ M[jw].T)[:, :3], axis=1).max()); gap = max(gap, _g); _gs.append(_g); _ss.append(ysub(v_sw, M))
    smax = float(np.abs(a.S - 1).max()); ok_m = sy >= 0 and ty >= 0 and fy >= -2.0; ok_g = gap <= 1.0; ok_s = smax < 1e-4
    if nm(k) == RECALL_NAME:
        _gs = np.array(_gs); _ss = np.array(_ss); _x = np.arange(len(_gs)) / SUBS; _out = np.where(_gs > 1.0)[0]; _uk = np.where(_ss[::SUBS] < 0)[0]
        _one = len(_uk) > 0 and np.all(np.diff(_uk) == 1); ok_g = _gs[:3 * SUBS + 1].max() <= 1.0 and _gs[-1] <= 1.0
        _ext = np.c_[m.pos[v_sw][[m.pos[v_sw][:, i].argmin() for i in range(3)] + [m.pos[v_sw][:, i].argmax() for i in range(3)]], np.ones(6)]      # six extreme points of the scissors
        _P = np.array([_ext @ (L.pose_world(sk, b, f)[jw] @ sk.ibind[jw]).T for f in range(len(b.T))])[:, :, :3]; _k0, _k1 = int(_uk[0]) + 6, int(_uk[-1]) - 1      # (from the impact having settled to the last key before she pulls them out)
        _step = float(np.linalg.norm(np.diff(_P[_k0 * SUBS:_k1 * SUBS + 1:SUBS], axis=0), axis=2).max())
        _dev = np.array([max(np.linalg.norm(_P[q * SUBS + j] - ((1 - j / SUBS) * _P[q * SUBS] + j / SUBS * _P[(q + 1) * SUBS]), axis=1).max() for j in range(1, SUBS)) for q in range(_k0, _k1)])
        ok_m = ty >= 0 and fy >= -2.0 and _one and _ss.min() >= PLANT_DEPTH and _ss[_gs <= 1.0].min() >= 0 and _step <= PLANT_STEP
        recall_rows.append(f'   recall: scissors out of the fist on keys {_x[_out[0]]:.1f}-{_x[_out[-1]]:.1f} of {a.nframes - 1}; under the ground at the stored keys {_uk[0]}-{_uk[-1]} '
                           f'({"one unbroken stretch" if _one else "SEVERAL STRETCHES"}), deepest {_ss.min():.2f} (limit {PLANT_DEPTH}); fist closed on keys 0-3 and on the last key: {ok_g}')
        recall_rows.append(f'           planted blade, keys {_k0}-{_k1}: moves at most {_step:.2f} u from one key to the next (limit {PLANT_STEP}); informational - deviation from that between two keys '
                           f'(engine interpolation under the moving forearm): mean {_dev.mean():.2f}, 90% {np.percentile(_dev, 90):.2f}, max {_dev.max():.2f} u at key {_k0 + int(_dev.argmax())}; {int((_dev > 2).sum())} of {len(_dev)} key intervals above 2 u')
        # stowed on her back (from 6 keys after they leave the fist to 8 keys before her hand is back on them): the blade must not run
        # through her body or her tail. A scissors vertex counts as inside when it lies behind the surface at its nearest body / tail
        # vertex (within 14 units); gated at STOW_MAX of the scissors' vertices per stored key.
        from scipy.spatial import cKDTree
        _hd = []; _pn = []
        for f in range(a.nframes):
            _W = L.pose_world(sk, a, f); _p, _n = L.deform(m, sk, _W); _pn.append((_p, _n)); _hd.append(float(np.linalg.norm(_p[v_sw] - _W[jr][:3, 3], axis=1).min()))
        _s0 = int(np.ceil(_x[_out[0]])); _gr = next(f for f in range(_s0 + 10, a.nframes) if _hd[f] < 10.0); _w0, _w1 = _s0 + 6, _gr - 8; _nb = []; _nt = []
        for f in range(_w0, _w1 + 1):
            _p, _n = _pn[f]
            for _grp, _acc in ((v_body, _nb), (v_tail, _nt)):
                _d, _i = cKDTree(_p[_grp]).query(_p[v_sw]); _acc.append(int(((((_p[v_sw] - _p[_grp][_i]) * _n[_grp][_i]).sum(1) < 0) & (_d < 14.0)).sum()))
        _lim = int(STOW_MAX * len(v_sw)); ok_stow = _w1 > _w0 + 30 and max(_nb) <= _lim and max(_nt) <= _lim; ok_m = ok_m and ok_stow
        recall_rows.append(f'           stowed on her back, keys {_w0}-{_w1}: scissors vertices inside the body at most {max(_nb)} (key {_w0 + int(np.argmax(_nb))}), inside the tail at most {max(_nt)} '
                           f'(key {_w0 + int(np.argmax(_nt))}), of {len(v_sw)} (limit {_lim} each); keys with any inside: body {int((np.array(_nb) > 0).sum())}, tail {int((np.array(_nt) > 0).sum())} of {_w1 - _w0 + 1}')
    by = min(ysub(v_body, L.pose_world(sk, a, f) @ sk.ibind) for f in range(a.nframes))
    air = nm(k) in AIRBORNE; ok_h = air or hov <= 5.0
    motion_ok &= ok_m; grip_ok &= ok_g; scale_ok &= ok_s; hover_ok &= ok_h
    _ev = get(p.data, 'mEventDataMap'); _K = int(round(a.fps / 29.0))
    for _ek, _e in (_ev.data if _ev is not None else []):
        _d = {tables['binfields'].get(f.hash, f.hash): f.data for f in _e.data if not isinstance(f.data, (list, tuple))}
        if _d.get('mStartFrame') is not None: evrows.append((nm(k), _K, a.fps, tables['bintypes'].get(_e.hash_type, _e.hash_type), float(_d['mStartFrame']), (a.nframes - 1) / a.fps))
    mrows.append(f'   {"ok " if ok_m and ok_g and ok_s and ok_h else "BAD"} {nm(k):20s} lowest point over the interpolated motion: soles {fy:6.2f}  scissors {sy:6.2f}  tail {ty:6.2f} | grip gap max {gap:4.2f} | scale deviation {smax:.4f} | highest the lower sole gets {hov:6.2f}{" (airborne clip)" if air else ""} | lowest body vertex {by:6.2f}')
for r in rows: log(r)
log(f'motion checks (every stored frame + {SUBS - 1} interpolated poses between each pair; {len(v_feet)} sole, {len(v_sw)} scissors, {len(v_tail)} tail vertices, {len(grip)} grip contact points):')
for r in mrows: log(r)
for r in recall_rows: log(r)
check(allok, 'every clip file exists, is r3d2anmd v4, finite, unit quaternions, all tracks map to skeleton joints, no joint untracked (tail1..16 keyed), source key counts/rates checked individually (29 fps; wall vaults 30 fps)')
check(motion_ok and recall_rows, 'scissors and tail never below the ground, soles never lower than -2.0 units, also between stored frames (recall: scissors in the ground only while planted, where they stand still from key to key, and clear of body and tail while stowed on her back - see above)')
check(grip_ok, 'right fist stays closed on the scissors handle (contact points move <= 1.0 unit) in every clip, also between stored frames (recall: on its first four keys and its last one; in between the scissors leave the hand by design)')
check(scale_ok, 'no joint is scaled in any clip (no limb stretch)')
check(hover_ok, 'standing clips never float: outside the eleven airborne clips (' + ', '.join(AIRBORNE) + ') the lower sole is never more than 5 units above the ground, also between stored frames')
log('   ("lowest body vertex" is informational, stored frames only: body, limbs and clothes other than soles, scissors and tail - see the landing rolls and the death pose)')
log('events with a start frame (informational). Every clip is stored at the base key rate, so a start frame means the same key as in the base mod')
log('  (the melee Q, Spell1, was re-keyed so that its cut lands on key 7, just before its hit effect on key 8):')
for _n, _K, _fps, _cls, _sf, _dur in evrows:
    log(f'   {_n:20s} x{_K} ({_fps:5.1f} fps, {_dur:.3f} s)  {_cls:18s} mStartFrame {_sf:5.1f} -> {_sf / 29.0:.3f} s at the base rate' + (f', {_sf / _fps:.3f} s if counted in stored frames' if _K > 1 else ''))
# ---- looping clips (mFlags & 2) must end on their first pose
seam = []
for k, p in clips.data:
    if tables['bintypes'].get(p.hash_type) != 'AtomicClipData': continue
    _fl = get(p.data, 'mFlags')
    if _fl is None or not int(_fl.data) & 2: continue
    a = L.load_anm(FIN + '/w/' + get(get(p.data, 'mAnimationResourceData').data, 'mAnimationFilePath').data.lower())
    sr = float(np.degrees(2 * np.arccos(np.clip(np.abs((a.R[0] * a.R[-1]).sum(-1)), 0, 1))).max()); st = float(np.abs(a.T[0] - a.T[-1]).max())
    seam.append((nm(k), sr, st))
log('looping clips, last stored pose vs first: ' + ', '.join(f'{n} {sr:.2f} deg / {st:.3f} u' for n, sr, st in seam))
check(seam and all(sr <= 0.1 and st <= 0.01 for n, sr, st in seam), 'looping clips end exactly on their first pose (seam <= 0.1 deg, 0.01 units)')
# ---- blends between clips: every TimeBlendData pair of the graph with a blend time, evaluated the way the engine cross-fades
#      local transforms. The engine's exact scheduling is not known, so both conventions are checked (see below).
clipanm = {}; cliploop = {}
for k, p in clips.data:
    if tables['bintypes'].get(p.hash_type) == 'AtomicClipData':
        _fl = get(p.data, 'mFlags'); cliploop[int(k, 16) if isinstance(k, str) else int(k)] = bool(int(_fl.data) & 2) if _fl is not None else False      # flag 2 = looping clip
        clipanm[int(k, 16) if isinstance(k, str) else int(k)] = L.load_anm(FIN + '/w/' + get(get(p.data, 'mAnimationResourceData').data, 'mAnimationFilePath').data.lower())
def local_at(a, t):
    x = min(max(t * a.fps, 0.0), a.nframes - 1.0); f0 = min(int(x), a.nframes - 2); u = x - f0
    return (1 - u) * a.T[f0] + u * a.T[f0 + 1], L.slerp(a.R[f0], a.R[f0 + 1], np.full(a.R.shape[1], u)), (1 - u) * a.S[f0] + u * a.S[f0 + 1]
def measure(hashes, T, R, S_):
    b = L.Anim(); b.hashes = hashes; b.sig = 'r3d2anmd'; b.has = np.ones((len(hashes), 3), bool); b.T = T[None]; b.R = R[None]; b.S = S_[None]
    M = L.pose_world(sk, b, 0) @ sk.ibind
    return ysub(v_feet, M), ysub(v_sw, M), ysub(v_tail, M), float(np.linalg.norm((grip @ M[sk.index['r_hand']].T - grip @ M[sk.index['weapon']].T)[:, :3], axis=1).max())
bt = get(g.data, 'mBlendDataTable'); RECALL = 0x9260a8e8; LOCO = ('Run_Normal', 'Run_Homeguard'); MIDMOVE = ('Dance', 'Death', 'Spell3', 'Spell3_Long', 'Spell3_Medium')
SOLE_FROZEN, SOLE_PLAYING = -2.0, -3.0
time_ok = True; gated = []; loco = []; midmove = []; midcut = []; loco_hard = True; blend_ok = True; npairs = 0; cuts = []; cut_ok = True
def skinned(a, f):
    M = L.pose_world(sk, a, f) @ sk.ibind; P = np.c_[m.pos, np.ones(len(m.pos))]; V = np.zeros((len(P), 3))
    for k in range(infl.shape[1]): V += m.w[:, k:k + 1] * np.einsum('nij,nj->ni', M[infl[:, k], :3, :], P)
    return V
for key, p in (bt.data if bt is not None else []):
    if tables['bintypes'].get(p.hash_type) != 'TimeBlendData': continue
    _tm = get(p.data, 'mTime'); tm = float(_tm.data) if _tm is not None else 0.0; explicit0 = _tm is not None and float(_tm.data) == 0.0
    time_ok &= _tm is not None and np.isfinite(tm) and tm >= 0
    key = int(key); fr, to = key >> 32, key & 0xFFFFFFFF      # (from << 32) | to: e.g. every clip -> Death is 0 s
    if fr not in clipanm or to not in clipanm or fr == to: continue
    aO, aI = clipanm[fr], clipanm[to]; nO, nI = nm(f'{fr:08x}'), nm(f'{to:08x}')
    if tm <= 0:
        # a cut: only the two hand-offs that are chained by design must be continuous (X -> Death may cut from any pose)
        if nI == 'Death': pass
        elif to == RECALL and nO in MIDMOVE:
            if explicit0: midcut.append(nO)
        else:
            d = float(np.linalg.norm(skinned(aO, aO.nframes - 1) - skinned(aI, 0), axis=1).max()); cuts.append(f'{nO} -> {nI}: mesh moves {d:.2f} units'); cut_ok &= d <= 0.5
        continue
    mapI = {int(h): i for i, h in enumerate(aO.hashes)}; perm = [mapI[int(h)] for h in aI.hashes]
    loop = cliploop[fr]; durO = (aO.nframes - 1) / aO.fps
    exits = list(range(aO.nframes - 1)) if loop else [aO.nframes - 1]
    res = []
    for adv in (False, True):
        fy = sy = ty = 1e9; gp = 0.0
        for fo in exits:
            for st in np.linspace(0, tm, 33):
                # outgoing clip: (a) frozen at its exit pose, (b) still playing - a loop continues from the exit phase, any
                # other clip overlaps its last mTime seconds with the incoming one
                tO = fo / aO.fps if not adv else ((fo / aO.fps + st) % durO if loop else max(0.0, durO - tm + st))
                To, Ro, So = (x[perm] for x in local_at(aO, tO))
                w = st / tm; Ti, Ri, Si = local_at(aI, st)
                r = measure(aI.hashes, (1 - w) * To + w * Ti, L.slerp(Ro, Ri, np.full(len(Ro), w)), (1 - w) * So + w * Si)
                fy = min(fy, r[0]); sy = min(sy, r[1]); ty = min(ty, r[2]); gp = max(gp, r[3])
        res.append((fy, sy, ty, gp))
    (fa, sa, ta, ga), (fb, sb, tb, gb) = res
    ok = min(sa, sb) >= 0 and min(ta, tb) >= 0 and max(ga, gb) <= 1.0 and fa >= SOLE_FROZEN and fb >= SOLE_PLAYING; npairs += 1
    row = (ok, f'{nO:20s} -> {nI:20s} {tm:.2f}s  frozen: soles {fa:6.2f} scissors {sa:6.2f} tail {ta:6.2f} grip {ga:4.2f} | playing: soles {fb:6.2f} scissors {sb:6.2f} tail {tb:6.2f} grip {gb:4.2f}')
    if nO in LOCO: loco.append(row); loco_hard &= min(sa, sb) >= 0 and min(ta, tb) >= 0 and max(ga, gb) <= 1.0
    elif to == RECALL and nO in MIDMOVE: midmove.append(row)
    else: gated.append(row); blend_ok &= ok
check(time_ok, 'every TimeBlendData entry carries an explicit, finite, non-negative mTime')
log(f'cuts (0 s) between chained clips: ' + '; '.join(cuts))
check(cut_ok and cuts, 'clips joined by a 0 s cut continue from the exact same pose (mesh moves <= 0.5 units)')
log(f'cross-fades: {npairs} TimeBlendData pairs with a blend time. The incoming clip advances from its start; the outgoing clip is evaluated')
log('  both frozen at its exit pose and still playing (the engine\'s scheduling is not known). Looping clips exit at every key.')
log('  33 samples per blend; local T/S lerp, shortest-arc R slerp. A finite sampling of two modelled schedules, not the engine itself.')
log(f' (1) {len(gated)} pairs with a defined exit pose - idle -> dash / attack / spell / emote, dash -> landing, landing -> run or idle, and every\n     standing or finished clip -> recall:')
for ok, r in gated: log('   ' + ('ok  ' if ok else 'BAD ') + r)
check(blend_ok and gated, f'these cross-fades keep the scissors and tail above the ground, the fist on the handle and the soles >= {SOLE_FROZEN} (outgoing frozen) / {SOLE_PLAYING} (outgoing playing)')
log(f' (2) {len(loco)} pairs that leave the run loop mid-stride: scissors, tail and grip are gated, the soles are informational. Cross-fading a')
log('     stride into a standing pose turns the trailing leg towards vertical while its foot is still pitched for push-off, so the toes')
log('     pass a few units below the ground for a few hundredths of a second at some stride phases. The dip depends on the phase the')
log('     run is left at; the action and recall clips already open on the standing idle, and this build does not attempt to remove it:')
for ok, r in loco: log('   ' + ('ok  ' if ok else 'dip ') + r)
check(loco_hard and loco, 'leaving the run loop never puts the scissors or tail below the ground or opens the fist')
log(f' (3) recall from the clips that end or loop in a crouch, in the air or lying down ({", ".join(MIDMOVE)}): the base graph cross-fades')
log('     them into the recall like every other clip, which passed the scissors / feet through the ground (dance: scissors -15, soles -9).')
log('     They now cut to the recall (0 s), the way every clip cuts to the death clip. Entries with an explicit mTime of exactly 0: ' + ', '.join(sorted(midcut)))
check(sorted(midcut) == sorted(MIDMOVE) and not midmove, 'those five clips cut to the recall instead of cross-fading through the ground; no cross-fade is left ungated except the run soles in (2)')
# ---- transition clips: every TransitionClipBlendData names existing clips
trs = []; tr_ok = True
for key, p in (bt.data if bt is not None else []):
    if tables['bintypes'].get(p.hash_type) != 'TransitionClipBlendData': continue
    key = int(key); fr, to = key >> 32, key & 0xFFFFFFFF; c = int(get(p.data, 'mClipName').data, 16)
    tr_ok &= fr in clipanm and to in clipanm and c in clipanm; trs.append((nm(f'{fr:08x}'), nm(f'{to:08x}'), nm(f'{c:08x}')))
_by = {}
for a_, b_, c_ in trs: _by.setdefault(c_, []).append(f'{a_}>{b_}')
log('transition clips: ' + '; '.join(f'{c_}: {len(v)} pairs' + (' (' + ', '.join(sorted(v)) + ')' if len(v) <= 6 else '') for c_, v in sorted(_by.items())))
check(tr_ok and trs, f'all {len(trs)} transition-clip entries of the blend table name clips that exist')
VAULT = {('Spell3_Long', f'Idle{i}', '0x86b44e3c') for i in range(1, 5)} | {('Spell3_Medium', f'Idle{i}', 'Spell3_Medium_toRun') for i in range(1, 5)} | {('Spell3_Long', 'Run_Normal', '0x86b44e3c'), ('Spell3_Medium', 'Run_Normal', 'Spell3_Medium_toRun')}
check(VAULT <= set(trs), 'the long and the medium vault go through their landing clip both when she runs on and when she stops (Idle1-4)')
# ---- (4) transitions the blend table does not list. The table's explicit 0 s entries suggest that unlisted pairs get an engine
#      default cross-fade of unknown length; simulate them at 0.1 s and 0.2 s.
listed = {(int(key) >> 32, int(key) & 0xFFFFFFFF) for key, p in (bt.data if bt is not None else [])}
byname = {nm(f'{h:08x}'): h for h in clipanm}
def xfade(fr, to, tm, every=1, n=17):
    aO, aI = clipanm[fr], clipanm[to]; mapI = {int(h): i for i, h in enumerate(aO.hashes)}; perm = [mapI[int(h)] for h in aI.hashes]
    loop = cliploop[fr]; durO = (aO.nframes - 1) / aO.fps; exits = list(range(0, aO.nframes - 1, every)) if loop else [aO.nframes - 1]; out = []
    for adv in (False, True):
        fy = sy = ty = 1e9; gp = 0.0
        for fo in exits:
            for st in np.linspace(0, tm, n):
                tO = fo / aO.fps if not adv else ((fo / aO.fps + st) % durO if loop else max(0.0, durO - tm + st))
                To, Ro, So = (x[perm] for x in local_at(aO, tO)); w = st / tm; Ti, Ri, Si = local_at(aI, st)
                r = measure(aI.hashes, (1 - w) * To + w * Ti, L.slerp(Ro, Ri, np.full(len(Ro), w)), (1 - w) * So + w * Si)
                fy = min(fy, r[0]); sy = min(sy, r[1]); ty = min(ty, r[2]); gp = max(gp, r[3])
        out.append((fy, sy, ty, gp))
    return out
STARTS = ('Attack1', 'Attack2', 'Crit', 'Spell1', 'Spell1_Leap', 'Spell2', 'Spell4', 'Dance', 'Taunt', 'Joke', 'Laugh', 'Channel_Wndup')
def urow(a_, b_, every=1):
    fr, to = byname[a_], byname[b_]
    if (fr, to) in listed or fr == to: return None
    r = {tm: xfade(fr, to, tm, every) for tm in (0.1, 0.2)}
    sof = min(x[0][0] for x in r.values()); sop = min(x[1][0] for x in r.values())
    sw = min(v[1] for x in r.values() for v in x); tl = min(v[2] for x in r.values() for v in x); gp = max(v[3] for x in r.values() for v in x)
    txt = f'{a_:20s} -> {b_:14s} ' + ' | '.join(f'{tm:.1f} s: soles {x[0][0]:6.2f} / {x[1][0]:6.2f} scissors {min(x[0][1], x[1][1]):6.2f} tail {min(x[0][2], x[1][2]):6.2f}' for tm, x in r.items())
    return txt, sof, sop, sw, tl, gp
ONESHOT = [n for n in sorted(byname) if not cliploop[byname[n]] and n != 'Death']
u_idle = [x for x in (urow('Idle1', n, 2) for n in STARTS + ('Run_Normal',)) if x]; u_run = [x for x in (urow('Run_Normal', n) for n in STARTS + ('Idle1',)) if x]
u_end = [x for x in (urow(n, 'Idle1') for n in ONESHOT) if x]; u_endr = [x for x in (urow(n, 'Run_Normal') for n in ONESHOT) if x]
hard = lambda x: x[3] >= 0 and x[4] >= 0 and x[5] <= 1.0
full = lambda x: hard(x) and x[1] >= SOLE_FROZEN and x[2] >= SOLE_PLAYING
log(' (4) transitions the blend table does not list. Its explicit 0 s entries suggest that unlisted pairs get a default cross-fade of')
log('     unknown length, so each one is simulated at 0.1 s and at 0.2 s (soles: outgoing frozen / playing; 17 samples per blend).')
log('  (a) standing idle (left at every 2nd key) -> run, and any action start the table still leaves unlisted:')
for x in u_idle: log('   ' + ('ok  ' if full(x) else 'BAD ') + x[0])
log('  (b) the natural end of each one-shot clip -> standing idle:')
for x in u_end: log('   ' + ('ok  ' if full(x) else 'BAD ') + x[0])
check(u_idle and u_end and all(full(x) for x in u_idle + u_end), f'idle -> run and every one-shot clip that plays out -> idle keep the scissors and tail above the ground and the soles >= {SOLE_FROZEN} / {SOLE_PLAYING}, at 0.1 s and at 0.2 s')
log('  (c) informational, not gated: leaving the run loop (at every key) for the idle, an emote or the channel wind-up, and the exact')
log('      end of a one-shot clip -> run. Leaving the run has the mid-stride toe dip of (2). The emotes and the channel wind-up are')
log('      started standing; out of a run the channel wind-up swings the scissors about 2 units through the ground for an instant.')
log('      A one-shot clip that is still easing onto the idle stance while the run fades in can dip a sole by 2-5 units:')
for x in u_run + u_endr: log('   ' + ('ok  ' if full(x) else 'dip ') + x[0])
log('      Not simulated: interrupting a clip in mid-motion (attack cancels, crowd control). That blends from an arbitrary pose, on any skin.')
check(not dangling, f'sequencer / condition clips reference existing clips {dangling}')
need = ['Idle1', 'Run', 'Attack1', 'Attack2', 'Crit', 'Spell1', 'Spell2', 'Spell3', 'Spell4', 'Death', 'Recall', 'Dance', 'Taunt', 'Joke', 'Laugh', 'Channel', 'Channel_Wndup']
check(all(fh(n) in keys for n in need), 'standard clip names present (Idle1, Run, Attack1/2, Crit, Spell1-4, Death, Recall, emotes, Channel)')
# ---- resource resolver -> vfx entries
rr = ent[get(skin.data, 'mResourceResolver').data]; rm = get(rr.data, 'resourceMap'); badv = [nm(k) for k, v in rm.data if v not in ent]
check(not badv, f'all {len(rm.data)} resource-resolver links resolve to entries in this bin {badv}')
# ---- particle events -> resolver keys
rk = {k for k, v in rm.data}; bade = []
for k, p in clips.data:
    for f in walk(p):
        if (tables['binfields'].get(f.hash, '') if f.hash else '') == 'mEffectKey' and f.data not in rk: bade.append((nm(k), nm(f.data)))
check(not bade, f'particle events use effect keys known to the resolver {bade}')
log('\nRESULT:', 'ALL CHECKS PASSED' if not fails else f'{len(fails)} FAILED: {fails}')
