"""build_pack.py S OUT.fantome -> edits the skin bin, assembles a reference-checked WAD from the work-in-progress mod plus the
rebuilt rig / animations, verifies it by reading it back and packages the .fantome."""
import sys, os, io, json, copy, struct, zipfile, glob
S = sys.argv[1]; OUT = sys.argv[2]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binutil import *
import lolio as L
import pyzstd
from xxhash import xxh3_64
from pyritofile import WAD
from pyritofile.wad import WADCompressionType
B = 'assets/ice/wild/bum/characters/talon/skins/'
SRC_FANTOME = S + '/x/Ellen_Joe_Talon_V1_27B'; SRC_WAD = SRC_FANTOME + '/WAD/talon.wad.client'; SRC_TREE = S + '/w/ellen_talon/'
BUILD = S + '/' + (sys.argv[3] if len(sys.argv) > 3 else 'build')      # folder holding the freshly built rig + animations
NEW = BUILD + '/wad/'
BIN_PATH = 'data/characters/talon/skins/skin0.bin'
rep = open(BUILD + '/pack_report.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); rep.write(s + '\n')

# ---------- 1. source WAD ----------
wad = WAD(); wad.read(SRC_WAD); src = {}
with wad.stream(SRC_WAD, 'rb') as bs:
    for c in wad.chunks:
        bs.seek(c.offset); raw = bs.read(c.compressed_size); c.read_data(bs)
        src[c.hash] = dict(raw=raw, ctype=c.compression_type, usize=c.decompressed_size, checksum=c.checksum, data=c.data, sub=c.subchunk_count)
names = {}
for dp, dn, fn in os.walk(SRC_TREE):
    for f in fn:
        relp = os.path.relpath(os.path.join(dp, f), SRC_TREE).replace(os.sep, '/')
        if relp.startswith('_unknown/'): names[f.split('.')[0]] = relp
        else: names[ph(relp)] = relp
assert set(names) == set(src), (len(names), len(src))
log(f'source WAD: {len(src)} chunks, version {wad.version}')
tables = load_tables(S + '/hashes')
game = set()
with open(S + '/hashes/hashes.game.txt', 'r', encoding='utf-8', errors='replace') as f:
    for line in f: game.add(line[:16])

# ---------- 2. skin bin ----------
b = BIN(); b.read('', raw=src[ph(BIN_PATH)]['data'])
assert b.write('', raw=True) == src[ph(BIN_PATH)]['data'], 'bin does not round-trip'
ent = {e.hash: e for e in b.entries}
skl = L.load_skl(NEW + B + 'skin12/talon_skin12.skl'); skn = L.load_skn(NEW + B + 'skin12/ellen/ellen_talon.skn')
bones = {fh(n): n for n in skl.names}; subm = {fh(n): n for n, *_ in skn.subs}
nm = lambda h: tables['binhashes'].get(h) or tables['binentries'].get(h) or '0x' + h
skin = ent[fh('Characters/Talon/Skins/Skin0')]; smp = get(skin.data, 'skinMeshProperties')
log('\n== bin edits')
# 2a. tail physics: same engine modifier (and values) the Samira Ellen mod uses for this tail, on the new tail chain
sb = BIN(); sb.read(glob.glob(S + '/w/samira/data/characters/samira/skins/*.bin')[0])
ssmp = next(get(e.data, 'skinMeshProperties') for e in sb.entries if get(e.data, 'skinMeshProperties') is not None)
rpm = copy.deepcopy(get(ssmp.data, 'rigPoseModifierData')); assert rpm is not None and len(rpm.data) == 1
mod = rpm.data[0]; assert tables['bintypes'].get(mod.hash_type) == 'ConformToPathRigPoseModifierData'
get(mod.data, 'mStartingJointName').data = fh('tail1'); get(mod.data, 'mEndingJointName').data = fh('tail16')
assert fh('tail1') in bones and fh('tail16') in bones and get(smp.data, 'rigPoseModifierData') is None
TAILMOD = os.environ.get('LOL_TAILMOD') == '1'      # V2.1: the tail is keyed in every clip; the engine modifier is only added on request (variant B)
if TAILMOD: i = smp.data.index(get(smp.data, 'materialOverride')); smp.data.insert(i + 1, rpm)
log('  + skinMeshProperties.rigPoseModifierData' if TAILMOD else '  (not added: tail is keyed in every clip) rigPoseModifierData = ConformToPathRigPoseModifierData tail1..tail16', {tables['binfields'].get(f.hash): f.data for f in mod.data})
# 2b. scissors texture: .dds -> .tex (a .dds needs 2x_/4x_ companions at lower texture quality; .tex does not)
DDS = B + 'skin12/ellen/ellen_scissors_tx_cm.dds'; TEXP = B + 'skin12/ellen/ellen_scissors_tx_cm.tex'
dds = src[ph(DDS)]['data']; w_, h_ = struct.unpack_from('<II', dds, 12)[::-1]
assert dds[:4] == b'DDS ' and dds[84:88] == b'DXT5' and len(dds) == 128 + w_ * h_ and struct.unpack_from('<I', dds, 28)[0] <= 1
scissors_tex = b'TEX\0' + struct.pack('<HHBBBB', w_, h_, 1, 12, 0, 0) + dds[128:]
n = 0
for e in b.entries:
    for f in walk(e.data):
        if f.type == BINType.String and isinstance(f.data, str) and f.data.lower() == DDS:
            f.data = f.data[:-4] + '.tex'; n += 1
assert n >= 1; log(f'  ~ scissors texture reference .dds -> .tex ({w_}x{h_} DXT5, same pixels; {n} reference(s))')
# 2c. E wall-vault clips -> the repaired clips; 2d. drop events that target bones / submeshes this model does not have
g = ent[fh('Characters/Talon/Animations/Skin12')]; clips = get(g.data, 'mClipDataMap')
assert get(get(skin.data, 'skinAnimationProperties').data, 'animationGraphData').data == g.hash
REPATH = {'Spell3_Long': 'ASSETS/ice/wild/bum/Characters/Talon/Skins/Skin12/Animations/talon_spell3_long.anm',
          'Spell3_Medium': 'ASSETS/ice/wild/bum/Characters/Talon/Skins/Skin12/Animations/talon_spell3_medium.anm'}
done = set()
for k, p in clips.data:
    cname = nm(k)
    if cname in REPATH:
        f = get(get(p.data, 'mAnimationResourceData').data, 'mAnimationFilePath'); log(f'  ~ clip {cname}: {f.data} -> {REPATH[cname]}'); f.data = REPATH[cname]; done.add(cname)
    ev = get(p.data, 'mEventDataMap')
    if ev is None: continue
    keep = []
    for ek, e in ev.data:
        cls = tables['bintypes'].get(e.hash_type, e.hash_type); miss = []; refs = 0
        for f in walk(e):
            fn = tables['binfields'].get(f.hash, '') if f.hash else ''
            for ty, v in values(f):
                if ty != BINType.Hash: continue
                if fn == 'mBoneName': refs += 1; miss += [nm(v)] if v not in bones else []
                if fn in ('mShowSubmeshList', 'mHideSubmeshList'): refs += 1; miss += [nm(v)] if v not in subm else []
        if miss:
            assert cls in ('ParticleEventData', 'SubmeshVisibilityEventData') and len(miss) == refs, (cname, cls, miss)
            log(f'  - clip {cname}: removed {cls} {nm(ek)} (targets missing {miss})')
        else: keep.append((ek, e))
    ev.data = keep
assert done == set(REPATH)
# 2e. leftover material definitions of the base mod that nothing links to
linked = set()
for e in b.entries:
    for f in walk(e.data):
        for ty, v in values(f):
            if ty in (BINType.Link, BINType.Hash): linked.add(v)
for h in ('03dd80ba', 'f310b137', 'f210afa4'):
    e = ent[h]; assert tables['bintypes'].get(e.type) == 'StaticMaterialDef' and h not in linked
    log(f"  - removed unlinked StaticMaterialDef {get(e.data, 'name').data} (base-mod leftover)"); b.entries.remove(e)
# 2f. second animation graph of the base mod: not linked by this skin, points at a missing run clip, and carries the hash of
#     the game's own base-Talon graph (so it could shadow it for other Talon skins)
h = fh('Characters/Talon/Animations/Skin0'); e = ent[h]; assert h not in linked and tables['bintypes'].get(e.type) == 'AnimationGraphData'
b.entries.remove(e); log('  - removed unlinked AnimationGraphData Characters/Talon/Animations/Skin0 (base-mod leftover, Wild Rift base-Talon clips)')
assert get(smp.data, 'Material').data in {e.hash for e in b.entries}
# 2g. recall from a clip that ends or loops in a crouch, in the air or lying down: the graph cross-fades every clip into the
#     recall over 0.1 s. The recall opens on the standing idle (scissors hanging just above the ground), and fading a low or
#     mid-move pose into that passes the scissors / feet through the ground (dance -> recall: scissors -15, soles -9).
#     No key in either clip can prevent that for an arbitrary exit pose, so these five cut to the recall instead, the same
#     way every clip already cuts to the death clip.
RECALL = 0x9260a8e8; cutfrom = {int(fh(n), 16): n for n in ('Dance', 'Death', 'Spell3', 'Spell3_Long', 'Spell3_Medium')}; done = []
for key, p in get(g.data, 'mBlendDataTable').data:
    k = int(key)
    if k & 0xFFFFFFFF == RECALL and (k >> 32) in cutfrom:
        assert tables['bintypes'].get(p.hash_type) == 'TimeBlendData'; t = get(p.data, 'mTime'); assert abs(t.data - 0.1) < 1e-6
        t.data = 0.0; done.append(cutfrom[k >> 32])
assert sorted(done) == sorted(cutfrom.values()), done
log('  - blend into the recall set from 0.1 s to 0 s (cut) for: ' + ', '.join(sorted(done)))
# 2h. transitions the base graph leaves to the engine's default blend (its length is not known). The base graph lists
#     idle / run -> dash at 0.1 s, and the clips are built to be entered over that time (they open on the standing idle for
#     0.07-0.1 s). Give the other action clips the same explicit 0.1 s entry, and let the long / medium vault use their landing
#     clips when she stops after the vault too (the base graph only does so when she runs on; standing still, it blended the
#     end of the vault - still in the air, or deep in the landing crouch - straight into the upright idle).
import copy
_bt = get(g.data, 'mBlendDataTable'); _have = {int(k) for k, p in _bt.data}; H = lambda n: int(n, 16) if n.startswith('0x') else int(fh(n), 16)
_ckeys = {int(k, 16) for k, p in get(g.data, 'mClipDataMap').data}
_t01 = next(p for k, p in _bt.data if tables['bintypes'].get(p.hash_type) == 'TimeBlendData' and abs(get(p.data, 'mTime').data - 0.1) < 1e-6)
_ttr = next(p for k, p in _bt.data if tables['bintypes'].get(p.hash_type) == 'TransitionClipBlendData')
IDLES = ('Idle1', 'Idle2', 'Idle3', 'Idle4'); RUNS = ('Run_Normal', 'Run_Homeguard')
ACTIONS = ('Attack1', 'Attack2', 'Crit', 'Spell1', 'Spell1_Leap', 'Spell2', 'Spell4'); STANDING = ('Channel_Wndup', 'Dance', 'Taunt', 'Joke', 'Laugh')
def _add(fr, to, tmpl, clip=None):
    assert H(fr) in _ckeys and H(to) in _ckeys, (fr, to); k = (H(fr) << 32) | H(to)
    if k in _have: return 0
    p = copy.deepcopy(tmpl)
    if clip is not None: assert H(clip) in _ckeys; get(p.data, 'mClipName').data = f'{H(clip):08x}'
    _bt.data.append((k, p)); _have.add(k); return 1
n1 = sum(_add(i, a, _t01) for i in IDLES for a in ACTIONS + STANDING) + sum(_add(r, a, _t01) for r in RUNS for a in ACTIONS)
n2 = sum(_add(v, i, _ttr, c) for v, c in (('Spell3_Long', '0x86b44e3c'), ('Spell3_Medium', 'Spell3_Medium_toRun')) for i in IDLES)
n3 = sum(_add(c, i, _t01) for c in ('0x86b44e3c', 'Spell3_Medium_toRun') for i in IDLES)
assert (n1, n2, n3) == (62, 8, 8), (n1, n2, n3)
log(f'  - blend table: {n1} explicit 0.1 s entries idle / run -> ' + ', '.join(ACTIONS) + ' and idle -> ' + ', '.join(STANDING) + ' (were left to the engine default)')
log(f'  - blend table: {n2} transition entries Spell3_Long / Spell3_Medium -> Idle1-4 through their landing clips, and {n3} entries landing clip -> Idle1-4 at 0.1 s')
# Source recall effects are definitions plus their texture/mesh dependency closure, not just animation.
from recall_vfx import source_vfx
_fxevents, _fxentries, _fxmap, _fxassets = source_vfx(S)
_recall = dict(clips.data)['9260a8e8']; assert _recall.hash_type == fh('AtomicClipData')      # Recall is a sequencer; events belong on its motion clip
_ev = get(_recall.data, 'mEventDataMap')
if _ev is None:
    _ev = copy.deepcopy(next(get(p.data,'mEventDataMap') for _,p in clips.data if get(p.data,'mEventDataMap') is not None)); _ev.data=[]; _recall.data.append(_ev)
_ev.data = [(k,e) for k,e in _ev.data if e.hash_type != fh('ParticleEventData')] + _fxevents
_rr = ent[get(skin.data,'mResourceResolver').data]; _rm=get(_rr.data,'resourceMap')
assert not set(k for k,_ in _fxmap) & set(k for k,_ in _rm.data)
_rm.data.extend(_fxmap)
assert not {e.hash for e in _fxentries} & {e.hash for e in b.entries}
b.entries.extend(_fxentries)
log(f'  + recall: {len(_fxevents)} source particle events, {len(_fxentries)} namespaced systems, {len(_fxassets)} bundled textures/meshes; 30 fps event times mapped to 29 fps')
new_bin = b.write('', raw=True)
b2 = BIN(); b2.read('', raw=new_bin); assert b2.write('', raw=True) == new_bin and len(b2.entries) == len(b.entries)
log(f'  bin: {len(src[ph(BIN_PATH)]["data"])} -> {len(new_bin)} bytes, {len(b2.entries)} entries, re-read OK')

# ---------- 3. audit the final bin ----------
log('\n== reference audit (final bin)')
refs = {}
for s_ in strings(b2):
    l = s_.lower()
    if '/' in l and '.' in l.rsplit('/', 1)[-1] and (l.startswith('assets/') or l.startswith('data/')): refs[ph(l)] = l
content = {}                                                  # hash -> (name, data bytes or None=copy from source)
for h, n_ in names.items(): content[h] = (n_, None)
def put(path, data): content[ph(path)] = (path, data)
put(BIN_PATH, new_bin); put(TEXP, scissors_tex)
for _target in _fxassets: put(_target, open(S + '/dl/recall_assets/' + _target, 'rb').read())
put(B + 'skin12/talon_skin12.skl', open(NEW + B + 'skin12/talon_skin12.skl', 'rb').read())
put(B + 'skin12/ellen/ellen_talon.skn', open(NEW + B + 'skin12/ellen/ellen_talon.skn', 'rb').read())
new_anims = sorted(glob.glob(NEW + B + 'skin12/animations/*.anm')); assert len(new_anims) == 24
for p_ in new_anims: put(B + 'skin12/animations/' + os.path.basename(p_), open(p_, 'rb').read())
dropped = []
for h in list(content):
    n_, d = content[h]
    if h == ph(BIN_PATH): continue
    if h not in refs: dropped.append((n_, len(src[h]['data']) if d is None else len(d), (src[h]['data'] if d is None else d)[:10])); del content[h]
# inherited: both source mods list this VO bank but ship only the events .bnk + audio .wpk (voice data lives in the .wpk)
KNOWN_ABSENT = {'assets/ice/wild/bum/sounds/wwise2016/vo/en_us/characters/talon/skins/base/talon_base_vo_audio.bnk'}
missing = [(l) for h, l in refs.items() if h not in content and h not in game and l not in KNOWN_ABSENT]
for l in KNOWN_ABSENT & set(refs.values()): log('  listed but absent, exactly as in both source mods (left untouched):', l)
ingame = [(l) for h, l in refs.items() if h not in content and h in game]
log(f'  asset paths referenced by the bin: {len(refs)}; shipped in WAD: {sum(1 for h in refs if h in content)}; resolved by the game client: {len(ingame)}; MISSING: {len(missing)}')
for l in ingame: log('    game file:', l)
for l in missing: log('    MISSING:', l)
assert not missing
for p_ in new_anims: assert ph(B + 'skin12/animations/' + os.path.basename(p_)) in refs, p_
log(f'  dropped {len(dropped)} unreferenced chunks ({sum(d[1] for d in dropped) / 1e6:.1f} MB):')
by = {}
for n_, sz, head in dropped: by.setdefault(os.path.dirname(n_), []).append(os.path.basename(n_) + (' [#PROP_text junk]' if head.startswith(b'#PROP_text') else ''))
for d_, fs in sorted(by.items()): log(f'    {d_}/ ({len(fs)}): ' + ', '.join(sorted(fs)))
# bones / submeshes the bin names must exist on the model
bad = []
for e in b2.entries:
    for f in walk(e.data):
        fn = tables['binfields'].get(f.hash, '') if f.hash else ''
        for ty, v in values(f):
            if fn in ('mBoneName', 'mStartingJointName', 'mEndingJointName') and v not in bones: bad.append((fn, nm(v)))
            if fn == 'attachToBone' and fh(v) not in bones: bad.append((fn, v))
            if fn in ('mShowSubmeshList', 'mHideSubmeshList') and v not in subm: bad.append((fn, nm(v)))
            if fn == 'submesh' and v not in subm.values(): bad.append((fn, v))
log('  bone / submesh references that do not exist on the model:', bad if bad else 'none')
g12 = next(e for e in b2.entries if e.hash == fh('Characters/Talon/Animations/Skin12'))
left = [nm(v) for f in walk(g12.data) for ty, v in values(f) if (tables['binfields'].get(f.hash, '') if f.hash else '') == 'mSubmeshesToDraw']
assert not [x for x in bad if x[0] != 'mSubmeshesToDraw']

# ---------- 4. write WAD (v3.3 like the source), read back ----------
items = []
for h, (n_, d) in content.items():
    if d is None:
        s_ = src[h]
        if s_['ctype'] in (WADCompressionType.Raw, WADCompressionType.Gzip, WADCompressionType.Zstd) and not s_['sub']:
            items.append((int(h, 16), s_['raw'], s_['usize'], s_['ctype'].value, s_['checksum'], s_['data'])); continue
        d = s_['data']
    if n_.endswith(('.bnk', '.wpk')): raw, ct = d, 0
    else: raw, ct = pyzstd.compress(d, 9), 3
    items.append((int(h, 16), raw, len(d), ct, xxh3_64(raw).intdigest(), d))
items.sort(key=lambda x: x[0])
head = b'RW' + bytes([3, 3]) + b'\0' * 256 + struct.pack('<Q', 0) + struct.pack('<I', len(items))
off = len(head) + 32 * len(items); toc = b''; blob = b''
for h, raw, usize, ct, cs, d in items:
    toc += struct.pack('<QIIIBBHQ', h, off + len(blob), len(raw), usize, ct, 0, 0, cs); blob += raw
wad_bytes = head + toc + blob
os.makedirs(BUILD + '/out', exist_ok=True); wp = BUILD + '/out/talon.wad.client'; open(wp, 'wb').write(wad_bytes)
chk = WAD(); chk.read(wp); assert len(chk.chunks) == len(items) and [c.hash for c in chk.chunks] == sorted(c.hash for c in chk.chunks)
with chk.stream(wp, 'rb') as bs:
    for c, it in zip(chk.chunks, items):
        c.read_data(bs); assert int(c.hash, 16) == it[0] and c.data == it[5], c.hash
log(f'\n== WAD: {len(items)} chunks, {len(wad_bytes) / 1e6:.2f} MB (source {os.path.getsize(SRC_WAD) / 1e6:.2f} MB); read back and compared chunk by chunk: OK')

# ---------- 5. fantome ----------
info = json.load(open(SRC_FANTOME + '/META/info.json', encoding='utf-8'))
info['Name'] = 'Ellen Joe • Talon — V2.4' + (' (engine tail sway)' if TAILMOD else ''); info['Version'] = '2.4.0'
info['Description'] = ('Ellen Joe on the upright Talon rig (Wild Rift Talon skin12 animations). V2.4: skeleton re-proportioned to Ellen, '
                       'scissors held in the right hand, animated tail, every animation at the original key rate, bent-elbow sprint homeguard, low lateral tail sway without flutter, compact melee Q, E wall-vault repaired, '
                       'recall choreography and particle effects of the Ellen Joe Samira skin, no crouch-stance leftovers, hand-on-hip idle. '
                       'Built on "aemeth talon" by ceerel (rig and animations) and "ellen joe samira" by disco (Ellen model, recall).')
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('META/info.json', json.dumps(info, indent=2, ensure_ascii=True))
    z.write(SRC_FANTOME + '/META/image.png', 'META/image.png')
    z.write(wp, 'WAD/talon.wad.client', compress_type=zipfile.ZIP_STORED)
with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None and z.read('WAD/talon.wad.client') == wad_bytes
    log('== fantome:', OUT, f'{os.path.getsize(OUT) / 1e6:.2f} MB', [(i.filename, i.file_size) for i in z.infolist()])
