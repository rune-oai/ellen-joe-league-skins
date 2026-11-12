"""Regression checks for the client-reported arm, tail flutter and absent recall VFX."""
import sys,json,zipfile,io
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import lolio as L
from binutil import *
from recall_vfx import source_vfx, ANCHORS
from pyritofile import WAD
S=Path(sys.argv[1]); build=S/'build24/wad'; base='assets/ice/wild/bum/characters/talon/skins/skin12/'
sk=L.load_skl(str(build/base/'talon_skin12.skl')); m=L.load_skn(str(build/base/'ellen/ellen_talon.skn'))
for version in ('build23','build24'):
    for clip in ('run','homeguard'):
        a=L.load_anm(str(S/version/'wad'/base/f'animations/talon_skin12_{clip}.anm'))
        W=np.array([L.pose_world(sk,a,f) for f in range(a.nframes)]);p=W[:,:,:3,3]
        v=p[:,sk.index['l_forearm']]-p[:,sk.index['l_uparm']];w=p[:,sk.index['l_hand']]-p[:,sk.index['l_forearm']]
        bend=np.degrees(np.arccos(np.clip((v*w).sum(1)/np.linalg.norm(v,axis=1)/np.linalg.norm(w,axis=1),-1,1)))
        tip=p[:,sk.index['tail16']]-p[:,sk.index['tail1']];span=np.ptp(tip,axis=0);acc=np.diff(np.r_[tip[-2:-1],tip,tip[1:2]],n=2,axis=0)
        print(version,clip,'elbow bend',bend.min(),bend.max(),'tail relative tip span XYZ',span,'max vertical second difference',abs(acc[:,1]).max())
        if version=='build24':
            assert bend.min()>45 and bend.max()<125
            assert span[0] > 2*span[1], 'tail motion must be mainly lateral'
            assert abs(acc[:,1]).max()<1.0, 'no high-frequency vertical flutter'
# Inspect final packaged event, resolver, definition and dependency wiring.
fan=Path(sys.argv[2]); out=S/'review24/check_pack.wad.client'
with zipfile.ZipFile(fan) as z:assert z.testzip() is None;out.write_bytes(z.read('WAD/talon.wad.client'))
w=WAD();w.read(str(out));chunks={}
with w.stream(str(out),'rb') as bs:
    for c in w.chunks:c.read_data(bs);chunks[c.hash]=c.data
b=BIN();b.read('',raw=chunks[ph('data/characters/talon/skins/skin0.bin')]);assert b.write('',raw=True)==chunks[ph('data/characters/talon/skins/skin0.bin')]
ent={e.hash:e for e in b.entries};g=ent[fh('Characters/Talon/Animations/Skin12')];rc=dict(get(g.data,'mClipDataMap').data)['9260a8e8'];events=[(k,e) for k,e in get(rc.data,'mEventDataMap').data if e.hash_type==fh('ParticleEventData')]
expect,fx,rm,assets=source_vfx(str(S));assert len(events)==len(expect)==17
for (k,e),(k0,e0) in zip(events,expect):
    assert k==k0
    for f in walk(e):
        if f.hash==fh('mStartFrame'):assert 0<=f.data<=259
        if f.hash==fh('mBoneName'):assert f.data in {fh(n) for n in sk.names}
for e in fx:assert e.hash in ent
skin=ent[fh('Characters/Talon/Skins/Skin0')];resolver=ent[get(skin.data,'mResourceResolver').data];actual=dict(get(resolver.data,'resourceMap').data)
assert all(actual.get(k)==h and h in ent for k,h in rm)
manifest=json.load(open(S/'dl/recall_assets_manifest.json'))
import hashlib
for f in manifest['files']:
    assert hashlib.sha256(chunks[ph(f['target'])]).hexdigest()==f['sha256']
print('PASS:',len(events),'recall particle events;',len(fx),'systems;',len(assets),'verified raw dependencies; valid bones/resolver; BIN roundtrip')
