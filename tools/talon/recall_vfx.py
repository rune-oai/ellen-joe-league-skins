"""Source recall events and their transitive particle entries, from the cached Samira skin30 export."""
import struct, json, copy
from pathlib import Path
from binutil import *
def split_entries(path):
    b=Path(path).read_bytes(); assert b[:4]==b"PROP"; version,n=struct.unpack_from("<II",b,4); assert version==3; o=12; links=[]
    for _ in range(n):
        ln=struct.unpack_from("<H",b,o)[0];o+=2;links.append(b[o:o+ln].decode());o+=ln
    n=struct.unpack_from("<I",b,o)[0];o+=4; types=struct.unpack_from("<"+"I"*n,b,o);o+=4*n; result={}
    for ty in types:
        ln,h=struct.unpack_from("<II",b,o); raw=b[o:o+4+ln];o+=4+ln
        result[f"{h:08x}"]=(ty,raw)
    assert o==len(b)
    return result
def decode_entry(ty,raw):
    blob=b"PROP"+struct.pack("<IIII",3,0,1,ty)+raw
    b=BIN();b.read("",raw=blob); assert b.write("",raw=True)==blob
    return b.entries[0]

ANCHORS = {'CSTM_Buffbone_GodsEye1':'knife_01', 'CSTM_Buffbone_GodsEye2':'knife_02',
           'CSTM_Buffbone_GodsEye3':'knife_03', 'Pistol':'knife_04', 'Revolver':'knife_05',
           'Pistol_Muzzle':'knife_06', 'Revolver_Muzzle':'knife_07', 'Sword_Tip2':'BUFFBONE_GLB_WEAPON_1',
           'chest':'chest', 'C_Buffbone_GLB_Layout_Loc':'C_BUFFBONE_GLB_LAYOUT_LOC',
           'BUFFBONE_GLB_GROUND_LOC':'BUFFBONE_GLB_GROUND_LOC'}

def source_vfx(S, fps=29.0):
    root=Path(S)/'dl/samira30'; raw=split_entries(root/'skin30.bin')
    graph=BIN();graph.read(str(root/'animations.bin'))
    g=next(e for e in graph.entries if e.type==fh('AnimationGraphData'))
    recall=dict(get(g.data,'mClipDataMap').data)[fh('Recall')]
    resolver=decode_entry(*raw[fh('Characters/Samira/Skins/Skin30/Resources')])
    rm=dict(get(resolver.data,'resourceMap').data); events=[]; roots=[]; mapping={}
    for k,e in get(recall.data,'mEventDataMap').data:
        if e.hash_type!=fh('ParticleEventData'):continue
        e=copy.deepcopy(e); key=get(e.data,'mEffectKey').data; h=rm[key]; roots.append(h);mapping[key]=h
        for f in walk(e):
            if f.hash in (fh('mStartFrame'),fh('mEndFrame')):f.data*=fps/30.0
            if f.hash==fh('mBoneName'):
                hits=[target for n,target in ANCHORS.items() if fh(n)==f.data];assert len(hits)==1, f.data;f.data=fh(hits[0])
        events.append((fh('EllenRecall:'+k),e))
    # Follow entry links and particlePath strings, retaining only the recall dependency closure.
    todo=roots[:];entries={}
    while todo:
        h=todo.pop()
        if h in entries:continue
        assert h in raw, ('external VFX entry',h)
        e=decode_entry(*raw[h]);entries[h]=e
        for f in walk(e.data):
            for ty,v in values(f):
                if ty==BINType.Link and v!='00000000':todo.append(v)
                if ty==BINType.String and isinstance(v,str) and v.lower().startswith('characters/') and fh(v) in raw:todo.append(fh(v))
    rename={h:fh('Characters/Talon/Skins/Skin0/Particles/EllenRecall/'+h) for h in entries}
    assets={}
    for h,e in entries.items():
        e.hash=rename[h]
        for f in walk(e.data):
            if f.type==BINType.Link and f.data in rename:f.data=rename[f.data]
            if f.type!=BINType.String or not isinstance(f.data,str):continue
            v=f.data
            if v.lower().startswith('characters/') and fh(v) in rename:
                f.data='Characters/Talon/Skins/Skin0/Particles/EllenRecall/'+fh(v)
            elif v.lower().startswith('assets/') and '.' in v.rsplit('/',1)[-1]:
                original=v.lower(); target='assets/ellen_talon_recall/'+original[7:];assets[target]=original;f.data=target
    return events,list(entries.values()),[(key,rename[h]) for key,h in mapping.items()],assets
