"""Package-only motion checks and source-GLB attachment correspondence. No build generator imported."""
import sys,copy,zipfile,hashlib,struct
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).parent))
import lolio as L
from glbanim import Glb
from binutil import *
from pyritofile import WAD
S=Path(sys.argv[1]); fan=Path(sys.argv[2]); O=S/'review24/complete_checks';O.mkdir(exist_ok=True)
assert hashlib.sha256(fan.read_bytes()).hexdigest()=='b8f8813b6df5f5b257a1a259249688d5f98ffca61c78b510aca4bc328172c3ec'
with zipfile.ZipFile(fan) as z:(O/'candidate.wad').write_bytes(z.read('WAD/talon.wad.client'))
w=WAD();w.read(str(O/'candidate.wad'));chunks={}
with w.stream(str(O/'candidate.wad'),'rb') as bs:
 for c in w.chunks:c.read_data(bs);chunks[c.hash]=c.data
base='assets/ice/wild/bum/characters/talon/skins/skin12/'
def file(rel):
 p=O/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(chunks[ph(base+rel)]);return str(p)
sk=L.load_skl(file('talon_skin12.skl'));m=L.load_skn(file('ellen/ellen_talon.skn'))
def anim(n):return L.load_anm(file('animations/'+n+'.anm'))
a=anim('talon_recall');CW=np.array([L.pose_world(sk,a,f) for f in range(a.nframes)])
source=S/'w/samira/assets/.sami0_characters/samira/skins/skin30'
sks=L.load_skl(str(source/'samira_skin30.skl'));ms=L.load_skn(str(source/'samira_skin30.skn'));g=Glb(str(S/'dl/samira30/model.glb'))
smp=g.sample('Recall',np.minimum(np.arange(a.nframes)/a.fps,244/30));names=[n for n in sks.names if n in smp]
asrc=L.Anim();asrc.sig='r3d2anmd';asrc.hashes=np.array([L.elf(n) for n in names]);asrc.T=np.stack([smp[n][0] for n in names],1);asrc.R=np.stack([smp[n][1] for n in names],1);asrc.S=np.stack([smp[n][2] for n in names],1)
asrc.has=np.ones((len(names),3),bool)
SW=np.array([L.pose_world(sks,asrc,f) for f in range(a.nframes)])
face=next(np.arange(vs,vs+vc) for n,vs,vc,_,_ in ms.subs if n=='Ellen_Face_D');sword=next(np.arange(vs,vs+vc) for n,vs,vc,_,_ in m.subs if n=='sword')
def scale(sel):
 ps=ms.pos[sel]-ms.pos[sel].mean(0);pn=m.pos[sel]-m.pos[sel].mean(0)
 return np.sqrt((pn*pn).sum()/(ps*ps).sum())
sb=scale(face);sf=scale(sword);print('body/scissors scales',sb,sf)
inf=np.array(sk.influences)[m.infl];feet=np.where((np.isin(inf,[sk.index[n] for n in ('l_foot','r_foot','l_toe','r_toe')])*m.w).sum(1)>.5)[0]
tk=lambda f:round(f/30*a.fps); kp=tk(180);ps=L.deform(ms,sks,SW[kp])[0];tip0=ms.pos[sword[np.argmin(ps[sword,1])]];dl=[]
for f in range(tk(229),tk(244)+1):
 p=L.deform(ms,sks,SW[f])[0];foot=p[feet][np.argsort(p[feet,1])[:30]].mean(0);v=p[sword][np.linalg.norm(p[sword]-foot,axis=1).argmin()]
 blade=SW[f,sks.index['Sword']]@sks.ibind[sks.index['Sword']];dl.append(v-(blade@np.r_[tip0,1])[:3])
dperch=-(sb-sf)*np.mean(dl,axis=0);u=np.clip((np.minimum(np.arange(a.nframes)/a.fps,244/30)*30-216)/10,0,1);u=u*u*(3-2*u)
print('independently reconstructed perch shift',dperch)
for src,dst,hand in [('CSTM_Buffbone_GodsEye1','knife_01',None),('CSTM_Buffbone_GodsEye2','knife_02',None),('CSTM_Buffbone_GodsEye3','knife_03',None),('Pistol','knife_04','r_hand'),('Revolver','knife_05','l_hand'),('Pistol_Muzzle','knife_06','r_hand'),('Revolver_Muzzle','knife_07','l_hand')]:
 j=sk.index[dst];assert sk.parents[j]<0 and not ((inf==j)*m.w).any();expected=sb*SW[:,sks.index[src],:3,3]
 if hand:
  sh='R_Hand' if hand=='r_hand' else 'L_Hand';expected+=CW[:,sk.index[hand],:3,3]-sb*SW[:,sks.index[sh],:3,3]
 else:expected+=u[:,None]*dperch
 error=np.linalg.norm(CW[:,j,:3,3]-expected,axis=1).max();rot=max(np.abs(CW[f,j,:3,:3]-np.linalg.svd(SW[f,sks.index[src],:3,:3])[0]@np.linalg.svd(SW[f,sks.index[src],:3,:3])[2]).max() for f in range(a.nframes))
 print(src,'all-key max position/orientation matrix error',error,rot);assert error<1e-3 and rot<1e-4
j=sk.index['BUFFBONE_GLB_WEAPON_1'];worst=0
for f in range(a.nframes):
 p=L.deform(m,sk,CW[f])[0];worst=max(worst,np.linalg.norm(p[sword]-CW[f,j,:3,3],axis=1).min())
print('blade-tip anchor max distance to blade vertex',worst);assert worst<1e-3
A={n:anim(n) for n in ['talon_skin12_idle1','talon_skin12_run','talon_skin12_homeguard','talon_skin12_homeguard_in','talon_spell3_long','talon_spell3_medium','talon_skin12_spell3_longtorun','talon_skin12_spell3_medium_torun','talon_recall']}
def mesh(a,f):return L.deform(m,sk,L.pose_world(sk,a,f))[0]
pairs=[('talon_skin12_run','talon_skin12_homeguard_in',0,0),('talon_skin12_homeguard_in','talon_skin12_homeguard',-1,0),('talon_spell3_long','talon_skin12_spell3_longtorun',-1,0),('talon_spell3_medium','talon_skin12_spell3_medium_torun',-1,0),('talon_skin12_spell3_longtorun','talon_skin12_run',-1,0),('talon_skin12_spell3_medium_torun','talon_skin12_run',-1,0),('talon_recall','talon_skin12_idle1',-1,0)]
pairs += [(n,n,0,-1) for n in ('talon_skin12_idle1','talon_skin12_run','talon_skin12_homeguard')]
for n,q,f,h in pairs:
 d=np.linalg.norm(mesh(A[n],f)-mesh(A[q],h),axis=1).max();print('mesh seam',n,q,d);assert d<1e-3
own=inf[np.arange(len(inf)),m.w.argmax(1)];tail=np.isin(own,[sk.index[f'tail{i}'] for i in range(1,17)]);blade=own==sk.index['weapon']
for name in ('talon_skin12_idle1','talon_skin12_run','talon_skin12_homeguard','talon_skin12_homeguard_in'):
 a=A[name];q=copy.copy(a);q.T=a.T[:1].copy();q.R=a.R[:1].copy();q.S=a.S[:1].copy();gap=1e9;y=1e9
 for f in range(a.nframes-1):
  for u in (0,.25,.5,.75,1):
   q.T[0]=(1-u)*a.T[f]+u*a.T[f+1];q.S[0]=(1-u)*a.S[f]+u*a.S[f+1];q.R[0]=L.slerp(a.R[f],a.R[f+1],np.full(len(a.hashes),u));p=mesh(q,0);gap=min(gap,cKDTree(p[blade]).query(p[tail])[0].min());y=min(y,p[tail,1].min())
 print('quarter-key clearance',name,'tail/blade nearest vertices',gap,'tail ground',y);assert gap>10 and y>0
# Direct byte equality with original Riot-WAD chunks, not only the extraction manifest hashes.
orig=WAD();orig.read(str(S/'dl/samira30/source_sparse.wad.client'));by={c.hash:c for c in orig.chunks}
import json
mapping=json.load(open(S/'dl/recall_assets.json'));scb=0
with orig.stream(str(S/'dl/samira30/source_sparse.wad.client'),'rb') as bs:
 for target,source in mapping.items():
  c=by[ph(source)];c.read_data(bs);data=chunks[ph(target)];assert data==c.data
  if target.endswith('.scb'):
   # Parse every face record, including degenerate faces, without the library's skip.
   assert data[:8]==b'r3d2Mesh'
   major,minor=struct.unpack_from('<HH',data,8); assert (major,minor) in ((2,1),(3,1),(3,2))
   nv,nf,flags=struct.unpack_from('<III',data,140); assert nv and nf
   assert np.isfinite(struct.unpack_from('<6f',data,152)).all()
   off=176; vt=0
   if (major,minor)==(3,2): vt,=struct.unpack_from('<I',data,off);off+=4
   assert vt in (0,1); assert np.isfinite(struct.unpack_from('<%df'%(nv*3),data,off)).all();off+=nv*12+nv*4*vt
   assert np.isfinite(struct.unpack_from('<3f',data,off)).all();off+=12
   for face in range(nf):
    ids=struct.unpack_from('<3I',data,off);assert max(ids)<nv
    assert np.isfinite(struct.unpack_from('<6f',data,off+76)).all();off+=100
   assert off==len(data), (target,off,len(data))
   scb+=1
print('PASS original WAD byte equality',len(mapping),'dependencies; full SCB parsing',scb,'meshes; anchors and motion seams. No client/VFX renderer used.')
