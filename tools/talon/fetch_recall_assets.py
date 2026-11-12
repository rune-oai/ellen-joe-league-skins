"""Fetch only the recall dependency bytes from Riot's Samira WAD, using manifest/bundle ranges."""
import sys, pickle, bisect, json, struct, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests, pyzstd
S=Path(sys.argv[1]);sys.path.insert(0,str(S/'tools/CDTB'));sys.path.insert(0,str(Path(__file__).parent))
from pyritofile import WAD
from binutil import ph
url,files=pickle.load(open(S/'dl/samira_manifest.pkl','rb'));f=files[0]
cache=S/'dl/riot/ranges';cache.mkdir(parents=True,exist_ok=True)
starts=[];o=0
for c in f.chunks:starts.append(o);o+=c.target_size
assert o==f.size
sparse=S/'dl/samira30/source_sparse.wad.client'
if not sparse.exists():
 with sparse.open('wb') as out:out.truncate(f.size)
def chunk(i):
 c=f.chunks[i];p=cache/f'{c.chunk_id:016X}.raw'
 if not p.exists():
  u=f'https://lol.dyn.riotcdn.net/channels/public/bundles/{c.bundle.bundle_id:016X}.bundle'
  last=None
  for attempt in range(3):
   try:
    r=requests.get(u,headers={'Range':f'bytes={c.offset}-{c.offset+c.size-1}'},timeout=40);r.raise_for_status();assert r.status_code==206 and len(r.content)==c.size
    b=pyzstd.decompress(r.content);assert len(b)==c.target_size;p.write_bytes(b);break
   except Exception as ex:last=ex
  else:raise last
 return i,p.read_bytes()
def ensure(off,n):
 ids=list(range(max(0,bisect.bisect_right(starts,off)-1),bisect.bisect_right(starts,off+n-1)))
 with ThreadPoolExecutor(max_workers=12) as pool, sparse.open('r+b') as out:
  for i,b in pool.map(chunk,ids):out.seek(starts[i]);out.write(b)
def read(off,n):
 ensure(off,n)
 with sparse.open('rb') as out:out.seek(off);return out.read(n)
b=read(0,272);assert b[:2]==b'RW';n=struct.unpack_from('<I',b,268)[0];ensure(272,n*32)
w=WAD();w.read(str(sparse));by={c.hash:c for c in w.chunks};assets=json.load(open(S/'dl/recall_assets.json'));missing=[];needed=[]
for target,original in assets.items():
 c=by.get(ph(original))
 if c is None:missing.append(original)
 else:needed.append((target,original,c))
print('WAD',len(w.chunks),'assets found',len(needed),'missing',missing,flush=True)
assert not missing
ids=set()
for target,original,c in needed:ids.update(range(max(0,bisect.bisect_right(starts,c.offset)-1),bisect.bisect_right(starts,c.offset+c.compressed_size-1)))
print('fetching',len(ids),'manifest ranges, compressed MB',sum(f.chunks[i].size for i in ids)/1e6,flush=True)
with ThreadPoolExecutor(max_workers=12) as pool,sparse.open('r+b') as out:
 for i,b in pool.map(chunk,sorted(ids)):out.seek(starts[i]);out.write(b)
manifest=[]
with w.stream(str(sparse),'rb') as bs:
 for target,original,c in needed:
  c.read_data(bs);assert len(c.data)==c.decompressed_size
  p=S/'dl/recall_assets'/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(c.data)
  manifest.append(dict(target=target,original=original,size=len(c.data),sha256=hashlib.sha256(c.data).hexdigest()))
(S/'dl/recall_assets_manifest.json').write_text(json.dumps(dict(riot_manifest=url,files=manifest),indent=2))
print('saved all',len(manifest),'raw texture/mesh dependencies',flush=True)
