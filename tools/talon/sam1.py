"""sam1.py S -> inspect the Ellen-on-Samira mod rig and the downloaded Samira skin30 GLB"""
import sys, os, json, struct, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
D = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'
sk = L.load_skl(D + 'samira_skin30.skl'); m = L.load_skn(D + 'samira_skin30.skn')
print('skl joints', len(sk.names)); inf = np.array(sk.influences)[m.infl]
for n, vs, vc, i0, ic in m.subs:
    sel = np.arange(vs, vs + vc); w = np.zeros(len(sk.names))
    for k in range(4): np.add.at(w, inf[sel, k], m.w[sel, k])
    top = np.argsort(-w)[:6]; print(f'  sub {n:22s} verts {vc:6d}  ' + ', '.join(f'{sk.names[j]} {w[j] / vc * 100:.0f}%' for j in top if w[j] > 0))
for j, n in enumerate(sk.names): print(j, n, '<-', sk.names[sk.parents[j]] if sk.parents[j] >= 0 else '-', np.round(np.linalg.inv(sk.ibind[j])[:3, 3], 1))
b = open(S + '/dl/samira30/model.glb', 'rb').read(); jl = struct.unpack('<I', b[12:16])[0]; g = json.loads(b[20:20 + jl])
print('glb anims:', [(a['name'], len(a['channels'])) for a in g['animations'] if 'ecall' in a['name'] or 'omeguard' in a['name']])
names = [g['nodes'][j].get('name') for j in g['skins'][0]['joints']]
print('glb joints not in skl:', [n for n in names if n.lower() not in [x.lower() for x in sk.names]]); print('skl joints not in glb:', [n for n in sk.names if n.lower() not in [x.lower() for x in names]])
