"""sam3.py S -> compare the Ellen mesh bound to Samira's skeleton with the one bound to the Talon rig"""
import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
D = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'; B = 'assets/ice/wild/bum/characters/talon/skins/'
skS = L.load_skl(D + 'samira_skin30.skl'); mS = L.load_skn(D + 'samira_skin30.skn')
RIG = S + '/build22/wad/' + B; skN = L.load_skl(RIG + 'skin12/talon_skin12.skl'); mN = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
for tag, m in (('samira', mS), ('talon', mN)):
    print(tag, 'verts', len(m.pos), 'bbox', np.round(m.pos.min(0), 1), np.round(m.pos.max(0), 1))
    for n, vs, vc, i0, ic in m.subs: p = m.pos[vs:vs + vc]; print(f'   {n:22s} {vc:6d} bbox {np.round(p.min(0), 1)} {np.round(p.max(0), 1)}')
print('talon rig joints:'); bw = np.linalg.inv(skN.ibind)
for j, n in enumerate(skN.names): print(' ', j, n, '<-', skN.names[skN.parents[j]] if skN.parents[j] >= 0 else '-', np.round(bw[j][:3, 3], 1))
