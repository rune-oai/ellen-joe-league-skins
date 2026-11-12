import sys, os, pickle, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
D = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'; B = 'assets/ice/wild/bum/characters/talon/skins/'
skS = L.load_skl(D + 'samira_skin30.skl'); mS = L.load_skn(D + 'samira_skin30.skn')
RIG = S + '/build22/wad/' + B; skN = L.load_skl(RIG + 'skin12/talon_skin12.skl'); mN = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
names, T, R, Sc, fps = pickle.load(open(S + '/dl/samira30_recall.pkl', 'rb'))
a = L.Anim(); a.hashes = np.array([L.elf(n) for n in names], np.uint32); a.sig = 'r3d2anmd'; a.has = np.ones((len(names), 3), bool); a.T, a.R, a.S = T, R, Sc; a.nframes = len(T); a.fps = fps
ibS = skS.ibind[skS.index['Sword']]
print('hand joints (R_Hand, L_Hand) in Sword bind-mesh space per frame:')
for f in list(range(84, 126, 3)) + list(range(141, 165, 3)) + [213, 216, 228, 231, 240, 249]:
    W = L.pose_world(skS, a, f); Mi = np.linalg.inv(W[skS.index['Sword']] @ ibS)
    r = (Mi @ np.r_[W[skS.index['R_Buffbone_Glb_Hand_Loc']][:3, 3], 1])[:3]; l = (Mi @ np.r_[W[skS.index['L_Buffbone_Glb_Hand_Loc']][:3, 3], 1])[:3]
    print(f, 'R', np.round(r, 1), 'L', np.round(l, 1))
# talon rig: tracks of a base clip and dominant vertex counts per joint
a2 = L.load_anm(S + '/w/aemeath_talon/' + B + 'skin12/animations/talon_skin12_channel.anm'); hs = set(int(h) for h in a2.hashes)
infN = np.array(skN.influences)[mN.infl]; print('talon joints: name tracks? verts(w>0.3) ')
for j, n in enumerate(skN.names):
    c = int((((infN == j) * mN.w).sum(1) > 0.3).sum()); print(f'  {n:28s} {"T" if L.elf(n) in hs else "-"} {c}')
print('tracks not in skeleton:', len([h for h in hs if h not in set(L.elf(n) for n in skN.names)]))
