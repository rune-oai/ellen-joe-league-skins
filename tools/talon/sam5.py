import sys, os, pickle, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L
D = S + '/w/samira/assets/.sami0_characters/samira/skins/skin30/'; B = 'assets/ice/wild/bum/characters/talon/skins/'
skS = L.load_skl(D + 'samira_skin30.skl'); mS = L.load_skn(D + 'samira_skin30.skn')
RIG = S + '/build22/wad/' + B; skN = L.load_skl(RIG + 'skin12/talon_skin12.skl'); mN = L.load_skn(RIG + 'skin12/ellen/ellen_talon.skn')
names, T, R, Sc, fps = pickle.load(open(S + '/dl/samira30_recall.pkl', 'rb'))
a = L.Anim(); a.hashes = np.array([L.elf(n) for n in names], np.uint32); a.sig = 'r3d2anmd'; a.has = np.ones((len(names), 3), bool); a.T, a.R, a.S = T, R, Sc; a.nframes = len(T); a.fps = fps
sw = next((vs, vc) for n, vs, vc, i0, ic in mS.subs if n == 'sword'); sel = np.arange(sw[0], sw[0] + sw[1])
# similarity sword: samira bind -> talon bind
P = mS.pos[sel]; Q = mN.pos[sel]; pc = P.mean(0); qc = Q.mean(0); U, sg, Vt = np.linalg.svd((Q - qc).T @ (P - pc)); Rf = U @ Vt; sf = sg.sum() / ((P - pc) ** 2).sum(); tf = qc - sf * Rf @ pc
gN = np.linalg.inv(skN.ibind)[skN.index['weapon']][:3, 3]; gS = Rf.T @ (gN - tf) / sf      # grip point (talon weapon joint origin) in samira bind mesh space
ibS = skS.ibind[skS.index['Sword']]; gL = (ibS @ np.r_[gS, 1])[:3]; VL = (np.c_[P, np.ones(len(P))] @ ibS.T)[:, :3]
print('grip point in samira bind space', np.round(gS, 1), 'in Sword joint space', np.round(gL, 1), 'sword verts in joint space bbox', np.round(VL.min(0), 1), np.round(VL.max(0), 1))
print('Sword_R_Hand_Snap bind', np.round(np.linalg.inv(skS.ibind)[skS.index['Sword_R_Hand_Snap']][:3, 3], 1), 'R_Hand bind', np.round(np.linalg.inv(skS.ibind)[skS.index['R_Hand']][:3, 3], 1))
inf = np.array(skS.influences)[mS.infl]; feet = np.where(sum(((inf == skS.index[n]) * mS.w).sum(1) for n in ('L_Foot', 'R_Foot', 'L_Toe', 'R_Toe')) > 0.5)[0]
print(' f   rootY  soleMinY  swordMinY swordMaxY  grip-RHand grip-LHand  gripY   tipdir_y')
far = VL[np.argmax(np.linalg.norm(VL - gL, axis=1))]
for f in range(0, a.nframes, 3):
    W = L.pose_world(skS, a, f); M = W @ skS.ibind; Ws = W[skS.index['Sword']]; g = Ws[:3, :3] @ gL + Ws[:3, 3]; V = VL @ Ws[:3, :3].T + Ws[:3, 3]
    fy = (np.einsum('nkij,nj->nki', M[inf[feet]][:, :, :3, :], np.c_[mS.pos[feet], np.ones(len(feet))]) * mS.w[feet][:, :, None]).sum(1)[:, 1].min()
    d = Ws[:3, :3] @ (far - gL); d /= np.linalg.norm(d)
    print(f'{f:3d} {W[0][1, 3]:7.1f} {fy:8.1f} {V[:, 1].min():9.1f} {V[:, 1].max():9.1f} {np.linalg.norm(g - W[skS.index["R_Hand"]][:3, 3]):9.1f} {np.linalg.norm(g - W[skS.index["L_Hand"]][:3, 3]):9.1f} {g[1]:7.1f} {d[1]:7.2f}')
