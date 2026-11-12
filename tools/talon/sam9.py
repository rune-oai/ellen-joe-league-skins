import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glbanim import Glb
g = Glb(S + '/dl/samira30/model.glb'); times = np.arange(270) / 30.0; smp = g.sample('Recall', times)
for n in ('Pelvis', 'L_Hip', 'L_KneeUpper', 'L_KneeLower', 'L_Foot', 'L_Toe', 'R_Hip', 'R_KneeLower', 'R_Foot', 'Spine1', 'Chest', 'L_Shoulder', 'L_Elbow', 'L_Hand', 'R_Elbow', 'R_Hand', 'Head', 'Neck'):
    T = smp[n][0]; d = np.linalg.norm(T - T[0], axis=1); L0 = np.linalg.norm(T[0]); Sc = smp[n][2]
    print(f'{n:12s} rest len {L0:6.1f}  translation change max {d.max():6.1f} (frame {d.argmax()})  frames > 3 u: {(d > 3).sum():3d}   scale range {Sc.min():.2f}..{Sc.max():.2f}')
