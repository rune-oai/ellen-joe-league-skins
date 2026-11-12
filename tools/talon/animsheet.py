"""animsheet.py S <mod: aemeath|ellen|new> out.png [yaw] [clip,clip,...] [frames-per-clip]"""
import sys, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lolio as L, render as R
B = 'assets/ice/wild/bum/characters/talon/skins/'
def setup(mod):
    if mod == 'aemeath':
        root = S + '/w/aemeath_talon/'; A = root + B + 'skin12/'
        sk = L.load_skl(A + 'talon_skin12.skl'); m = L.load_skn(A + 'talon_skin12.skn')
        tex = {'Body': L.load_image(A + 'atlas_00003.dds'), 'Ribbon': L.load_image(A + 'atlas_00003.dds'), 'Sword': L.load_image(A + 'atlas_00006.dds')}
    elif mod == 'ellen':
        root = S + '/w/ellen_talon/'; A = root + B + 'skin12/'
        sk = L.load_skl(A + 'talon_skin12.skl'); m = L.load_skn(A + 'ellen/ellen_talon.skn')
        body = L.load_image(A + 'ellen/ellen_body_tx_cm.tex'); tex = {n: body for n, *_ in m.subs}; tex['sword'] = L.load_image(A + 'ellen/ellen_scissors_tx_cm.dds')
    else:
        root = os.environ.get('LOL_RIG', S + '/build') + '/wad/'; A = root + B + 'skin12/'
        sk = L.load_skl(A + 'talon_skin12.skl'); m = L.load_skn(A + 'ellen/ellen_talon.skn')
        body = L.load_image(A + 'ellen/ellen_body_tx_cm.tex'); tex = {n: body for n, *_ in m.subs}; tex['sword'] = L.load_image(A + 'ellen/ellen_scissors_tx_cm.dds')
    return root, sk, m, tex
CL = [('Idle1', 'skin12/animations/talon_skin12_idle1.anm'), ('Run', 'skin12/animations/talon_skin12_run.anm'), ('Attack1', 'skin12/animations/talon_skin12_attack1.anm'), ('Attack2', 'skin12/animations/talon_skin12_attack2.anm'),
      ('Crit', 'skin12/animations/talon_skin12_crit.anm'), ('Spell1', 'skin12/animations/talon_skin12_spell1.anm'), ('Spell1_Leap', 'skin12/animations/talon_skin12_spell1_leap.anm'), ('Spell2', 'skin12/animations/talon_skin12_spell2.anm'),
      ('Spell3', 'skin12/animations/talon_skin12_spell3_jump.anm'), ('Spell3_Long', 'skin20/animations/talon_skin20_spell3_long_healthbar_fixed.anm'), ('Spell3_Medium', 'skin20/animations/talon_skin20_spell3_medium_healthbar_fixed.anm'),
      ('S3_longtorun', 'skin12/animations/talon_skin12_spell3_longtorun.anm'), ('S3_medtorun', 'skin12/animations/talon_skin12_spell3_medium_torun.anm'), ('Spell4', 'skin12/animations/talon_skin12_spell4.anm'),
      ('Channel', 'skin12/animations/talon_skin12_channel.anm'), ('Channel_Wndup', 'skin12/animations/talon_skin12_channel_windup.anm'), ('Recall', 'skin12/animations/talon_recall.anm'), ('Homeguard', 'skin12/animations/talon_skin12_homeguard.anm'),
      ('Homeguard_IN', 'skin12/animations/talon_skin12_homeguard_in.anm'), ('Dance', 'skin12/animations/talon_skin12_dance.anm'), ('Taunt', 'skin12/animations/talon_skin12_taunt.anm'), ('Joke', 'skin12/animations/talon_skin12_joke.anm'),
      ('Laugh', 'skin12/animations/talon_skin12_laugh.anm'), ('Death', 'skin12/animations/talon_skin12_death.anm')]
if __name__ == '__main__':
    mod = sys.argv[2]; out = sys.argv[3]; yaw = float(sys.argv[4]) if len(sys.argv) > 4 else 30
    only = sys.argv[5].split(',') if len(sys.argv) > 5 and sys.argv[5] else None
    nf = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    SIZE = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (300, 340)
    HH = float(sys.argv[9]) if len(sys.argv) > 9 else 125
    root, sk, m, tex = setup(mod)
    parts = [(i0//3, ic//3, tex.get(n), (200, 200, 200)) for n, vs, vc, i0, ic in m.subs]
    ims, labels = [], []
    for name, rel in CL:
        if only and name not in only: continue
        if mod == 'new': rel = rel.replace('skin20/animations/talon_skin20_spell3_long_healthbar_fixed', 'skin12/animations/talon_spell3_long').replace('skin20/animations/talon_skin20_spell3_medium_healthbar_fixed', 'skin12/animations/talon_spell3_medium')
        a = L.load_anm(root + B + rel)
        for f in np.linspace(0, a.nframes - 1, nf).round().astype(int):
            w = L.pose_world(sk, a, int(f)); p, n = L.deform(m, sk, w)
            c = w[sk.index['spine'], :3, 3]
            ims.append(R.render(p, n, m.uv, m.idx, parts, size=SIZE, yaw=yaw, center=(c[0], c[1]-15, c[2]), half_h=HH)); labels.append(f'{name} f{f}/{a.nframes}')
    R.sheet(ims, labels, cols=nf*2 if nf <= 5 else nf).save(out); print('saved', out, len(ims))
