"""Numpy helpers for League SKL / SKN / ANM / TEX files (raw file space throughout)."""
import struct, io
import numpy as np
from pyritofile import read_skl, read_skn
from pyritofile.ermmm import Elf

def elf(name): return Elf(name.lower()) & 0xFFFFFFFF

# ---------- math ----------
def quat_to_mat(q):
    q = np.asarray(q, dtype=np.float64); q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2*(y*y + z*z); m[..., 0, 1] = 2*(x*y - z*w); m[..., 0, 2] = 2*(x*z + y*w)
    m[..., 1, 0] = 2*(x*y + z*w); m[..., 1, 1] = 1 - 2*(x*x + z*z); m[..., 1, 2] = 2*(y*z - x*w)
    m[..., 2, 0] = 2*(x*z - y*w); m[..., 2, 1] = 2*(y*z + x*w); m[..., 2, 2] = 1 - 2*(x*x + y*y)
    return m

def mat_to_quat(m):
    m = np.asarray(m, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; q = [(m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s, 0.25*s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; q = [0.25*s, (m[0, 1]+m[1, 0])/s, (m[0, 2]+m[2, 0])/s, (m[2, 1]-m[1, 2])/s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; q = [(m[0, 1]+m[1, 0])/s, 0.25*s, (m[1, 2]+m[2, 1])/s, (m[0, 2]-m[2, 0])/s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; q = [(m[0, 2]+m[2, 0])/s, (m[1, 2]+m[2, 1])/s, 0.25*s, (m[1, 0]-m[0, 1])/s]
    q = np.array(q); return q / np.linalg.norm(q)

def trs(t, q, s):
    """Compose (...,4,4) from translate (...,3), quat xyzw (...,4), scale (...,3)."""
    t = np.asarray(t, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    m = np.zeros(t.shape[:-1] + (4, 4)); r = quat_to_mat(q)
    m[..., :3, :3] = r * s[..., None, :]; m[..., :3, 3] = t; m[..., 3, 3] = 1
    return m

def slerp(a, b, f):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = np.sum(a*b, -1, keepdims=True); b = np.where(d < 0, -b, b); d = np.abs(d)
    f = np.asarray(f)[..., None] if np.ndim(f) else f
    th = np.arccos(np.clip(d, -1, 1)); st = np.sin(th)
    lin = st < 1e-5
    wa = np.where(lin, 1 - f, np.sin((1 - f)*th) / np.where(lin, 1, st)); wb = np.where(lin, f, np.sin(f*th) / np.where(lin, 1, st))
    r = wa*a + wb*b; return r / np.linalg.norm(r, axis=-1, keepdims=True)

# ---------- SKL ----------
class Skel:
    pass

def load_skl(path):
    s = read_skl(path); k = Skel(); n = len(s.joints)
    k.raw = s; k.names = [j.name for j in s.joints]; k.parents = np.array([j.parent for j in s.joints])
    k.hashes = np.array([elf(nm) for nm in k.names], dtype=np.uint32)
    k.lt = np.array([[j.local_translate.x, j.local_translate.y, j.local_translate.z] for j in s.joints], dtype=np.float64)
    k.lr = np.array([[j.local_rotate.x, j.local_rotate.y, j.local_rotate.z, j.local_rotate.w] for j in s.joints], dtype=np.float64)
    k.ls = np.array([[j.local_scale.x, j.local_scale.y, j.local_scale.z] for j in s.joints], dtype=np.float64)
    k.it = np.array([[j.ibind_translate.x, j.ibind_translate.y, j.ibind_translate.z] for j in s.joints], dtype=np.float64)
    k.ir = np.array([[j.ibind_rotate.x, j.ibind_rotate.y, j.ibind_rotate.z, j.ibind_rotate.w] for j in s.joints], dtype=np.float64)
    k.isc = np.array([[j.ibind_scale.x, j.ibind_scale.y, j.ibind_scale.z] for j in s.joints], dtype=np.float64)
    k.influences = list(s.influences)
    k.world = world_from_local(k.parents, trs(k.lt, k.lr, k.ls))
    k.ibind = trs(k.it, k.ir, k.isc)           # inverse bind as stored in file
    k.index = {nm: i for i, nm in enumerate(k.names)}
    return k

def world_from_local(parents, local):
    """local: (...,J,4,4). Parents are not guaranteed to precede children."""
    J = len(parents); world = np.empty_like(local); done = np.zeros(J, bool)
    order = []
    def visit(i):
        if done[i]: return
        if parents[i] >= 0: visit(parents[i])
        done[i] = True; order.append(i)
    for i in range(J): visit(i)
    for i in order:
        p = parents[i]
        world[..., i, :, :] = local[..., i, :, :] if p < 0 else world[..., p, :, :] @ local[..., i, :, :]
    return world

# ---------- SKN ----------
class Skin:
    pass

def load_skn(path):
    k = read_skn(path); m = Skin(); m.raw = k
    v = k.vertices
    m.pos = np.array([[x.position.x, x.position.y, x.position.z] for x in v], dtype=np.float64)
    m.nrm = np.array([[x.normal.x, x.normal.y, x.normal.z] for x in v], dtype=np.float64)
    m.uv = np.array([[x.uv.x, x.uv.y] for x in v], dtype=np.float64)
    m.infl = np.array([x.influences for x in v], dtype=np.int64)
    m.w = np.array([x.weights for x in v], dtype=np.float64)
    m.idx = np.array(k.indices, dtype=np.int64).reshape(-1, 3)
    m.subs = [(s.name, s.vertex_start, s.vertex_count, s.index_start, s.index_count) for s in k.submeshes]
    return m

# ---------- ANM ----------
class Anim:
    pass

def _dq(raw):
    """Decompress (N,6) uint8 -> (N,4) quats xyzw."""
    raw = raw.astype(np.uint64)
    bits = raw[:, 0] | raw[:, 1] << 8 | raw[:, 2] << 16 | raw[:, 3] << 24 | raw[:, 4] << 32 | raw[:, 5] << 40
    mi = ((bits >> 45) & 3).astype(np.int64)
    k1, k2 = 0.70710678118, 0.00004315969
    a = ((bits >> 30) & 32767).astype(np.float64)*k2 - k1; b = ((bits >> 15) & 32767).astype(np.float64)*k2 - k1; c = (bits & 32767).astype(np.float64)*k2 - k1
    d = np.sqrt(np.maximum(0, 1 - (a*a + b*b + c*c)))
    out = np.empty((len(raw), 4))
    for i, cols in enumerate([(d, a, b, c), (a, d, b, c), (a, b, d, c), (a, b, c, d)]):
        sel = mi == i
        for j in range(4): out[sel, j] = cols[j][sel]
    return out

def load_anm(path):
    """Returns Anim with: fps, nframes, hashes (T,), T (F,T,3), R (F,T,4), S (F,T,3) dense local transforms."""
    data = open(path, 'rb').read(); a = Anim(); sig = data[:8]; ver = struct.unpack_from('<I', data, 8)[0]
    a.sig = sig.decode(); a.version = ver
    if sig == b'r3d2canm':
        (size, token, flags, ntr, nfr, njc, maxt, fps) = struct.unpack_from('<IIIIIiff', data, 12)
        o = 12 + 32
        em = struct.unpack_from('<6f', data, o); o += 24
        tmin = np.array(struct.unpack_from('<3f', data, o)); tmax = np.array(struct.unpack_from('<3f', data, o+12))
        smin = np.array(struct.unpack_from('<3f', data, o+24)); smax = np.array(struct.unpack_from('<3f', data, o+36)); o += 48
        fo, jco, jho = struct.unpack_from('<iii', data, o)
        a.fps = fps; a.maxt = maxt; a.flags = flags; a.em = em
        a.hashes = np.frombuffer(data, '<u4', ntr, jho + 12).copy()
        fr = np.frombuffer(data, np.dtype([('t', '<u2'), ('b', '<u2'), ('v', 'u1', 6)]), nfr, fo + 12)
        nF = max(int(round(maxt * fps)) + 1, 1); a.nframes = nF
        times = fr['t'].astype(np.float64) / 65535.0 * maxt * fps
        jid = (fr['b'] & 0x3FFF).astype(np.int64); typ = (fr['b'] >> 14).astype(np.int64)
        vals16 = fr['v'].astype(np.float64); u = (fr['v'][:, 0::2].astype(np.float64) + fr['v'][:, 1::2].astype(np.float64) * 256) / 65535.0
        quats = _dq(fr['v'])
        a.T = np.zeros((nF, ntr, 3)); a.R = np.zeros((nF, ntr, 4)); a.R[..., 3] = 1; a.S = np.ones((nF, ntr, 3))
        a.has = np.zeros((ntr, 3), bool); a.keycount = np.zeros((ntr, 3), int)
        ft = np.arange(nF, dtype=np.float64)
        for typ_i in range(3):
            sel = typ == typ_i
            for j in np.unique(jid[sel]):
                s2 = sel & (jid == j); t = times[s2]; od = np.argsort(t, kind='stable'); t = t[od]
                a.has[j, typ_i] = True; a.keycount[j, typ_i] = len(t)
                if typ_i == 0:
                    q = quats[s2][od]
                    # make continuous
                    for i in range(1, len(q)):
                        if np.dot(q[i], q[i-1]) < 0: q[i] = -q[i]
                    hi = np.clip(np.searchsorted(t, ft, 'right'), 1, len(t)-1) if len(t) > 1 else np.zeros(nF, int)
                    if len(t) == 1: a.R[:, j] = q[0]
                    else:
                        lo = hi - 1; f = np.clip((ft - t[lo]) / np.maximum(t[hi] - t[lo], 1e-9), 0, 1)
                        a.R[:, j] = slerp(q[lo], q[hi], f)
                else:
                    mn, mx = (tmin, tmax) if typ_i == 1 else (smin, smax)
                    v = u[s2][od] * (mx - mn) + mn
                    out = np.stack([np.interp(ft, t, v[:, c]) for c in range(3)], 1)
                    if typ_i == 1: a.T[:, j] = out
                    else: a.S[:, j] = out
        return a
    elif sig == b'r3d2anmd':
        if ver == 5:
            size, token, f1, f2, ntr, nfr, fdur = struct.unpack_from('<IIIIIIf', data, 12)
            jho, _, _, vo, qo, fo = struct.unpack_from('<6i', data, 40)
            a.fps = 1.0 / fdur; a.nframes = nfr
            nj = (fo - jho) // 4; nv = (qo - vo) // 12; nq = (jho - qo) // 6
            a.hashes = np.frombuffer(data, '<u4', nj, jho + 12)[:ntr].copy()
            vecs = np.frombuffer(data, '<f4', nv*3, vo + 12).reshape(-1, 3).astype(np.float64)
            quats = _dq(np.frombuffer(data, 'u1', nq*6, qo + 12).reshape(-1, 6))
            fr = np.frombuffer(data, '<u2', nfr*ntr*3, fo + 12).reshape(nfr, ntr, 3)
            a.T = vecs[fr[..., 0]]; a.S = vecs[fr[..., 1]]; a.R = quats[fr[..., 2]]
            a.has = np.ones((ntr, 3), bool); return a
        elif ver == 4:
            size, token, f1, f2, ntr, nfr, fdur = struct.unpack_from('<IIIIIIf', data, 12)
            _, _, _, vo, qo, fo = struct.unpack_from('<6i', data, 40)
            a.fps = 1.0 / fdur; a.nframes = nfr
            nv = (qo - vo) // 12; nq = (fo - qo) // 16
            vecs = np.frombuffer(data, '<f4', nv*3, vo + 12).reshape(-1, 3).astype(np.float64)
            quats = np.frombuffer(data, '<f4', nq*4, qo + 12).reshape(-1, 4).astype(np.float64)
            fr = np.frombuffer(data, np.dtype([('h', '<u4'), ('t', '<u2'), ('s', '<u2'), ('r', '<u2'), ('p', '<u2')]), nfr*ntr, fo + 12).reshape(nfr, ntr)
            a.hashes = fr['h'][0].copy(); a.T = vecs[fr['t']]; a.S = vecs[fr['s']]; a.R = quats[fr['r']]
            a.has = np.ones((ntr, 3), bool); return a
        else:
            raise Exception(f'legacy anm v{ver} unsupported')
    raise Exception('bad anm ' + repr(sig))

def pose_world(skel, anim, frame):
    """World matrices (J,4,4) for skeleton at anim frame; joints with no track keep bind local."""
    lt, lr, ls = skel.lt.copy(), skel.lr.copy(), skel.ls.copy()
    hm = {int(h): i for i, h in enumerate(anim.hashes)}
    for j, h in enumerate(skel.hashes):
        t = hm.get(int(h))
        if t is None: continue
        if anim.has[t, 1] or anim.sig != 'r3d2canm': lt[j] = anim.T[frame, t]
        if anim.has[t, 0] or anim.sig != 'r3d2canm': lr[j] = anim.R[frame, t]
        if anim.has[t, 2] or anim.sig != 'r3d2canm': ls[j] = anim.S[frame, t]
    return world_from_local(skel.parents, trs(lt, lr, ls))

def skin_matrices(skel, world, ibind=None):
    return world @ (skel.ibind if ibind is None else ibind)

def deform(mesh, skel, world, infl_map=None):
    """LBS. mesh.infl indexes skel.influences (SKN influence ids -> joint ids)."""
    M = skin_matrices(skel, world)
    inf = np.array(skel.influences)[mesh.infl] if infl_map is None else infl_map[mesh.infl]
    P = np.concatenate([mesh.pos, np.ones((len(mesh.pos), 1))], 1)
    out = np.zeros((len(P), 3)); nout = np.zeros((len(P), 3))
    for k in range(4):
        Mk = M[inf[:, k]]
        out += mesh.w[:, k:k+1] * np.einsum('nij,nj->ni', Mk[:, :3, :], P)
        nout += mesh.w[:, k:k+1] * np.einsum('nij,nj->ni', Mk[:, :3, :3], mesh.nrm)
    nl = np.linalg.norm(nout, axis=1, keepdims=True); nout = nout / np.where(nl < 1e-9, 1, nl)
    return out, nout

# ---------- textures ----------
def load_image(path):
    from PIL import Image
    data = open(path, 'rb').read()
    if data[:4] == b'TEX\x00':
        w, h, unk, fmt, rtype, flags = struct.unpack_from('<HHBBBB', data, 4)
        body = data[12:]
        if fmt in (10, 11, 12):
            bs = 8 if fmt == 10 else 16
            size = max(1, (w+3)//4) * max(1, (h+3)//4) * bs
            main = body[-size:] if flags & 1 else body[:size]
            fourcc = b'DXT1' if fmt == 10 else (b'DXT3' if fmt == 11 else b'DXT5')
            hdr = b'DDS ' + struct.pack('<7I', 124, 0x1007 | 0x80000, h, w, size, 0, 1) + b'\0'*44 + struct.pack('<2I4s5I', 32, 4, fourcc, 0, 0, 0, 0, 0) + struct.pack('<5I', 0x1000, 0, 0, 0, 0)
            return Image.open(io.BytesIO(hdr + main)).convert('RGBA')
        elif fmt == 20:
            size = w*h*4; main = body[-size:] if flags & 1 else body[:size]
            arr = np.frombuffer(main, np.uint8).reshape(h, w, 4)[..., [2, 1, 0, 3]]
            return Image.fromarray(arr, 'RGBA')
        raise Exception(f'tex fmt {fmt}')
    return Image.open(io.BytesIO(data)).convert('RGBA')

# ---------- ANM v4 writer ----------
def write_anm_v4(path, hashes, T, R, S_, frame_duration):
    """hashes (K,), T (F,K,3), R (F,K,4 xyzw), S_ (F,K,3). Writes uncompressed r3d2anmd v4 with deduplicated palettes."""
    F, K = T.shape[:2]
    vec_all = np.concatenate([T.reshape(-1, 3), S_.reshape(-1, 3)]).astype('<f4')
    uv, inv = np.unique(vec_all.view([('', '<f4')]*3), return_inverse=True)
    inv = inv.reshape(-1); ti = inv[:F*K].reshape(F, K); si = inv[F*K:].reshape(F, K)
    q = R.reshape(-1, 4).astype('<f4')
    uq, qinv = np.unique(q.view([('', '<f4')]*4), return_inverse=True); ri = qinv.reshape(F, K)
    if len(uv) > 65535 or len(uq) > 65535: raise Exception(f'anm palette overflow vecs={len(uv)} quats={len(uq)}')
    vecs = uv.view('<f4').reshape(-1, 3); quats = uq.view('<f4').reshape(-1, 4)
    fr = np.zeros((F, K), np.dtype([('h', '<u4'), ('t', '<u2'), ('s', '<u2'), ('r', '<u2'), ('p', '<u2')]))
    fr['h'] = np.asarray(hashes, '<u4')[None, :]; fr['t'] = ti; fr['s'] = si; fr['r'] = ri
    vo = 64; qo = vo + vecs.nbytes; fo = qo + quats.nbytes; total = 12 + fo + fr.nbytes
    out = bytearray()
    out += b'r3d2anmd' + struct.pack('<I', 4)
    out += struct.pack('<4I', total, 0xBE0794D3, 0, 0) + struct.pack('<2I', K, F) + struct.pack('<f', frame_duration)
    out += struct.pack('<6i', 0, 0, 0, vo, qo, fo) + b'\0' * 12
    assert len(out) == 12 + vo
    out += vecs.tobytes() + quats.tobytes() + fr.tobytes()
    assert len(out) == total
    with open(path, 'wb') as f: f.write(out)
    return total
