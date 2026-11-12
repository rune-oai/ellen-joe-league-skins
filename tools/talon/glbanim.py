"""minimal glTF-binary reader for skeleton animations (data only: nothing in the file is executed)"""
import json, struct, numpy as np
CT = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}; NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
class Glb:
    def __init__(self, path):
        b = open(path, 'rb').read(); assert b[:4] == b'glTF'; jl = struct.unpack('<I', b[12:16])[0]; self.g = json.loads(b[20:20 + jl])
        o = 20 + jl; bl = struct.unpack('<I', b[o:o + 4])[0]; assert b[o + 4:o + 8] == b'BIN\x00'; self.bin = b[o + 8:o + 8 + bl]
        g = self.g; self.nodes = g['nodes']; self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get('children', []): self.parent[c] = i
        self.joints = g['skins'][0]['joints']; self.names = [self.nodes[j]['name'] for j in self.joints]
        self.ibm = self.acc(g['skins'][0]['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.anims = {a['name']: a for a in g['animations']}
    def acc(self, i):
        a = self.g['accessors'][i]; v = self.g['bufferViews'][a['bufferView']]; n = NC[a['type']]; dt = np.dtype(CT[a['componentType']])
        off = v.get('byteOffset', 0) + a.get('byteOffset', 0); st = v.get('byteStride', 0)
        if st and st != dt.itemsize * n:
            raw = np.frombuffer(self.bin, np.uint8, st * a['count'], off).reshape(a['count'], st)[:, :dt.itemsize * n]; arr = np.frombuffer(raw.tobytes(), dt)
        else: arr = np.frombuffer(self.bin, dt, a['count'] * n, off)
        arr = arr.reshape(a['count'], n).astype(float)
        if a.get('normalized'): arr = arr / {'i1': 127.0, 'u1': 255.0, '<i2': 32767.0, '<u2': 65535.0}[CT[a['componentType']]]
        return arr
    def rest(self, node):
        n = self.nodes[node]; assert 'matrix' not in n
        return np.array(n.get('translation', [0, 0, 0]), float), np.array(n.get('rotation', [0, 0, 0, 1]), float), np.array(n.get('scale', [1, 1, 1]), float)
    def duration(self, name): return max(float(self.acc(s['input']).max()) for s in self.anims[name]['samplers'])
    def sample(self, name, times):
        """local T, R (xyzw), S per joint name at the given times (linear / slerp between the file's keys)"""
        a = self.anims[name]; out = {}
        for nm, j in zip(self.names, self.joints): t, r, s = self.rest(j); out[nm] = [np.repeat(t[None], len(times), 0), np.repeat(r[None], len(times), 0), np.repeat(s[None], len(times), 0)]
        node2name = {j: nm for nm, j in zip(self.names, self.joints)}
        for ch in a['channels']:
            nm = node2name.get(ch['target'].get('node'))
            if nm is None: continue
            sm = a['samplers'][ch['sampler']]; assert sm.get('interpolation', 'LINEAR') in ('LINEAR', 'STEP'), sm.get('interpolation')
            tin = self.acc(sm['input'])[:, 0]; val = self.acc(sm['output']); p = ch['target']['path']
            i1 = np.clip(np.searchsorted(tin, times, 'right'), 1, len(tin) - 1) if len(tin) > 1 else np.zeros(len(times), int); i0 = np.maximum(i1 - 1, 0)
            den = np.maximum(tin[i1] - tin[i0], 1e-9); x = np.clip((times - tin[i0]) / den, 0, 1) if len(tin) > 1 else np.zeros(len(times))
            if sm.get('interpolation') == 'STEP': x = np.floor(x)
            if p == 'rotation':
                q0 = val[i0]; q1 = val[i1].copy(); d = (q0 * q1).sum(1); q1[d < 0] *= -1; d = np.abs(d).clip(0, 1); th = np.arccos(d); sn = np.sin(th)
                w0 = np.where(sn > 1e-6, np.sin((1 - x) * th) / np.maximum(sn, 1e-9), 1 - x); w1 = np.where(sn > 1e-6, np.sin(x * th) / np.maximum(sn, 1e-9), x)
                q = w0[:, None] * q0 + w1[:, None] * q1; out[nm][1] = q / np.linalg.norm(q, axis=1, keepdims=True)
            elif p in ('translation', 'scale'): out[nm][0 if p == 'translation' else 2] = (1 - x[:, None]) * val[i0] + x[:, None] * val[i1]
        return out
