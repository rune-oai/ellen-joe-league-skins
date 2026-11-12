"""Dump a League .bin to readable text with hash resolution. usage: bindump.py <bin> <hashdir> [out]"""
import sys, os
from pyritofile import BIN, BINType
from pyritofile.bin import BINField, BINEntry
_tables = {}
def tables(hashdir):
    if _tables: return _tables
    for nm in ('binentries', 'binfields', 'bintypes', 'binhashes'):
        d = {}
        p = os.path.join(hashdir, f'hashes.{nm}.txt')
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                h, _, v = line.rstrip('\n').partition(' ')
                d[h] = v
        _tables[nm] = d
    return _tables
def hx(v):
    if isinstance(v, int): return f'{v:08x}'
    return str(v)
def nm(t, table, v):
    s = hx(v); return t[table].get(s, '0x' + s) if len(s) == 8 else s
def dump_value(t, v, vt, ind, out):
    pad = '  ' * ind
    if isinstance(v, BINField):
        dump_field(t, v, ind, out); return
    if vt in (BINType.Hash, BINType.Link):
        s = hx(v); out.append(pad + (t['binentries'].get(s) or t['binhashes'].get(s) or '0x' + s)); return
    if hasattr(v, '__json__') and not isinstance(v, BINField):
        v = v.__json__()
    out.append(pad + repr(v))
def dump_field(t, f, ind, out):
    pad = '  ' * ind
    name = nm(t, 'binfields', f.hash) if f.hash is not None else ''
    ty = BINType(f.type).name if f.type is not None else '?'
    if f.type in (BINType.Pointer, BINType.Embed):
        cls = nm(t, 'bintypes', f.hash_type) if f.hash_type is not None else 'null'
        out.append(f'{pad}{name}: {ty} = {cls} {{')
        for sub in (f.data or []): dump_field(t, sub, ind + 1, out)
        out.append(pad + '}')
    elif f.type in (BINType.List, BINType.List2, BINType.Option):
        out.append(f'{pad}{name}: {ty}[{BINType(f.value_type).name}] = {{')
        data = f.data if isinstance(f.data, (list, tuple)) else ([] if f.data is None else [f.data])
        for sub in data: dump_value(t, sub, f.value_type, ind + 1, out)
        out.append(pad + '}')
    elif f.type == BINType.Map:
        out.append(f'{pad}{name}: Map[{BINType(f.key_type).name},{BINType(f.value_type).name}] = {{')
        for k, v in f.data:
            ks = (t['binentries'].get(hx(k)) or t['binhashes'].get(hx(k)) or '0x' + hx(k)) if f.key_type in (BINType.Hash, BINType.Link) else repr(k)
            if isinstance(v, BINField):
                out.append(f'{pad}  {ks} =>'); dump_field(t, v, ind + 2, out)
            else:
                tmp = []; dump_value(t, v, f.value_type, 0, tmp); out.append(f'{pad}  {ks} => {tmp[0] if tmp else None}')
        out.append(pad + '}')
    else:
        tmp = []; dump_value(t, f.data, f.type, 0, tmp)
        out.append(f'{pad}{name}: {ty} = {tmp[0] if tmp else None}')
def dump(path, hashdir):
    t = tables(hashdir); b = BIN(); b.read(path); out = []
    out.append(f'# signature={b.signature} version={b.version} is_patch={b.is_patch}')
    out.append('links: ' + repr(b.links))
    for e in b.entries:
        out.append(f'ENTRY {nm(t, "binentries", e.hash)} : {nm(t, "bintypes", e.type)} {{')
        for f in e.data: dump_field(t, f, 1, out)
        out.append('}')
    if b.patches: out.append(f'# patches: {len(b.patches)}')
    return '\n'.join(out), b
if __name__ == '__main__':
    txt, b = dump(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3: open(sys.argv[3], 'w').write(txt)
    else: print(txt)
    # round-trip check
    raw = open(sys.argv[1], 'rb').read()
    try:
        out = b.write('', raw=True)
        print('roundtrip identical:', out == raw, len(out), len(raw), file=sys.stderr)
    except Exception as e:
        print('roundtrip failed', repr(e), file=sys.stderr)
