"""helpers for walking / editing pyritofile BIN trees"""
import os
from pyritofile import BIN, BINType
from pyritofile.bin import BINField, BINEntry
from pyritofile.ermmm import FNV1a
from xxhash import xxh64
def fh(name): return f'{FNV1a(name) & 0xFFFFFFFF:08x}'
def ph(path): return xxh64(path.lower().encode()).hexdigest()
def get(fields, name):
    h = fh(name)
    for f in fields or []:
        if f.hash == h: return f
    return None
def walk(v, vt=None):
    """yield every BINField in a tree (entry.data list, field, or raw container values)"""
    if isinstance(v, BINField):
        yield v
        if v.type in (BINType.Pointer, BINType.Embed):
            for s in v.data or []: yield from walk(s)
        elif v.type in (BINType.List, BINType.List2):
            for s in v.data or []:
                if isinstance(s, BINField): yield from walk(s)
        elif v.type == BINType.Option:
            if isinstance(v.data, BINField): yield from walk(v.data)
        elif v.type == BINType.Map:
            for k, x in v.data:
                if isinstance(k, BINField): yield from walk(k)
                if isinstance(x, BINField): yield from walk(x)
    elif isinstance(v, (list, tuple)):
        for s in v:
            if isinstance(s, BINField): yield from walk(s)
def values(f):
    """yield (type, python value) for every scalar stored in field f (not recursing into nested fields)"""
    if f.type in (BINType.List, BINType.List2):
        for s in f.data or []:
            if not isinstance(s, BINField): yield f.value_type, s
    elif f.type == BINType.Option:
        if f.data is not None and not isinstance(f.data, BINField): yield f.value_type, f.data
    elif f.type == BINType.Map:
        for k, x in f.data:
            if not isinstance(k, BINField): yield f.key_type, k
            if not isinstance(x, BINField): yield f.value_type, x
    elif f.type not in (BINType.Pointer, BINType.Embed):
        yield f.type, f.data
def strings(b):
    out = []
    for e in b.entries:
        for f in walk(e.data):
            for t, v in values(f):
                if t == BINType.String and isinstance(v, str): out.append(v)
    return out
def load_tables(hashdir, names=('binentries', 'binfields', 'bintypes', 'binhashes')):
    t = {}
    for nm in names:
        d = {}
        with open(os.path.join(hashdir, f'hashes.{nm}.txt'), 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                h, _, v = line.rstrip('\n').partition(' '); d[h] = v
        t[nm] = d
    return t
