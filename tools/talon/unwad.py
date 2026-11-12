import sys, os
from pyritofile import WAD, guess_extension
src, dst, hashfile = sys.argv[1], sys.argv[2], sys.argv[3]
hashes = {}
with open(hashfile, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        h, _, p = line.rstrip('\n').partition(' ')
        hashes[h] = p
wad = WAD(); wad.read(src)
print('chunks', len(wad.chunks))
unk = 0
root = os.path.realpath(dst)
with wad.stream(src, 'rb') as bs:
    for c in wad.chunks:
        c.read_data(bs)
        name = hashes.get(c.hash)
        if name is None:
            unk += 1
            ext = c.extension or (guess_extension(c.data) if c.data else None) or 'bin_unk'
            name = f'_unknown/{c.hash}.{ext}'
        out = os.path.realpath(os.path.join(root, name))
        if not out.startswith(root + os.sep):
            print('SKIP unsafe path', name); continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as o:
            o.write(c.data or b'')
        c.free_data()
print('unknown', unk)
