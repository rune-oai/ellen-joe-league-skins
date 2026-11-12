import sys, re
txt = open(sys.argv[1]).read().split('\n')
# find AnimationGraphData entries and list clip name -> type -> anm path
i = 0; cur = None
while i < len(txt):
    l = txt[i]
    if l.startswith('ENTRY'):
        cur = l if 'AnimationGraphData' in l else None
        if cur: print(cur)
    elif cur:
        m = re.match(r"^    (\S.*) =>$", l)
        if m and i + 1 < len(txt):
            m2 = re.match(r"^      : (\w+) = (\w+) \{", txt[i+1])
            if m2:
                name, cls = m.group(1), m2.group(2); j = i + 2; path = None; extra = []
                while j < len(txt) and not re.match(r"^    \S.* =>$", txt[j]) and not txt[j].startswith('  }') :
                    mm = re.search(r"mAnimationFilePath: String = '(.*)'", txt[j])
                    if mm: path = mm.group(1)
                    j += 1
                print(f'  {name:42s} {cls:28s} {path or ""}')
    i += 1
