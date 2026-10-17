import re,zlib,sys
d=open(sys.argv[1],'rb').read()
out=[]
for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream',d,re.S):
    try: s=zlib.decompress(m.group(1))
    except Exception: continue
    t=re.findall(rb'\((.*?)\)\s*Tj|\[(.*?)\]\s*TJ',s,re.S)
    if not t: continue
    line=[]
    for a,b in t:
        if a: line.append(a)
        else: line.append(b''.join(re.findall(rb'\((.*?)\)',b)))
    out.append(b' '.join(line))
open(sys.argv[2],'wb').write(b'\n'.join(out))
