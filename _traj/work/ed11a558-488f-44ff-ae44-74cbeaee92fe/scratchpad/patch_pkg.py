import pathlib, sys
p = pathlib.Path(r"E:\work\AI\Strata\tools\hip\package_windows.py")
s = p.read_text(encoding="utf-8")
old = ('    for f in (rbin / "rocblas" / "library").iterdir():\n'
       '        m = re.search(r"gfx[0-9a-f]+", f.name)\n'
       '        if m is None or m.group() in archs:\n'
       '            shutil.copy2(f, lib / f.name)\n')
new = ('    for f in (rbin / "rocblas" / "library").iterdir():\n'
       '        m = re.search(r"gfx[0-9a-f]+", f.name)\n'
       '        if m is not None and m.group() not in archs:\n'
       '            continue\n'
       '        if f.is_dir():                                    # ROCm 10.0.0 release wheels: one folder per arch\n'
       '            shutil.copytree(f, lib / f.name)\n'
       '        else:                                             # 10.2 nightlies: the files side by side\n'
       '            shutil.copy2(f, lib / f.name)\n')
if s.count(old) != 1:
    sys.exit("pattern not found")
p.write_text(s.replace(old, new), encoding="utf-8", newline="\n")
print("patched package_windows.py")
