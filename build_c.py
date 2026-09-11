#!/usr/bin/env python3
"""Build the C viewer; retain B's world renderer and build outputs."""
from pathlib import Path
from flylab.c import VERSION
ROOT = Path(__file__).resolve().parent
read = lambda p: (ROOT/p).read_text(encoding='utf-8')
html = read('src/c/template.html')
html = html.replace('/*VERSION*/', VERSION)
html = html.replace('/*STYLE*/', read('src/c/style.css'))
world_renderer = read('src/ui/world-view.js').replace('D. melanogaster · B', 'D. melanogaster · C')
html = html.replace('/*CORE*/', read('src/core/math.js') + '\n' + world_renderer)
html = html.replace('/*APP*/', '\n'.join(read(p) for p in ('src/c/signals.js', 'src/c/playback.js', 'src/c/workbench.js', 'src/c/app.js')))
(ROOT/'FLY_LAB_C.html').write_text(html,encoding='utf-8')
print('Built', ROOT/'FLY_LAB_C.html')
