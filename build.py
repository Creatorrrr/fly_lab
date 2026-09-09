#!/usr/bin/env python3
"""Build the browser VIEWER only. B physics still requires the Python server."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT/'src/template.html').read_text()
read=lambda rel:(ROOT/rel).read_text()
core='\n'.join(read('src/core/'+n+'.js') for n in ('math','connectome'))
app='\n'.join(read(p) for p in ('src/runtime/b-transport.js','src/ui/world-view.js','src/ui/monitor.js','src/ui/app.js'))
s=s.replace('/*STYLE*/',read('src/style.css')).replace('/*DATA*/',read('data/circuit.json').replace('</','<\\/')).replace('/*WORKER*/','').replace('/*CORE*/',core).replace('/*APP*/',app)
(ROOT/'FLY_LAB_B.html').write_text(s)
print('Built',ROOT/'FLY_LAB_B.html')
