"""Portable evidence bundles with independently verifiable member hashes."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile
from .integrity import file_hash, canonical, write_json

def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or not p.parts:
        raise ValueError('Unsafe evidence member')
    return p

def verify_archive(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(set(names))!=len(names):raise ValueError('Duplicate evidence member')
        for name in names:safe_name(name)
        manifest=json.loads(z.read('EVIDENCE_MANIFEST.json'))
        if manifest.get('schema')!='flylab.evidence.v1':raise ValueError('Unknown evidence schema')
        expected={r['path'] for r in manifest['files']}
        if len(expected)!=len(manifest['files']) or set(names)!=expected|{'EVIDENCE_MANIFEST.json'}:
            raise ValueError('Evidence member list mismatch')
        for row in manifest['files']:
            h=hashlib.sha256();size=0
            with z.open(row['path']) as stream:
                for block in iter(lambda:stream.read(1024**2),b''):h.update(block);size+=len(block)
            if size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Evidence hash mismatch: '+row['path'])
    return dict(status='PASS',files=len(manifest['files']),bytes=sum(r['bytes'] for r in manifest['files']),
                archive_sha256=file_hash(path),manifest=manifest)

def package(root,out,paths,*,include_source=True):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('Evidence output is immutable; choose a new name')
    selected=set()
    if include_source:
        listing=subprocess.check_output(['git','ls-files','-c','-o','--exclude-standard','-z'],cwd=root)
        for name in listing.decode().split('\0'):
            if name and (root/name).is_file():selected.add(root/name)
    for source in paths:
        source=Path(source).resolve()
        source.relative_to(root)
        if not source.exists():raise ValueError('Missing evidence: '+str(source))
        candidates=[source] if source.is_file() else source.rglob('*')
        for file in candidates:
            if file.is_symlink():raise ValueError('Evidence symlinks are not accepted')
            if file.is_file():selected.add(file)
    if out in selected:raise ValueError('Archive cannot include itself')
    rows=[]
    for file in sorted(selected):
        if file.is_symlink():raise ValueError('Source symlinks are not accepted')
        name=file.relative_to(root).as_posix();safe_name(name)
        rows.append(dict(path=name,bytes=file.stat().st_size,sha256=file_hash(file)))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    manifest=dict(schema='flylab.evidence.v1',base_commit=commit,files=rows,
                  source_state='working source including uncommitted files; per-file SHA256 is authoritative',
                  evidence_interpretation='File integrity only. Read individual technical, task and biological gates.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for row in rows:z.write(root/row['path'],row['path'])
        z.writestr('EVIDENCE_MANIFEST.json',canonical(manifest))
    result=verify_archive(out)
    index={k:v for k,v in result.items() if k!='manifest'}
    index.update(schema='flylab.evidence-index.v1',archive=out.name,base_commit=commit,
                 distribution='local release artifact; no external hosting claimed')
    write_json(out.with_suffix(out.suffix+'.index.json'),index)
    return index
