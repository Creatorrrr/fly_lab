"""Implementation identity for strict continuation and evidence provenance."""
from pathlib import Path
from .integrity import file_hash, digest

def source_identity():
    root=Path(__file__).resolve().parents[2]
    paths=sorted(p for p in (root/'flylab').rglob('*') if p.suffix in ('.py','.metal','.cu'))
    paths+=sorted(p for p in (root/'data/releases').glob('*.json'))
    if (root/'data/circuit.json').is_file():paths.append(root/'data/circuit.json')
    return digest({str(p.relative_to(root)):file_hash(p) for p in paths})
