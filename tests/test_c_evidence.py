import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from flylab.c.evidence import verify_archive,safe_name
from flylab.c.integrity import digest
import hashlib

class EvidenceTests(unittest.TestCase):
    def test_corruption_and_missing_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);content=b'actual raw trace\n'
            manifest=dict(schema='flylab.evidence.v1',files=[dict(path='trace.jsonl',bytes=len(content),sha256=hashlib.sha256(content).hexdigest())])
            for variant,value in [('valid',content),('corrupt',b'fake'),('missing',None)]:
                path=p/(variant+'.zip')
                with zipfile.ZipFile(path,'w') as z:
                    z.writestr('EVIDENCE_MANIFEST.json',json.dumps(manifest))
                    if value is not None:z.writestr('trace.jsonl',value)
                if variant=='valid':self.assertEqual(verify_archive(path)['status'],'PASS')
                else:
                    with self.assertRaises(ValueError):verify_archive(path)
    def test_archive_names_cannot_escape_root(self):
        for name in ('../outside','/absolute','a/../../outside','a\\b'):
            with self.assertRaises(ValueError):safe_name(name)
if __name__=='__main__':unittest.main()
