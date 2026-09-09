import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from flylab.c.graph import GraphStore, external_id
from flylab.c.integrity import write_json
from flylab.c.data_identity import reference_for, verify_snapshot, validation_status
from tests.c_fixtures import graph_fixture, bindings_fixture


class IdentityTests(unittest.TestCase):
    def test_tiny_cli_cannot_claim_full_snapshot_but_loaded_scope_can_compute(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);g=graph_fixture();g.save(path/'graph');write_json(path/'bindings.json',bindings_fixture(g).spec)
            for scope,code in [('full-snapshot',1),('loaded-graph',0)]:
                r=subprocess.run([sys.executable,'tools/verify_c.py','--graph',str(path/'graph'),
                    '--bindings',str(path/'bindings.json'),'--out',str(path/scope),'--scope',scope],capture_output=True,text=True)
                self.assertEqual(r.returncode,code,r.stdout+r.stderr)
                import json
                report=json.loads((path/scope/'report.json').read_text())
                self.assertEqual(report['gates']['loaded_graph_compute']['status'],'PASS')
                self.assertEqual(report['gates']['full_snapshot_compute']['status'],'FAIL')

    def test_self_declared_full_tiny_graph_is_rejected_by_pinned_source(self):
        g=graph_fixture();g.manifest.update(scope='full_snapshot',raw_file_hashes={'neurons.csv.gz':'invented'})
        self.assertEqual(verify_snapshot(g)['status'],'FAIL')

    def test_roster_and_wiring_must_both_match_independent_reference(self):
        g=graph_fixture();g.manifest.update(scope='full_snapshot',raw_file_hashes={'test':'fixed'})
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'reference.json';write_json(p,reference_for(g))
            self.assertEqual(verify_snapshot(g,p)['status'],'PASS')
            other=graph_fixture(pre=(0,2,1));other.manifest.update(scope='full_snapshot',raw_file_hashes={'test':'fixed'})
            self.assertEqual(verify_snapshot(other,p)['status'],'FAIL')
            other=graph_fixture();other.manifest.update(scope='full_snapshot',raw_file_hashes={'test':'fixed'})
            other.nodes[0]=dict(other.nodes[0],id=external_id('999'),root_id='999')
            self.assertEqual(verify_snapshot(other,p)['status'],'FAIL')

    def test_requested_blocked_gate_is_not_success(self):
        g={'compute':dict(status='PASS'),'cuda':dict(status='BLOCKED')}
        self.assertEqual(validation_status(g,['compute','cuda'])['exit_code'],2)
        self.assertEqual(validation_status(g,['compute'])['exit_code'],0)

    def test_content_hash_ignores_acquisition_clock_but_changes_on_annotation(self):
        g=graph_fixture()
        def build(release,nodes):
            return GraphStore.from_edges(nodes,[0],[1],[4],metadata=dict(graph_hash_schema='content-v2',annotation_release=release))
        self.assertEqual(build('yesterday',g.nodes).hash,build('today',g.nodes).hash)
        nodes=copy.deepcopy(g.nodes);nodes[0]['cell_type']='different'
        self.assertNotEqual(build('today',g.nodes).hash,build('today',nodes).hash)

    def test_namespaces_do_not_mix_specimens(self):
        self.assertNotEqual(external_id('123'),external_id('123','banc','888'))
        with self.assertRaises(ValueError): external_id('123','../other','888')


if __name__=='__main__':unittest.main()
