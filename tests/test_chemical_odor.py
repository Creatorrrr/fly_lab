import copy
from types import SimpleNamespace
import unittest
import numpy as np
from flylab.chemical_odor import ChemicalOdor
from flylab.c.ports import PortBindings,SensoryEncoder
from tests.c_fixtures import graph_fixture,bindings_fixture


class NamedOdorTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(schema='flylab.chemical-odor.v1',evidence=['synthetic response table'],
            odorants=[dict(inchikey='excitatory'),dict(inchikey='inhibitory')],
            receptors=[dict(receptor='OrA',site='palp',baseline=.2,responses=dict(excitatory=.8,inhibitory=0.)),
                       dict(receptor='OrB',site='antenna',baseline=.1,responses=dict(excitatory=.3))])
        self.field=SimpleNamespace(model=dict(half_concentration=1.),observe=lambda b,w:
            dict(concentration=np.full((4,2),.5*len(w['sources']))))
        self.body=SimpleNamespace(physics_time=lambda:0.)

    def test_excitation_inhibition_missing_measurements_and_unlabelled_sources(self):
        chemical=ChemicalOdor(self.spec,self.field)
        sample=chemical.observe(self.body,dict(sources=[dict(id='x',odorant='excitatory'),dict(id='generic')]))
        self.assertAlmostEqual(sample['features']['OrA:palp_left'],.3)
        self.assertEqual(sample['unlabelled_sources'],['generic'])
        sample=chemical.observe(self.body,dict(sources=[dict(id='x',odorant='inhibitory')]))
        self.assertAlmostEqual(sample['features']['OrA:palp_right'],-.1)
        self.assertEqual(sample['unmeasured_pairs'],[dict(receptor='OrB',odorant='inhibitory')])
        with self.assertRaisesRegex(ValueError,'absent'):
            chemical.observe(self.body,dict(sources=[dict(id='unknown',odorant='not-in-table')]))

    def test_chemical_port_site_and_finite_values_are_validated(self):
        graph=graph_fixture();base=copy.deepcopy(bindings_fixture(graph).spec)
        base['sensor_model']=dict(kind='four-site-odor-v1',chemical_odor=self.spec)
        base['sensory'][0].update(channel='chemical_odor',receptor='OrA',site='palp_left')
        PortBindings(graph,base)
        base['sensory'][0]['site']='antenna_left'
        with self.assertRaises(ValueError):PortBindings(graph,base)
        for invalid in (float('nan'),-.1,1.1):
            bad=copy.deepcopy(self.spec);bad['receptors'][0]['responses']['excitatory']=invalid
            with self.assertRaises(ValueError):ChemicalOdor(bad,self.field)


if __name__=='__main__':unittest.main()
