"""Synthetic-only E266 regression tests; no saved scientific evaluation."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_opponent_current as m


def fixture():
    edges = np.sort(m.SIX + [8191320])
    six = edges != 8191320
    rate = np.zeros((101, 7), np.float32)
    rate[:, six] = (2 + np.arange(101)[:, None] / 100)
    rate[:, ~six] = 20 - np.arange(101)[:, None] * .06
    w0 = np.zeros(7, np.float32)
    w7 = np.ones(7, np.float32)
    w6 = w7.copy()
    w6[~six] = 0
    pres = np.arange(7, dtype=np.int64)
    pres[~six] = 128621
    return {'control_tick': np.arange(101, dtype=np.int64), 'time_s': np.arange(101) * .005,
                'edge_ids': edges.astype(np.int64), 'pre_ids': pres, 'post_ids': np.full(7, 111970, np.int64),
                'counts': np.where(six, 1, 2).astype(np.int64), 'original_row_columns': np.arange(7, dtype=np.int64),
                'rate': rate, 'mask': np.ones((101, 7), np.float32), 'W0': w0, 'W6': w6, 'W7': w7,
                'six_mask': six, 'opponent_mask': ~six}


def test_independent_scalar_recurrence():
    a = fixture()
    arrays, report = m.compute(a)
    expected = []
    for row in range(101):
        expected.append(sum((float(a['W6'][col]) - float(a['W0'][col]))
                            * float(a['rate'][row, col]) * float(a['mask'][row, col]) for col in range(7)))
    assert np.array_equal(arrays['J6'], expected)
    bias = -sum(expected[1:100]) / 99
    assert arrays['bias64'][0] == pytest.approx(bias, abs=1e-14)
    assert arrays['bias32'][0] == np.float32(arrays['bias64'][0])
    assert report['H266_classification'] == 'supported'
    assert set(report['fixed_cases']) == {'43', '56', '70'}
    assert report['covariance_J6_Jopp'] < 0


def test_refuted_and_masks():
    a = fixture()
    a['mask'][:, ~a['six_mask']] = 0
    arrays, report = m.compute(a)
    assert np.array_equal(arrays['Jopp'], np.zeros(101))
    assert report['ratio'] == 1
    assert report['H266_original_count_balance'] is False


def test_zero_denominator():
    a = fixture()
    a['rate'][:] = 1
    _, report = m.compute(a)
    assert report['ratio'] is None
    assert report['H266_original_count_balance'] is None
    assert report['H266_classification'] == 'inconclusive'


@pytest.mark.parametrize('defect', ['nan', 'control', 'edge', 'weight', 'mask', 'columns'])
def test_invalid_alignment(defect):
    a = fixture()
    if defect == 'nan':
        a['rate'][3, 2] = np.nan
    elif defect == 'control':
        a['control_tick'][56] = 57
    elif defect == 'edge':
        a['edge_ids'][0] += 1
    elif defect == 'weight':
        a['W6'][a['opponent_mask']] = 1
    elif defect == 'mask':
        a['mask'][1, 1] = -1
    else:
        a['original_row_columns'][1] = 0
    with pytest.raises(ValueError):
        m.compute(a)


def args(tmp_path):
    return SimpleNamespace(out=str(tmp_path / 'out'), freeze='none', freeze_sha256='none',
                           review='none', review_sha256='none')


def test_gate_failure_preserves_first_error(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'gate', lambda *args: (_ for _ in ()).throw(ValueError('first failure')))
    assert m.run(args(tmp_path)) == 1
    r = m.read(tmp_path / 'out/result.json')
    assert r['analysis'] is None and r['H266_original_count_balance'] is None
    assert m.read(tmp_path / 'out/errors.json')[0]['message'] == 'first failure'
    assert (tmp_path / 'out/arrays.npz').is_file()


def test_publication_withdraws_success(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'gate', lambda *args: {'output': 'out'})
    monkeypatch.setattr(m, 'load_inputs', lambda f, a, check: a.update(fixture()))
    original = m.write
    def fail(path, value):
        if path.name == 'output-manifest.json':
            raise OSError('manifest failure')
        original(path, value)
    monkeypatch.setattr(m, 'write', fail)
    assert m.run(args(tmp_path)) == 1
    r = m.read(tmp_path / 'out/result.json')
    assert r['validity'] == 'invalid' and r['H266_original_count_balance'] is None
    assert r['analysis'] is None and r['H266_classification'] == 'not_tested'
    assert m.read(tmp_path / 'out/errors.json')[0]['stage'] == 'publication'


def test_fresh_output_only(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    (tmp_path / 'out').mkdir()
    with pytest.raises(ValueError):
        m.run(args(tmp_path))


def test_double_publication_failure_quarantines_stale_success(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'gate', lambda *args: {'output': 'out'})
    monkeypatch.setattr(m, 'load_inputs', lambda f, a, check: a.update(fixture()))
    original = m.write
    calls = {'result': 0}
    def fail(path, value):
        if path.name == 'output-manifest.json':
            # Simulate a partial successful manifest before the write fails.
            original(path, {'validity': 'valid'})
            raise OSError('first manifest failure')
        if path.name == 'result.json':
            calls['result'] += 1
            if calls['result'] >= 2:
                raise OSError('recovery result failure')
        original(path, value)
    monkeypatch.setattr(m, 'write', fail)
    assert m.run(args(tmp_path)) == 1
    out = tmp_path / 'out'
    assert not (out / 'result.json').exists()
    assert (out / 'result.json.unverified-raw').exists()
    # Recovery manifest may itself leave partial bytes; it must not claim success.
    if (out / 'output-manifest.json').exists():
        assert m.read(out / 'output-manifest.json').get('validity') != 'valid'
    assert (out / 'arrays.npz').exists()
    assert m.read(out / 'errors.json')[0]['message'] == 'first manifest failure'
