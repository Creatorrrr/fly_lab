"""v3 binary signals: little-endian length-prefixed JSON header + raw arrays.

Layout: b'FLC3', uint32 header_bytes, UTF-8 JSON, planar float32 voltage/rate,
then packed spike events (uint64 tick, uint32 subscription-local index).
"""
import json
import struct
import numpy as np
from .integrity import canonical, bounded_int

EVENT_DTYPE = np.dtype([('tick', '<u8'), ('channel', '<u4')])


def signal_frame(engine):
    engine.sequence += 1
    indices = engine.subscription
    ids = [engine.graph.nodes[int(i)]['id'] for i in indices]
    values = engine.neural.readout(indices) if engine.neural else dict(voltage_mV=np.empty(0), rate_Hz=np.empty(0))
    if not engine.neural:
        ids = []; indices = np.empty(0, dtype=np.int32)
    n = len(ids)
    voltage = np.asarray(values['voltage_mV'], dtype='<f4')
    rates = np.asarray(values['rate_Hz'], dtype='<f4')
    lookup = {int(i): j for j, i in enumerate(indices)}
    selected = [e for e in engine.selected_events if e['index'] in lookup]
    spikes = np.asarray([(e['tick'], lookup[e['index']]) for e in selected], dtype=EVENT_DTYPE)
    payload = voltage.tobytes() + rates.tobytes() + spikes.tobytes()
    # The selected events may span several control steps in one RPC response.
    start = engine.signal_start_tick
    header = dict(schema='flylab.signals.v3', sequence=engine.sequence,
                  subscription_epoch=engine.subscription_epoch, start_tick=start, end_tick=engine.tick,
                  neural_dt=engine.parameters.dt, dtype='<f4', channel_count=n, ids=ids,
                  unit_descriptor=['mV', 'Hz'], voltage_offset=0, rate_offset=n*4,
                  spike_offset=n*8, spike_count=len(spikes), spike_dtype='uint64_tick,uint32_channel',
                  payload_bytes=len(payload))
    encoded = canonical(header)
    return b'FLC3' + struct.pack('<I', len(encoded)) + encoded + payload


def decode_signals(data):
    if len(data) < 8 or data[:4] != b'FLC3': raise ValueError('Invalid signal frame magic')
    length = struct.unpack('<I', data[4:8])[0]
    if length > 1_000_000 or 8+length > len(data): raise ValueError('Invalid signal header length')
    h = json.loads(data[8:8+length])
    bounded_int(h.get('sequence'), 'sequence', 1)
    bounded_int(h.get('subscription_epoch'), 'subscription_epoch', 1)
    start = bounded_int(h.get('start_tick'), 'start_tick')
    bounded_int(h.get('end_tick'), 'end_tick', start)
    n = bounded_int(h.get('channel_count'), 'channel_count', 0, 512)
    count = bounded_int(h.get('spike_count'), 'spike_count', 0, 10_000_000)
    payload = data[8+length:]
    if h.get('schema') != 'flylab.signals.v3' or h.get('dtype') != '<f4' or h.get('unit_descriptor') != ['mV', 'Hz']:
        raise ValueError('Unknown signal schema/units')
    if len(payload) != n*8 + count*12 or h.get('payload_bytes') != len(payload): raise ValueError('Truncated signal payload')
    if h.get('voltage_offset') != 0 or h.get('rate_offset') != n*4 or h.get('spike_offset') != n*8:
        raise ValueError('Signal offset mismatch')
    if not isinstance(h.get('ids'), list) or len(h['ids']) != n or any(not isinstance(i, str) for i in h['ids']):
        raise ValueError('Signal ID mismatch')
    voltage = np.frombuffer(payload, dtype='<f4', count=n).copy()
    rate = np.frombuffer(payload, dtype='<f4', count=n, offset=n*4).copy()
    spikes = np.frombuffer(payload, dtype=EVENT_DTYPE, count=count, offset=n*8).copy()
    if not np.isfinite(voltage).all() or not np.isfinite(rate).all(): raise ValueError('Non-finite signals')
    if count and (spikes['channel'].max() >= n or spikes['tick'].max() > h['end_tick'] or spikes['tick'].min() < h['start_tick']):
        raise ValueError('Invalid spike channel/timestamp')
    return h, voltage, rate, spikes
