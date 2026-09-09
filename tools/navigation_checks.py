"""Versioned, pure trajectory checks used both online and on recorded evidence.

No simulator, no dataset-specific thresholds. Millimetres/radians/model seconds.
Enabled motion is an explicit experiment contract: a zero drive is NOT inferred
as intentional stopping, because a stuck controller may output zero forever.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class NavigationPolicy:
    version: str = 'flylab.navigation-checks.v2'
    stuck_seconds: float = 3.0
    envelope_mm: float = 1.0
    max_sample_gap_seconds: float = .15
    resume_seconds: float = 2.0
    resume_net_mm: float = 1.0
    resume_forward_mm: float = 1.0

    def __post_init__(self):
        for k,v in asdict(self).items():
            if k != 'version' and (isinstance(v,bool) or not math.isfinite(v) or v<=0):
                raise ValueError('Positive finite policy parameter required: '+k)


def _number(x,name):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
        raise ValueError('Finite trajectory value required: '+name)
    return float(x)


def _sample(row):
    """One normalized row. Callers must label any reconstructed fields explicitly."""
    if not isinstance(row,dict): raise ValueError('Trajectory row must be a dict')
    r=dict(row)
    r['t']=_number(r['t'],'t')
    if r['t']<0: raise ValueError('Negative model time')
    if not isinstance(r.get('p'),(list,tuple)) or len(r['p'])!=3: raise ValueError('XYZ required')
    r['p']=[_number(x,'p') for x in r['p']]
    r['yaw']=_number(r['yaw'],'yaw')
    for k in ('avoiding','recovering','motorCoupled','motionExpected'):
        if type(r.get(k)) is not bool: raise ValueError('Explicit boolean required: '+k)
    return r


def _distance(a,b):
    return math.hypot(a['p'][0]-b['p'][0],a['p'][2]-b['p'][2])


def _span(rows):
    return math.hypot(max(r['p'][0] for r in rows)-min(r['p'][0] for r in rows),
                      max(r['p'][2] for r in rows)-min(r['p'][2] for r in rows))


class NavigationMonitor:
    """Time-based rolling stationary and measured walking-resumption checks.

    A motor-off or explicitly unexpected-motion period resets the rolling window.
    Missing time samples make the record INCOMPLETE, not silently PASS. Avoidance,
    recovery, contact and command values describe a failure; none gates it away.
    """
    def __init__(self,policy=None):
        self.policy=policy or NavigationPolicy()
        self.rows=deque();self.previous=None;self.samples=0;self.windows=0
        self.min_span=None;self.stationary=[];self.sampling_gaps=[]
        self.disabled_samples=0;self.resumptions=[];self.pending=None

    @staticmethod
    def _enabled(r): return r['motorCoupled'] and r['motionExpected']

    @staticmethod
    def _avoiding(r): return r['avoiding'] or r['recovering']

    def _close_resume(self,status,t):
        if self.pending is not None:
            self.pending.update(status=status,endTime=t)
            self.resumptions.append(self.pending)
            self.pending=None

    def add(self,row):
        r=_sample(row);p=self.policy;prev=self.previous;self.samples+=1
        if prev is not None and r['t']<=prev['t']: raise ValueError('Times must strictly increase')
        gap=prev is not None and r['t']-prev['t']>p.max_sample_gap_seconds+1e-9
        if gap:
            self.sampling_gaps.append(dict(start=prev['t'],end=r['t']))
            self.rows.clear();self._close_resume('INCOMPLETE_SAMPLING',r['t'])
        enabled=self._enabled(r)
        if not enabled:
            self.disabled_samples+=1;self.rows.clear();self._close_resume('CENSORED_MOTOR_OFF',r['t'])
        else:
            self.rows.append(r)
            # Retain the left endpoint at or immediately before t-window.
            while len(self.rows)>1 and self.rows[1]['t']<=r['t']-p.stuck_seconds+1e-9:
                self.rows.popleft()
            if r['t']-self.rows[0]['t']>=p.stuck_seconds-1e-9:
                self.windows+=1;span=_span(self.rows)
                self.min_span=span if self.min_span is None else min(self.min_span,span)
                if span<p.envelope_mm:
                    sweep=sum(abs(math.atan2(math.sin(b['yaw']-a['yaw']),math.cos(b['yaw']-a['yaw'])))
                              for a,b in zip(self.rows,list(self.rows)[1:]))
                    self.stationary.append(dict(startTime=self.rows[0]['t'],endTime=r['t'],
                        spanMM=span,netMM=_distance(self.rows[0],r),yawSweepRad=sweep,
                        avoidingSamples=sum(x['avoiding'] for x in self.rows),
                        recoveringSamples=sum(x['recovering'] for x in self.rows),
                        samples=len(self.rows)))
            # A flag transition is a candidate, not evidence of resumed walking.
            if (not gap and prev is not None and self._enabled(prev)
                    and self._avoiding(prev) and not self._avoiding(r)):
                self._close_resume('INTERRUPTED',r['t'])
                self.pending=dict(startTime=r['t'],startPosition=r['p'][:],
                                  forwardMM=0.,maxNetMM=0.,status='PENDING')
            elif self.pending is not None:
                elapsed=r['t']-self.pending['startTime']
                if self._avoiding(r): self._close_resume('INTERRUPTED_BY_AVOIDANCE',r['t'])
                elif elapsed>p.resume_seconds+1e-9: self._close_resume('FAILED_NO_RESUMPTION',r['t'])
                else:
                    # Signed body-forward projection; a pure reverse retreat does
                    # not satisfy walking resumed. Also require NET displacement.
                    dx=r['p'][0]-prev['p'][0];dz=r['p'][2]-prev['p'][2]
                    self.pending['forwardMM']+=dx*math.cos(prev['yaw'])+dz*math.sin(prev['yaw'])
                    net=math.hypot(r['p'][0]-self.pending['startPosition'][0],r['p'][2]-self.pending['startPosition'][2])
                    self.pending['maxNetMM']=max(self.pending['maxNetMM'],net)
                    if net>=p.resume_net_mm and self.pending['forwardMM']>=p.resume_forward_mm:
                        self._close_resume('PASS',r['t'])
        self.previous=r

    def result(self):
        episodes=[dict(x) for x in self.resumptions]
        if self.pending is not None: episodes.append({**self.pending,'status':'CENSORED_END_OF_RECORD'})
        status=('FAIL' if self.stationary else 'INCOMPLETE' if self.sampling_gaps
                else 'PASS' if self.windows else 'NOT_EVALUATED')
        return dict(policy=asdict(self.policy),status=status,samples=self.samples,
                    evaluatedWindows=self.windows,disabledSamples=self.disabled_samples,
                    minimumWindowSpanMM=self.min_span,stationaryWindows=self.stationary,
                    samplingGaps=self.sampling_gaps,resumptionEpisodes=episodes,
                    walkingResumed=any(x['status']=='PASS' for x in episodes))


def audit_trajectory(rows,policy=None):
    monitor=NavigationMonitor(policy)
    for row in rows: monitor.add(row)
    return monitor.result()
