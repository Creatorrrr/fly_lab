#!/usr/bin/env python3
"""Recompute checks from a supplied wall_fix_20260909 evidence tree.

This NEVER steps MuJoCo. 'RECORDED' results are not new native executions.
Source failures, retries, and original JSON files are never rewritten.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.navigation_checks import audit_trajectory
from tools.run_physics_campaign import compare_results
from flylab.brain import Circuit,NeuralController


def read(p):
    def bad(v): raise ValueError('Nonfinite JSON token '+v)
    return json.loads(p.read_text(encoding='utf-8'),parse_constant=bad)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def read_table(p,columns):
    with p.open(encoding='utf-8-sig',newline='') as f:
        it=csv.reader(f);header=next(it);assert len(header)==columns,(p,len(header))
        rows=[]
        for row in it:
            assert len(row)==columns,(p,len(row))
            values=[float(x) for x in row];assert all(math.isfinite(x) for x in values),p
            rows.append(values)
    assert all(b[0]>a[0] for a,b in zip(rows,rows[1:])),p
    return header,rows

def brain_check(cp):
    b=NeuralController(Circuit(cp['graph']),cp['seed']);b.restore(cp['brain'])
    assert b.snapshot()==cp['brain']
    assert abs(cp['tick']*cp['controlDt']-cp['brain']['clock'])<1e-5
    assert all(math.isfinite(v) for v in cp['body']['state'])


def reevaluate(evidence:Path,source:Path):
    result=dict(schema='flylab.evidence-audit.fix2.v1',mode='RECORDED_DATA_RECOMPUTATION',
                newNativePhysicsExecuted=False,sourceRoot=str(source),evidenceRoot=str(evidence),checks=[])
    def check(name,ok,detail=None):
        result['checks'].append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+': '+str(detail))
    m=read(evidence/'source_manifest.json')
    # Use the baseline checksum list retained alongside fix_2, not the new list.
    sums=source/'docs/FIX1_CHECKSUMS.sha256'
    if not sums.exists():sums=source/'CHECKSUMS.sha256'
    check('recorded source checksum-list hash matches fix_1',sha(sums)==m['sourceChecksumsSHA256'])
    archive=source/'verification/source_inputs/FLY_LAB_B_fix_1.zip'
    check('preserved fix_1 archive matches reviewed input',archive.exists() and sha(archive)=='2931abb6047fee907c09522420c855e1ed60e11d4fa844159d1088abd3a70a18')
    verified=[];missing=[]
    with zipfile.ZipFile(archive) as z:
        for line in sums.read_text().splitlines():
            digest,name=line.split(None,1);name=name.lstrip('*')
            if name not in z.namelist():missing.append(name);continue
            check('original source checksum '+name,hashlib.sha256(z.read(name)).hexdigest()==digest)
            verified.append(name)
    check('only known original packaging omission',missing==['.gitignore'])
    result['originalSourceFiles']=dict(verified=verified,missing=missing,
        note='The original omitted .gitignore remains omitted in the preserved original ZIP. fix_2 has its own new .gitignore.')
    result['provenance']=dict(sourceManifest=m,sourceChecksumListSHA256=sha(sums),
        limitation='Final source manifest is not a per-run signed attestation; uploaded records are reviewed, not independently recreated.')
    gate=read(evidence/'campaign/gate.json');result['recordedGate']=gate
    check('recorded native gate has 14 successful checks',gate['status']=='PASS' and gate['physicalExecuted'] and len(gate['checks'])==14 and all(x['passed'] for x in gate['checks']))
    final=read(evidence/'campaign_final_report.json');runs=[];cases=[];ncp=0
    check('33 unique final cases and result paths',len(final['cases'])==len(final['resultFiles'])==33 and len({(r['seed'],r['case']) for r in final['cases']})==33)
    for name,summary in zip(final['resultFiles'],final['cases']):
        rel=Path(name);path=evidence.parent.parent/rel
        if not path.resolve().is_relative_to(evidence.resolve()):raise ValueError('Result path outside evidence root')
        d=read(path);runs.append(d);folder=path.parent
        check('individual result matches summary '+str((d['seed'],d['case'])),{k:v for k,v in d.items() if k!='trajectory'}==summary)
        initial=read(folder/'initial_checkpoint.json');end=read(folder/'final_checkpoint.json');replay=read(folder/'replay.json')
        for cp in [initial,end]:brain_check(cp);ncp+=1
        h=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
        assert h==d['initialStateSHA256'];assert replay['initial']==initial
        assert end['tick']==replay['endTick']==2000 and initial['tick']==0
        assert d['physicalExecuted'] and d['physicalAttempted'] and d['status']=='PASS'
        assert all(c['passed'] for c in d['checks']) and len(d['trajectory'])==100
        nh,nr=read_table(folder/'neural.csv',177);ph,pr=read_table(folder/'physics.csv',135)
        assert len(nr)==len(pr)==101 and nr[0][0]==0 and nr[-1][0]==10
        for r,n,q in zip(d['trajectory'],nr[1:],pr[1:]):
            assert abs(r['time']-n[0])<1e-10 and abs(r['time']-q[0])<1e-10
            assert max(abs(r['position'][i]-n[j]) for i,j in [(0,1),(1,2),(2,3)])<1e-10
            assert max(abs(a-b) for a,b in zip(r['descending'],q[-2:]))<1e-10
        assert all(0<=a<=1 for n in nr for a in n[7:])
        motor_off=d['case']=='motor-off'
        if motor_off:assert all(q[-1]==q[-2]==0 for q in pr if q[0]>3)
        if d['case']=='left-suppression':
            ix=[i for i,x in enumerate(nh) if x.startswith('model:PFL3-L:')];assert len(ix)==16
            assert all(n[i]==0 for n in nr if n[0]>3 for i in ix)
        if d['case']!='baseline':
            assert d['interventionAppliedAt']==3 and len(replay['commands'])==1 and replay['commands'][0]['tick']==600
        normalized=[dict(t=r['time'],p=r['position'],yaw=r['yaw'],avoiding=False,recovering=False,
            motorCoupled=not(motor_off and r['time']>3+1e-9),motionExpected=True) for r in d['trajectory']]
        audit=audit_trajectory(normalized)
        # Old campaign records do not have avoidance/recovery flags. They are
        # irrelevant to the stationary rule, but DO NOT infer a resume verdict.
        audit['walkingResumed']=None;audit['resumptionEpisodes']=[]
        audit['fieldProvenance']=dict(motorCoupled='from case, onset and replay configuration',
            avoidanceFlags='not recorded; unused placeholders for stationary-only analysis',
            motionExpected='analysis question: active-motion stationary audit; motor-off excluded')
        cases.append(dict(seed=d['seed'],case=d['case'],resultFile=name,status=d['status'],
            neuralRows=len(nr),physicalRows=len(pr),initialHashVerified=True,csvTrajectoryConsistent=True,
            stationaryAudit=audit,interventionDetails=d.get('interventionDetails'),metrics=d['metrics']))
    comparisons=compare_results(runs)
    check('30 comparisons independently recomputed',comparisons==final['comparisons'] and len(comparisons)==30 and all(x['status']=='PASS' for x in comparisons))
    old=read(evidence/'campaign/report.json');failed=read(evidence/'campaign/seed-7_obstacle/result.json')
    check('first obstacle failure explicitly retained',old['status']=='FAIL' and failed['status']=='FAIL' and final['retry']['seed']==7)
    check('30 enabled-motion campaign traces without stationary windows',all(c['stationaryAudit']['status']=='PASS' for c in cases if c['case']!='motor-off'))
    result['campaign']=dict(uniqueConditions=33,originalAttempts=33,retries=1,totalAttempts=34,
        originalStatus=old['status'],finalRecordedStatus=final['status'],firstFailureReason=failed.get('reason'),
        cases=cases,comparisons=comparisons,postInterventionMinimumSeparationMM=min(c['maxPostOnsetSeparationMM'] for c in comparisons),
        postInterventionMaximumSeparationMM=max(c['maxPostOnsetSeparationMM'] for c in comparisons))
    navigation=[]
    for path in sorted((evidence/'navigation').glob('*/result.json')):
        d=read(path);norm=[{**r,'recovering':False,'motorCoupled':True,'motionExpected':True} for r in d['trajectory']]
        audit=audit_trajectory(norm)
        audit['fieldProvenance']=dict(avoiding='recorded avoiding flag',recovering='not recorded; false placeholder. Source recovery implies avoiding.',motorCoupled='trial protocol and final checkpoint both enabled',motionExpected='trial protocol: continuous navigation')
        cp=read(path.parent/'final_checkpoint.json');assert cp['config']['motorCoupled'];brain_check(cp);ncp+=1
        assert d['status']=='PASS' and all(x['passed'] for x in d['checks'])
        check('strict navigation '+path.parent.name,audit['status']=='PASS' and audit['walkingResumed'])
        navigation.append(dict(case=d['case'],seed=d['seed'],recordedMetrics=d['metrics'],audit=audit))
    result['navigation']=navigation
    candidate=read(evidence/'navigation_candidate1/explore-19/result.json')
    c=audit_trajectory([{**r,'recovering':False,'motorCoupled':True,'motionExpected':True} for r in candidate['trajectory']])
    check('failed candidate remains a failure with strict rule',candidate['status']=='FAIL' and c['status']=='FAIL')
    result['failedCandidate']=dict(case='explore-19',originalStatus=candidate['status'],audit=c)
    frame=read(evidence/'browser_frame.json');history=read(evidence/'browser_history.json');cp=read(evidence/'browser_checkpoint.json')
    summary=read(evidence/'browser_summary.json');brain_check(cp);ncp+=1
    assert frame['tick']==cp['tick'] and frame['simTime']==summary['physicalSeconds']==73.415
    assert len(history)==summary['samples']==735 and history[-1]['t']==73.4
    transitions=[]
    for e in frame['events']:
        if e['message'].startswith('configure '):
            conf=json.loads(e['message'][10:])
            if 'motorCoupled' in conf:transitions.append((e['time'],conf['motorCoupled']))
    norm=[]
    for h in history:
        motor=True
        for t,value in transitions:
            if h['t']>=t:motor=value
        norm.append(dict(t=h['t'],p=h['body']['position'],yaw=h['body']['yaw'],
            avoiding=False,recovering=False,motorCoupled=motor,motionExpected=True))
        assert len(h['physics']['jointAngles'])==42 and len(h['values'])==170
        if not motor:assert h['physics']['descending']==[0.,0.]
        assert h['physics']['fault'] is None
    a=audit_trajectory(norm);a['walkingResumed']=None;a['resumptionEpisodes']=[]
    check('browser stationary audit excludes explicit motor-off interval',a['status']=='PASS')
    result['browser']=dict(frameModelSeconds=frame['simTime'],lastSampleTime=history[-1]['t'],samples=len(history),
        motorTransitions=transitions,stationaryAudit=a,restoreRecord=read(evidence/'browser_restore.json'),
        limitation='Only archived UI/log/state evidence. No new browser session or native execution in this audit.')
    result['validatedNeuralCheckpoints']=ncp
    result['status']='PASS_RECORDED_EVIDENCE_AUDIT'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=ROOT/'verification/wall_fix_20260909')
    p.add_argument('--source',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=ROOT/'verification/fix2_review/reevaluation.json')
    args=p.parse_args()
    if sys.flags.optimize: p.error('Do not run evidence validation with -O / -OO')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    try:result=reevaluate(args.evidence.resolve(),args.source.resolve());code=0
    except Exception as e:
        import traceback
        result=dict(status='FAIL_EVIDENCE_AUDIT',newNativePhysicsExecuted=False,reason=str(e),traceback=traceback.format_exc());code=1
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(result['status']);return code
if __name__=='__main__':raise SystemExit(main())
