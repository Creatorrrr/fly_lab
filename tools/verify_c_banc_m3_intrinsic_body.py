"""Actual moving-body feedback for the intrinsic, source-connected BANC candidate."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from flylab.c.walking_trace import capture_control, contact_diagnostics
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_synaptic_efficacy import patch_weights
from tools.prepare_c_banc_m3_e99_lf_tarsal import source_screen as lf_source_screen
from tools.prepare_c_banc_m3_e105_startup import accepted_e104_failed_fit_review
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.probe_c_banc_m3_tonic_onset import phase_trace, quadratic_peaks
from tools.probe_c_banc_m3_tonic_phase import screens as source_phase_screens
from tools.verify_c_banc_observation import exact


def complete_source_trace(times, rate, ids, expected_ids, controls, reason):
    """Require every 5ms endpoint through 10s before a full source pass."""
    return bool(len(times)==2000
        and np.array_equal(times,np.arange(1,2001)*.005)
        and np.array_equal(ids,expected_ids)
        and rate.shape==(2000,len(expected_ids))
        and np.isfinite(rate).all()
        and controls==2000 and reason=='duration')


def equal_integer_ids(left,right,label):
    """GraphStore int32 and saved calibration int64 IDs may match by value."""
    left=np.asarray(left)
    right=np.asarray(right)
    if (left.dtype.kind not in 'iu' or right.dtype.kind not in 'iu'
            or left.shape!=right.shape or not np.array_equal(left,right)):
        raise ValueError(label)


def own_rm_inverse_at_actual_phase(times, e1_rate, mask, bins, warm_muscle, offset):
    """Sample one frozen inverse target at this run's bracketed RM E1 peaks."""
    peaks=quadratic_peaks(times,e1_rate)
    phase=phase_trace(times[mask],peaks)
    if not np.isfinite(phase).all():
        raise ValueError('Every RM target sample needs this run\'s bracketed RM E1 phase')
    target=np.interp(np.mod(phase-offset,1),np.r_[bins,1],
        np.r_[warm_muscle,warm_muscle[0]])
    return target,phase,peaks


def e105_startup_diagnostic(rows,native,prediction):
    """Require an unbroken original 99-control prefix before transfer passes."""
    controls=max(0,min(len(rows)-1,99))
    times=np.asarray([row['time_s'] for row in rows[1:100]])
    actual_q=np.asarray([row['joint_targets_rad'] for row in rows[1:100]],dtype=float)
    tendon_keys=tuple(f'{leg}_tarsus' for leg in ('lf','lm','lh','rf','rm','rh'))
    actual_ltm=np.asarray([[row['tendon_inputs'][key] for key in tendon_keys]
        for row in rows[1:100]],dtype=float)
    if not len(actual_q):
        actual_q=actual_q.reshape(0,42)
        actual_ltm=actual_ltm.reshape(0,6)
    predicted_q=np.asarray(prediction['command'])
    predicted_ltm=np.asarray(prediction['tendon'])
    complete=bool(controls==99 and len(native)>=4950
        and np.array_equal(times,np.arange(1,100)*.005)
        and actual_q.shape==(99,42) and actual_ltm.shape==(99,6)
        and predicted_q.shape==(2001,42) and predicted_ltm.shape==(2001,6)
        and np.isfinite(actual_q).all() and np.isfinite(actual_ltm).all()
        and np.isfinite(predicted_q[1:100]).all()
        and np.isfinite(predicted_ltm[1:100]).all())
    q_count=min(len(actual_q),max(0,len(predicted_q)-1)) if (
        actual_q.ndim==2 and actual_q.shape[1]==42
        and predicted_q.ndim==2 and predicted_q.shape[1]==42) else 0
    ltm_count=min(len(actual_ltm),max(0,len(predicted_ltm)-1)) if (
        actual_ltm.ndim==2 and actual_ltm.shape[1]==6
        and predicted_ltm.ndim==2 and predicted_ltm.shape[1]==6) else 0
    with np.errstate(over='ignore',invalid='ignore'):
        q_difference=(actual_q[:q_count]-predicted_q[1:q_count+1]
            if q_count else np.empty((0,42)))
        ltm_difference=(actual_ltm[:ltm_count]-predicted_ltm[1:ltm_count+1]
            if ltm_count else np.empty((0,6)))
    metrics_available=bool(complete and np.isfinite(q_difference).all()
        and np.isfinite(ltm_difference).all())
    q_rms=q_max=ltm_rms=ltm_max=None
    if metrics_available:
        with np.errstate(over='ignore',invalid='ignore'):
            q_rms=float(np.sqrt(np.mean(q_difference**2)))
            q_max=float(np.max(abs(q_difference)))
            ltm_rms=float(np.sqrt(np.mean(ltm_difference**2)))
            ltm_max=float(np.max(abs(ltm_difference)))
        metrics_available=bool(np.isfinite([q_rms,q_max,ltm_rms,ltm_max]).all())
        if not metrics_available:
            q_rms=q_max=ltm_rms=ltm_max=None
    transfer=bool(metrics_available and q_rms<=.002 and q_max<=.01)
    return {'complete_99_control_prefix':complete,'observed_controls':controls,
        'q_RMS_rad':q_rms,'q_max_abs_rad':q_max,'LTM_RMS':ltm_rms,
        'LTM_max_abs':ltm_max,'conditional_transfer_screen':transfer,
        'tendon_order':list(tendon_keys)},times,actual_q,actual_ltm,q_difference,ltm_difference


def run(source,root,graph_path,out,*,experiment='M3-E64-intrinsic-source-moving-body',
        prior_execution_wall_s=0.):
    out.mkdir(parents=True,exist_ok=False)
    native_all_leg_body = experiment == 'M3-E88-native-all-leg-recruitment-body'
    own_leg_body = experiment == 'M3-E90-own-leg-native-recruitment-body'
    coupling_phase_body = experiment == 'M3-E94-native-coupling-phase-body'
    rm_e2_body = experiment == 'M3-E95-rm-e2-source-body'
    e98_body = experiment == 'M3-E98-native-readout-body-discriminator'
    e100_body = experiment == 'M3-E100-LF-tarsal-native-body-discriminator'
    e102_body = experiment == 'M3-E102-independent-edge-native-body'
    e105_body = experiment == 'M3-E105-startup-native-body-diagnostic'
    all_leg_development = native_all_leg_body or own_leg_body or coupling_phase_body or rm_e2_body or e98_body or e100_body or e102_body or e105_body
    exact_recruitment_observation = experiment == 'M3-E77-native-recruitment-currents'
    source_basis_observation = experiment == 'M3-E84-native-premotor-basis-body'
    recruitment_observation = exact_recruitment_observation or experiment in (
        'M3-E79-RH-paired-recruitment-body','M3-E81-RH-positive-recruitment-body',
        'M3-E84-native-premotor-basis-body','M3-E88-native-all-leg-recruitment-body',
        'M3-E90-own-leg-native-recruitment-body',
        'M3-E94-native-coupling-phase-body','M3-E95-rm-e2-source-body',
        'M3-E98-native-readout-body-discriminator',
        'M3-E100-LF-tarsal-native-body-discriminator',
        'M3-E102-independent-edge-native-body',
        'M3-E105-startup-native-body-diagnostic')
    controls = 1000 if exact_recruitment_observation or source_basis_observation else 2000
    spec={'experiment':experiment,
        'question':'Can the unchanged E63six-source fullBANC candidate initiate and sustain actual healthy walking from the original all-zero neural/neutral body state without any external waveform?',
        'scope':'E59cross-start convergence and E63held-body coordination failed. This is a bounded diagnostic of the original ultimate moving-body outcome and startup, not adoption or a retroactive pass of either neural assay.',
        'intervention':'Restore the exact E63candidate-neutral full158706network/body/adapter/navigation/parameters. Native session.advance samples current physical sensory input every5ms and supplies original fixed forward navigation drive. No core teacher, rate preload, time/phase oscillator, pose/root edits, recorded command or neural prewarming.',
        'measurement':'Requested duration10s and original F3-F5 window[.5,10). Record actual200Hzstate and every100us native physical substep contact load/slip/nonfoot/upright without modifying integration. Original evaluator and thresholds. One neutral start is not F6. Health fault stops the diagnostic early and remains a failure.',
        'checkpoints':'At1s save full candidate, independently restore and advance .1s, then require exact whole future at1.1s. Main evolution paused during clone. No checksum-only equivalence.',
        'limits':'OneCUDA job; <=10main+.1clone neural/physicalseconds,<=285wallseconds. Stop fault/nonfoot at any native substep or upright<.9. No UI/M2/F6 changes.',
        'source_sha256':sha256(source/'calibration.npz'),'tool_sha256':sha256(__file__),
        'evaluator_sha256':sha256(Path('tools/analyze_c_banc_biological_gait.py'))}
    if experiment != 'M3-E64-intrinsic-source-moving-body':
        spec.update(question='Does the E65single observed-mean source-current correction retain all-three LR coordination and improve actual autonomous walking from the same original neutral start?',
            scope='E65held-body six-core coordination passed; it is not body/Fverification. Compare to E64curvedwalking/wallcontact/failedmiddle-hindcoordination. Same original10s/healthguard/world/body/sensory/forward input and F1-F8.',
            intervention='Restore exact E65candidate-neutral, including only its frozen positiveDN+INXXX464source efficacy correction. No additional parameter/body/start/phase/waveform change. Original allzero neural state and actual ongoing body sensory feedback.',
            observation_revision='E66 matches existing M2v2 slip timing by forwarding a separate copied mjData after each native step; the actor is never forwarded or overwritten. E64direct solver-contact arrays remain archived as their original observation version. Also retain actual bilateralDNg100 rate/drive/recurrentcurrent/adaptation to detect transient tonic failure.')
    if experiment == 'M3-E68-premotor-timing-body':
        spec.update(question='Does the single measured LMpremotor delay correction improve actual middle-leg LR phase while retaining E65/E66neural motion and other legs?',
            scope='E66late neural coordination and actual recurrence/amplitudes passed their local checks, but physical middlephase.6915 and curvedwallcontact failed. E67is preparation only; this is the first native neuron/body test of its one tau change. Original F1-F8 retained.',
            intervention='Restore exact E67bcandidate-neutral. Only sourceIN16B016720575941590291084 and its LMpleuralMN720575941553503460 share one changed tau. Its value is frozen in timing-hypothesis.json from half E66measured transfer offset distributed equally over the two first-order cells, preserving native dt<=tau/50. No weights/coretau/body/initialstate/forwardinput/decoder/waveform change. Record all prior recruitment inputcells including this relay.',
            timing_sha256=sha256(source/'timing-hypothesis.json'),
            hypothesis_prediction=json.loads((source/'timing-hypothesis.json').read_text()))
    if experiment == 'M3-E70-two-DN-adaptation-body':
        spec.update(question='Does removing only the two stimulated DNg100 adaptation gains prevent the E68transient output loss and preserve actual leg coordination?',
            scope='E69prepared physiology intervention in the exact E67b/E68context; preparation not evidence. E68rightDN dropped to56.325 at3.495s and middle/hind phase concentration failed. Keep original F1-F8 and E65/E66best candidate.',
            intervention='Restore E69candidate-neutral. Only the two existing DNg100 adaptation gains change8to0. All source weights,26tauvalues,LM timing correction,decoder,body,world,neutral initialr/z,actualsenses and fixedforward drive5000 unchanged. No outputforcing, prewarming, time/phase input or increased stimulation.',
            prediction='After.5s in the actually observed prefix, both DNadaptation must remain exactly0 and DNrate>=190 if the proposed tonic stability mechanism is sufficient. This mechanism screen is separate from original10s/F3-F5bodycriteria; no DN drop alone cannot pass gait.',
            hypothesis_sha256=sha256(source/'descending-hypothesis.json'))
    if experiment == 'M3-E73-front-knee-delivery-body':
        spec.update(question='Does the single positive anatomical front-knee delivery correction reduce the observed yaw bias without losing neural/body coordination?',
            scope='E71exact recorded-command bodies and read-only source audit locate LF/RFcommand amplitude imbalance. E72fits two positive moment scalars to already exposed E42commands/E50actualmuscle activity; fitting alone is not performance evidence. Keep E69/E70 and all original F1-F8.',
            intervention='Restore E72candidate-neutral. Only LF/RFtibia moment rows5/26 have data-derived gains; fixedrest, source signs/muscle ratios, allnetwork weights/taus/adaptation/input, remainingdecoder/tendons/body/world/initial fullstate unchanged fromE69/E70. No externalteacher, time/phase input, rootcorrection or recordedcommand at runtime.',
            prediction='At matched observed[.5,2.2] absolute yaw drift should decrease at least30% fromE70 .582323rad if frontkneebalance materially improves direction. Local mechanism screen does not replace original10s/F3-F5; preserve original failures and investigate remaining silentRHcoxa/middle stance if not supported.',
            hypothesis_sha256=sha256(source/'delivery-hypothesis.json'))
    if exact_recruitment_observation:
        spec.update(question='What actual recurrent currents, source rates and muscle activations accompany the silent RHcoxa pools in the unchanged E73 wholeBANC trajectory?',
            scope='Measurement of the already exposed E73 first5s, not an improved candidate or independent confirmation. E73 original10s F3 failure and all F1-F8 remain unchanged.',
            intervention='Restore unchanged E72candidate-neutral. Add nonmutating endpoint current/drive/adaptation observations for the original437 cells and three existing RHpremotor path cells. Record all96 muscle activations. No neural/body/world/input/decoder/time/phase intervention.',
            prediction='If silence is caused by below-threshold total current, the actual RHpleural/sternal MN drive+recurrent-adaptation-threshold should be nonpositive with zero output. Competing explanation is temporal filtering/pooling despite positive recruitment. A source path existing in the graph is not evidence of recruitment or direction improvement.',
            measurement='5s at200Hz, current Wr evaluated after each full control neural interval; not a native-substep average. All original437 rate trajectories, DNfields, qpos/qvel/ctrl and native contact arrays must match the E73 prefix exactly. All newly observed values must be finite. Inspect current-space recruitment and existing RHrelay modulation before selecting a change.',
            limits='OneCUDA job;<=5main+.1clone neural/physicalseconds,<=285wallseconds. Original native health guards. No candidate fitting, F6 exposure or M2/UI changes.',
            comparison_sha256=sha256(root/'e73-front-knee-delivery-body/signals.npz'))
    if experiment in ('M3-E79-RH-paired-recruitment-body','M3-E81-RH-positive-recruitment-body'):
        spec.update(question='Does the E78joint recruitment of two silent RHcoxa muscles improve actual within-leg motion and walking direction?',
            scope='First actual native fullBANC/body test of the conditional E78fit. E73development F4/F5pass but F3circle remains comparator. Static geometry, sourcepath existence and fitted error alone do not establish gait improvement. Original F1-F8 unchanged.',
            intervention='Restore E78candidate-neutral: two original RHrelay input/physiology patches, seven existing source-to-twoRH MN efficacies and positive RHcoxa moment-row36/37 gains. Exact unchanged original neural/physicalstart, otherweights/26taus/decoderrest/body/world/forwardinput. No externalcoreteacher, phase, recordedcommand or root edit.',
            prediction='On observed stable complete sourcecycles, each RHcoxa muscle activation p95-p05>=1 and RHcoxa/foot reference error lower than unchanged E73/E77. Direction screen requires>=30% lower absolute yaw in matched[.5,2.2) versusE73 while healthy. Original10s F3-F5 are required separately; changes may perturb whole-network feedback and coordination.',
            measurement='200Hz full source/MNendpoint current,drive,adaptation,96muscleactivation and original physicalstate; every100us native contact/slip/upright. Current sampling is an endpoint observation, not native neural-time averaging.',
            hypothesis_sha256=sha256(source/'recruitment-hypothesis.json'))
    if experiment == 'M3-E81-RH-positive-recruitment-body':
        spec.update(question='Does the existing E1-following positive RHpremotor path recover rhythmic pleural recruitment and improve actual within-leg motion/direction?',
            scope='E78/E79tonic-plus-inhibitory design reduced local foot error but pleural remained almosttonic, earlyyawratio.8094 failed and5.295s nativecontact stopped. Target/source alignment was independently verified; filteredE1-following source correlates positively with desiredpleural, motivating one originalpositivepath alternative. E73remains10s/F4/F5developmentbaseline.',
            intervention='Restore E80candidate-neutral from the originalE72/E77baseline. Replace the proposed inhibitoryIN16B016 with existing excitatoryIN01A016720575941390358788; preserve originalgraph/signs. Same two-cell tonic/rhythmic input+adaptation policy, sixexistingRHmotor inputefficacies, two positiveRHcoxa momentgains; alltau/rest/othercontroller/body/world/fullzero initialstate/input unchanged. No previousIN16patch retained and no externalteacher/phase/command/root correction.')
    if source_basis_observation:
        spec.update(question='Do the three existing premotor source preparations transmit usable native steady/rhythmic signals under the whole neural/body feedback?',
            scope='E83 prepares only existingsource inputs/physiology; all391MNincoming weights and entiredecoder are exactE72/E73. Missing LFpositive tarsal direction is not yet recruited. This5s development data collection supports a later all-jointfit, not F3/F6 or an improved walking candidate.',
            intervention='Restore E83original neutral: RH IN01A015tonic/IN01A016rhythmic plus LF IN03A013 receiving its existingLFtonic andDNg100 originalinputs; only these three incoming rows and adaptation gains changed. Original26tauvalues and all motorweights/moments/rest/body/world/input unchanged. No teacher/phase/recordedcommand/rootedit.',
            prediction='In observed[1,5), new LFtonic source should have positive rate with coefficient of variation<=.05, RHtonic likewise, and RHrhythmic source should have p95-p05>=20 with visible E1-related modulation. Failure requires source diagnosis before fitting. Report actualsource currents/physiology and bodyquality separately.',
            measurement='Original complete5s/nativehealthguard; everycontrol all selected47source+391MN+bilateralDNendpointcurrent,drive,adaptation and96muscleactivation; original100usnativecontact. No bitexact E73trajectory expected after this specifiedthree-source change.',
            limits='OneCUDA job <=5main+.1clone neural/physicalseconds,<=180wallseconds;noM2/UI/F6changes.',
            hypothesis_sha256=sha256(source/'source-basis-hypothesis.json'))
    if native_all_leg_body:
        spec.update(
            question='Can the E86 native source cadence plus E87 existing-edge all-leg recruitment produce healthy forward walking in the original full BANC/body environment?',
            scope='E87 conditional fit improved static output but did not test moving body; E86 middle/hind neural LR failed. E73 remains the original10s F4/F5 development comparator with F3 circle failure. E85 is external-body reference only. This combined intervention cannot isolate source kinetics from recruitment weights.',
            intervention='Restore recovered E87 full158706 CUDA candidate-neutral from original E32 neural/body/world zero state: E86 constant30 source tau scaling and E87 positive efficacies shared by anatomical muscle on1502 existing source-to-MN edges. Preserve other weights, fixed moment/rest/pool/tendons, 32tau entries and original fixed forward DNg100 drive5000. Runtime uses native physical sensory input; no waveform/phase/prewarming/root correction.',
            measurement='One original10s/200Hz full BANC and actual moving body. Record all47 existing source cells, both DNg100, all391 MN, actual endpoint recurrent current/drive/adaptation and96 activation; all100us physical contact/health. Original F3-F5 numeric development screens on fixed initial heading and complete native health. One neutral start is not F6.',
            prediction='A finite conditional improvement may fail actual transfer, LR coordination or original-world direction. Any failure remains failure and does not change F1-F8.',
            limits='One CUDA/physics job <=10main+.1clone neural and physical seconds,<=285wallseconds. Stop any native nonfoot contact, upright<.9 or fault. No F6/UI/M2 changes.',
            recovered_fit_result_sha256=sha256(source/'result.json'),
            e86_calibration_sha256=sha256(root/'e86-native-source-cadence/calibration.npz'),
            e85_reference_result_sha256=sha256(root/'e85-space-compatible-cadence-reference/result.json'))
    if own_leg_body:
        spec.update(
            question='Does the E89 own-leg aligned conditional recruitment candidate produce healthy complete original-world forward walking from the original neutral BANC/body start?',
            scope='E89 offline own-leg shape screen passed against frozen E87 on the same recorded E86 source timeline. E88 actual whole-body 10s stayed healthy but failed F3/F4/F5. E90 is one actual combined E86 kinetics plus E89 source-efficacy trial; source middle/hind LR has not been repaired or separately isolated.',
            intervention='Restore E89 candidate-neutral: original E32 allzero full neural/body/world, E86 fixed30source kinetics and E89 fitted positive efficacies on1502 original signed source-to-MN edges. Fixed 32tau/moment/rest/pool/tendons/otherweights/navigation and DNg100 forward5000. Runtime native physical sensory input, no phase/teacher/prewarming/command/root correction.',
            measurement='Original complete10s/200Hz full158706 CUDA and moving body, actual47source+2DN+391MN endpoint rate/current/drive/adaptation,96muscleactivation,100us native physical contact. Original fixed initial heading F3, forward-eligible F4 and geometric knee F5 numeric screens. One neutral start is not F6.',
            prediction='Offline within-leg improvement may fail under moving-body feedback, original-world direction or source LR timing; all original F1-F8 and health guards remain fixed.',
            limits='One CUDA/physics job<=10main+.1clone neural and physical seconds,<=285wallseconds; native nonfoot/upright<.9/fault stops. No F6/UI/M2 changes.',
            e89_runner_sha256=sha256(source/'runner.py'),
            e89_result_sha256=sha256(source/'result.json'),
            e89_parent_review_sha256=sha256(source/'parent-review.json'),
            e88_result_sha256=sha256(root/'e88-native-all-leg-recruitment-body/result.json'),
            e86_calibration_sha256=sha256(root/'e86-native-source-cadence/calibration.npz'))
    if coupling_phase_body:
        spec.update(
            question='Does the E93 conditional six-coupling-kinetics/output candidate retain early three-pair neural antiphase and produce healthy original-world walking under actual complete BANC and moving-body feedback?',
            scope='E93 conditional 36-cell model passed its nine phase windows but exceeded its shared reduced-neural budget by10s; numerical candidate remains reviewable. E91 first-peak and E92 later-phase failures remain. E94 is one original neutral whole BANC/body development start, not F6 or final biological adoption.',
            intervention='Restore E93 candidate-neutral from original E32 zero neural/body/world. E89 motor recruitment remains; only six existing coupling taus, six selected existing coupling-to-E1 positive outputs, and six tonic taus differ. Original 158706-cell CUDA, native100us body, actual sensory feedback and fixed forward DNg100 5000. No phase, teacher, rate preload, root correction or additional fit.',
            measurement='One10s full BANC/body plus independent .1s whole clone; record original440 source/MN/DN and96 activation, append existing RH3 and four RM relay rate/current/drive/adaptation observations without changing dynamics. Evaluate E92 six source cadence and nine LR windows separately from fixed physical F3/F4/F5.',
            limits='One CUDA/physics job <=10main+.1clone neural and physical seconds, <=285wallseconds; native health guard every100us. F6/M2/UI untouched.',
            e93_result_sha256=sha256(source/'result.json'),
            e93_spec_sha256=sha256(source/'spec.json'),
            e93_runner_sha256=sha256(source/'runner.py'),
            e93_parent_second_review_sha256=sha256(source/'parent-second-review.json'),
            e93_fixed_inputs_sha256=sha256(source/'fixed-inputs.npz'),
            e89_calibration_sha256=sha256(root/'e89-own-leg-native-recruitment/calibration.npz'),
            e85_reference_sha256=sha256(root/'e85-space-compatible-cadence-reference/commands.npz'))
    if rm_e2_body:
        spec.update(
            question='Does the prepared existing RM E2-to-IN03A010 source row produce a useful native RM sternal recruitment waveform while keeping whole BANC source coordination?',
            scope='E94 full source passed but actual physical F3-F5 failed. E95 changes only one existing receiver row and that cell adaptation, with original motor readout and body fixed. Source preparation cannot itself establish physical gait or final adoption.',
            intervention='Restore E95 prepared candidate-neutral from original E32 zero neural/body/world: RM E2→IN03A010 existing positive input effective1.1707944, other existing nonzero receiver edges source*.001, gain8→0, tau20ms unchanged. Existing E93 coupling and E89 motor weights, original drive5000/sensory/physics remain. No fit, external phase, prewarming, root correction or command injection.',
            measurement='One10s complete CUDA/body plus .1s disk future. Basic48source+391MN+2DN=441 and existing RH/RM extra readouts deduplicated; actual5ms rates/current/drive/adaptation/96activation and100us native health. Separate complete-source and physical F screens. Offline own-leg inverse target correlation only after capture.',
            limits='Preparation<=90wall, body<=285wall, shared<=375wall; actual<=10.1 neural/physical seconds. No F6/M2/UI change.',
            e95_preparation_sha256=sha256(source/'result.json'),
            e95_calibration_sha256=sha256(source/'calibration.npz'),
            e95_runner_sha256=sha256(source/'runner.py'),
            e94_parent_second_review_sha256=sha256(root/'e94-native-coupling-phase-body-v3/parent-second-review.json'),
            E89_warm_start_sha256=sha256(root/'e89-own-leg-native-recruitment/warm-start.npz'),
            E89_own_leg_phase_sha256=sha256(root/'e89-own-leg-native-recruitment/own-leg-phase.npz'))
    if e98_body:
        spec.update(question='Does the frozen E97 readout yield actual healthy original-world walking despite its failed conditional q shape screen?',
            scope='E97 shape_supported is false and remains false. E98 is one direct full BANC/body discriminator, not a change to the conditional criterion or final F1-F8.',
            intervention='Restore E98 prepared original E32 zero-state candidate: E95 calibration plus E97 fixed 590 positive efficacy parameters on 1536 original signed source-to-MN edges only. No softplus, phase, teacher, prewarming, root correction or further fit.',
            measurement='One10s native full BANC/body and .1s disk clone; observe48sources+391MN+2DN and existing RH/RM relays plus LF IN19B003 index60038 endpoint current/rate/drive/adaptation and96 activation. Original native health, F3-F5 and full source screens remain.',
            limits='Preparation<=90wall, body<=285wall, shared<=375wall; actual<=10.1 neural/physical seconds. F6/M2/UI unchanged.',
            e98_preparation_sha256=sha256(source/'result.json'),
            e98_calibration_sha256=sha256(source/'calibration.npz'),
            e97_result_sha256=sha256(root/'e97-softplus-training-native-recruitment/result.json'),
            e97_parent_review_sha256=sha256(root/'e97-softplus-training-native-recruitment/parent-first-review.json'),
            e95_body_result_sha256=sha256(root/'e95-rm-e2-premotor-source/body/result.json'),
            lf_path_audit_sha256=sha256(root/'e96-observed-native-recruitment/parent-LF-existing-tarsal-paths.json'))
    if e100_body:
        if not np.isfinite(prior_execution_wall_s) or prior_execution_wall_s<0:
            raise ValueError('Nonnegative E100 prior failed execution wall cost required')
        spec.update(question='Do the two E99 prepared existing LF premotor sources retain their conditional rhythms under actual complete BANC and body feedback?',
            scope='E99 conditional source screens passed and parent second review accepted them; E98 actual physical F3-F5 failed. This is one source/body discriminator, not a fit or final adoption.',
            intervention='Restore reviewed E99 candidate-neutral from original E32 zero neural/body/world: only two existing LF source input rows and adaptation60038 changed versus E98. Retain E97 frozen391MN readout,38tau, fixed drive5000 and native sensory/body. No teacher, phase, prewarming, softplus, root correction or new fit.',
            measurement='One10s native full158706 CUDA/body and .1s independent whole disk future;49source+391MN+2DN basic,446 detailed unique cells,96activation and native100us health. Six cadence/nine LR and E99 two LF source screens on actual own-LF phase; physical F3-F5 separate.',
            limits='One body<=285wall, E99 prior19.175135 plus body<=375wall; actual<=10.1neural/physicalseconds. No F6/M2/UI/product changes.',
            e99_result_sha256=sha256(source/'result.json'),
            e99_amendment_sha256=sha256(source/'parent-first-review-amendment.json'),
            e99_parent_second_review_sha256=sha256(source/'parent-second-review.json'),
            e99_calibration_sha256=sha256(source/'calibration.npz'),
            e99_corrected_screen_sha256=sha256(source/'corrected-conditional-screen.json'),
            e98_result_sha256=sha256(root/'e98-native-readout-body-discriminator/body/result.json'),
            e97_result_sha256=sha256(root/'e97-softplus-training-native-recruitment/result.json'),
            e85_reference_sha256=sha256(root/'e85-space-compatible-cadence-reference/commands.npz'),
            E89_warm_start_sha256=sha256(root/'e89-own-leg-native-recruitment/warm-start.npz'),
            evaluator_sha256=sha256(Path(__file__).with_name('analyze_c_banc_biological_gait.py')))
        spec['prior_failed_execution_wall_s']=prior_execution_wall_s
    if e102_body:
        if not np.isfinite(prior_execution_wall_s) or prior_execution_wall_s<0:
            raise ValueError('Nonnegative E102 prior failed execution wall cost required')
        spec.update(question='Does the reviewed E101 independent-edge conditional candidate improve actual original-world physical walking?',
            scope='E101 passed its local conditional delivery screen; E100 actual source screens passed but physical F3-F5 failed. This one full-body development start does not alter F1-F8 or imply adoption.',
            intervention='Restore E101 original E32 zero-state candidate. Only1572 existing signed49source-to391MN edges carry fitted independent positive efficacies; 49source receiving rows/gains,38tau,decoder,body/world,drive5000 fixed. No teacher, phase, prewarm, root correction or new fit.',
            measurement='One original10s full BANC/native body plus independent .1s whole disk clone;442 basic,446 detailed,55 observational source IDs,96activation/endpoint current/rate/drive/adapt,100000native health. Original F3-F5,6cadence/9LR, two actual LF source screens separately.',
            limits='One body<=285wall, E101 prior44.2162915+body<=375wall; neural/physical<=10.1each. No F6/M2/UI/product changes.',
            e101_result_sha256=sha256(source/'result.json'),
            e101_calibration_sha256=sha256(source/'calibration.npz'),
            e101_coefficients_sha256=sha256(source/'fit-coefficients.npz'),
            e101_parent_first_review_sha256=sha256(source/'parent-first-review.json'),
            e99_calibration_sha256=sha256(root/'e99-lf-tarsal-source-preparation/calibration.npz'),
            e100_result_sha256=sha256(root/'e100-lf-tarsal-native-body-v2/result.json'),
            E89_warm_start_sha256=sha256(root/'e89-own-leg-native-recruitment/warm-start.npz'),
            evaluator_sha256=sha256(Path(__file__).with_name('analyze_c_banc_biological_gait.py')))
        spec['prior_failed_execution_wall_s']=prior_execution_wall_s
    if e105_body:
        if not np.isfinite(prior_execution_wall_s) or prior_execution_wall_s<0:
            raise ValueError('Nonnegative E105 prior failed execution cost required')
        spec.update(question='Does the E104 startup-passing but steady-failing fixed readout recover original native body posture over the first .5s and sustain actual walking?',
            scope='Separate direct physical startup discriminator. E104 steady/local_supported remains false; F3-F5 and full source screens unchanged.',
            intervention='Restore E105 prepared E99 original E32 zero-state with only E104 selected1572 actualfloat32 signed source-to-MN weights. No smoothing, phase, teacher, prewarm, root correction, fit or alternate start.',
            measurement='One10s full BANC/native moving body plus .1s independent whole disk future; 442basic/446detailed/55metadata observations, original native health/F3-F5/six cadence/nine LR/two LF source screens. Separate first.5s health and actual99 command/LTM transfer versus E104 stored prediction.',
            limits='Preparation<=60wall, body<=285wall, E10441.4175374+prep+body+failed-preflight<=375wall, actual<=10.1 neural/physicalseconds. No F6/M2/UI/product changes.',
            e105_preparation_sha256=sha256(source/'result.json'),
            e105_calibration_sha256=sha256(source/'calibration.npz'),
            e104_result_sha256=sha256(root/'e104-constrained-startup-recruitment/result.json'),
            e104_prediction_sha256=sha256(root/'e104-constrained-startup-recruitment/prediction.npz'),
            e104_parent_second_review_sha256=sha256(root/'e104-constrained-startup-recruitment/parent-second-review.json'),
            e99_calibration_sha256=sha256(root/'e99-lf-tarsal-source-preparation/calibration.npz'),
            E89_warm_start_sha256=sha256(root/'e89-own-leg-native-recruitment/warm-start.npz'),
            evaluator_sha256=sha256(Path(__file__).with_name('analyze_c_banc_biological_gait.py')))
        spec['prior_failed_execution_wall_s']=prior_execution_wall_s
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter()
    graph=GraphStore.load(graph_path)
    e99_prior_wall=0.
    e101_prior_wall=0.
    e105_prior_wall=0.
    if e105_body:
        prepared=json.loads((source/'result.json').read_text())
        e104=json.loads((root/'e104-constrained-startup-recruitment/result.json').read_text())
        review=json.loads((root/'e104-constrained-startup-recruitment/parent-second-review.json').read_text())
        if (prepared.get('validity')!='valid_prepared_startup_diagnostic'
                or prepared.get('neutral_disk_exact') is not True
                or prepared.get('E104_startup_supported') is not True
                or prepared.get('E104_steady_supported') is not False
                or prepared.get('E104_local_supported') is not False
                or prepared.get('fitted_edges')!=1572
                or e104.get('startup_screen',{}).get('supported') is not True
                or e104.get('shape_screen',{}).get('supported') is not False
                or e104.get('local_supported') is not False
                or not accepted_e104_failed_fit_review(review)):
            raise ValueError('Reviewed E104 failed-shape startup and E105 diagnostic preparation required')
        e105_prior_wall=float(prepared['wall_s'])
        fitted=np.load(source/'calibration.npz')
        previous=np.load(root/'e99-lf-tarsal-source-preparation/calibration.npz')
        coefficients=np.load(root/'e104-constrained-startup-recruitment/fit-coefficients.npz')
        predicted=np.load(root/'e104-constrained-startup-recruitment/prediction.npz')
        if (len(fitted['tau_ids'])!=38 or fitted['theta'].shape!=(96,48)
                or len(fitted['individual_source_basis_ids'])!=49
                or len(fitted['individual_edge_ids'])!=1572):
            raise ValueError('E105 legacy48 and independent49/1572 provenance required')
        for key in previous.files:
            if key!='values':
                exact(previous[key],fitted[key],'E105 unchanged E99 calibration '+key)
        for cal_key,fit_key in (('individual_edge_ids','edge_ids'),
                ('individual_edge_efficacies','theta'),
                ('individual_edge_motor_ids','motor_ids'),
                ('individual_edge_source_ids','source_ids')):
            exact(fitted[cal_key],coefficients[fit_key],'E105 E104 '+cal_key)
        exact(np.unique(np.r_[fitted['input_ids'],fitted['intrinsic_tonic_ids'],
            fitted['coupling_ids']]),fitted['individual_source_basis_ids'],
            'E105 exact49 source basis')
        base,_=rate_weights(graph)
        base=BancTargetEncoder(graph,forward_drive=5000).calibrate_weights(base,fitted['motor_ids'])
        before=patch_weights(base,previous['edge_ids'],previous['source_values'],previous['values'])
        after=patch_weights(base,fitted['edge_ids'],fitted['source_values'],fitted['values'])
        selected=fitted['individual_edge_ids']
        other=np.ones(len(after.data),dtype=bool)
        other[selected]=False
        exact(before.data[other],after.data[other],'E105 all nonfit/source receiving weights')
        exact(before.indptr,after.indptr,'E105 original CSR offsets')
        exact(before.indices,after.indices,'E105 original CSR columns')
        exact(after.data[selected],np.asarray(base.data[selected].astype(float)
            *fitted['individual_edge_efficacies'],dtype=np.float32),
            'E105 selected actualfloat32 signed edges')
        exact(after[fitted['motor_ids']][:,fitted['individual_source_basis_ids']]
            .toarray().astype(float),predicted['actual_float32_source_weights'],
            'E105 E104 actualfloat32 source-to-MN matrix')
    if e102_body:
        fitted_result=json.loads((source/'result.json').read_text())
        review=json.loads((source/'parent-first-review.json').read_text())
        if (fitted_result.get('validity')!='valid_conditional_fit'
                or fitted_result.get('shape_screen',{}).get('supported') is not True
                or fitted_result.get('resource_limit_met') is not True
                or fitted_result.get('neutral_disk_exact') is not True
                or fitted_result.get('fit_parameters')!=1572
                or fitted_result.get('fitted_edges')!=1572
                or review.get('accepted') is not True
                or review.get('blocking_defects')!=[]):
            raise ValueError('Accepted E101 independent-edge conditional candidate required')
        e101_prior_wall=float(fitted_result['wall_s'])
        fitted=np.load(source/'calibration.npz')
        previous=np.load(root/'e99-lf-tarsal-source-preparation/calibration.npz')
        coefficients=np.load(source/'fit-coefficients.npz')
        if (len(fitted['tau_ids'])!=38 or fitted['theta'].shape!=(96,48)
                or len(fitted['individual_source_basis_ids'])!=49
                or len(fitted['individual_edge_ids'])!=1572
                or len(np.unique(fitted['individual_edge_ids']))!=1572):
            raise ValueError('E101 49-source independent1572 edge and legacy48 theta required')
        for key in previous.files:
            if key!='values':
                exact(previous[key],fitted[key],'E102 unchanged E99 calibration '+key)
        for cal_key,fit_key in (('individual_edge_ids','edge_ids'),
                ('individual_edge_efficacies','theta'),
                ('individual_edge_motor_ids','motor_ids'),
                ('individual_edge_source_ids','source_ids')):
            exact(fitted[cal_key],coefficients[fit_key],'E102 exact E101 '+cal_key)
        exact(np.unique(np.r_[fitted['input_ids'],fitted['intrinsic_tonic_ids'],
            fitted['coupling_ids']]),fitted['individual_source_basis_ids'],
            'E102 exact49 source basis')
        base,_=rate_weights(graph)
        base=BancTargetEncoder(graph,forward_drive=5000).calibrate_weights(base,fitted['motor_ids'])
        before=patch_weights(base,previous['edge_ids'],previous['source_values'],previous['values'])
        after=patch_weights(base,fitted['edge_ids'],fitted['source_values'],fitted['values'])
        selected=fitted['individual_edge_ids']
        equal_integer_ids(fitted['individual_edge_source_ids'],graph.indices[selected],
            'E102 fitted source ID must equal original CSR presynaptic ID')
        exact(fitted['individual_edge_motor_ids'],np.searchsorted(graph.indptr,selected,
            side='right')-1,'E102 fitted motor ID is original CSR receiver ID')
        other=np.ones(len(after.data),dtype=bool)
        other[selected]=False
        exact(before.data[other],after.data[other],
            'E102 all nonfitted W and all49 source receiving rows')
        exact(after.data[selected],np.asarray(base.data[selected].astype(float)
            *fitted['individual_edge_efficacies'],dtype=np.float32),
            'E102 exact actualfloat32 1572 fitted signed weights')
        exact(after.indices,before.indices,'E102 original CSR columns')
        exact(after.indptr,before.indptr,'E102 original CSR offsets')
        if (not np.array_equal(np.sign(after.data[selected]),np.sign(base.data[selected]))
                or not np.all(np.isin(fitted['individual_edge_motor_ids'],fitted['motor_ids']))):
            raise ValueError('E102 only original signed MN receiver edges required')
    if e100_body:
        preparation=json.loads((source/'result.json').read_text())
        amendment=json.loads((source/'parent-first-review-amendment.json').read_text())
        review=json.loads((source/'parent-second-review.json').read_text())
        if (preparation.get('validity')!='valid_conditional_preparation'
                or preparation.get('conditional_source_screen') is not True
                or preparation.get('neutral_disk_exact') is not True
                or preparation.get('source_count')!=49
                or preparation.get('legacy_theta_columns')!=48
                or amendment.get('corrected_screen')!={'depressor_60038':True,'levator_11088':True}
                or amendment.get('shared_90s_limit_met') is not True
                or review.get('accepted') is not True
                or review.get('remaining_defects')!=[]):
            raise ValueError('Reviewed complete E99 source preparation required')
        e99_prior_wall=float(amendment['shared_wall_s'])
        fitted=np.load(source/'calibration.npz')
        previous=np.load(root/'e98-native-readout-body-discriminator/calibration.npz')
        if (len(fitted['tau_ids'])!=38 or fitted['theta'].shape!=(96,48)
                or len(np.unique(np.r_[fitted['input_ids'],fitted['intrinsic_tonic_ids'],
                    fitted['coupling_ids']]))!=49):
            raise ValueError('E99 49-source basis, legacy48 theta and38tau required')
        for key in previous.files:
            if key not in ('edge_ids','source_values','values','input_ids','zero_adaptation_ids'):
                exact(previous[key],fitted[key],'E100 unchanged E98 calibration '+key)
        exact(np.union1d(previous['zero_adaptation_ids'],[60038]),
            fitted['zero_adaptation_ids'],'E100 one new gain0 cell')
        previous_sources=np.unique(np.r_[previous['input_ids'],previous['intrinsic_tonic_ids'],
            previous['coupling_ids']])
        exact(np.union1d(previous_sources,[60038]),
            np.unique(np.r_[fitted['input_ids'],fitted['intrinsic_tonic_ids'],
                fitted['coupling_ids']]),'E100 49 original+LF source identities')
        base,_=rate_weights(graph)
        base=BancTargetEncoder(graph,forward_drive=5000).calibrate_weights(base,fitted['motor_ids'])
        before=patch_weights(base,previous['edge_ids'],previous['source_values'],previous['values'])
        after=patch_weights(base,fitted['edge_ids'],fitted['source_values'],fitted['values'])
        changed=np.flatnonzero(after.data!=before.data)
        posts=np.searchsorted(graph.indptr,changed,side='right')-1
        if (len(changed)!=595 or set(posts)!={60038,11088}
                or np.intersect1d(changed,np.concatenate([
                    np.arange(graph.indptr[m],graph.indptr[m+1]) for m in fitted['motor_ids']])).size):
            raise ValueError('E99 only two source receiver rows/595 changes and no MN change required')
        exact(before[fitted['motor_ids']].data,after[fitted['motor_ids']].data,
            'E100 original391 MN incoming rows')
    if e98_body:
        preparation=json.loads((source/'result.json').read_text())
        e97=json.loads((root/'e97-softplus-training-native-recruitment/result.json').read_text())
        if (preparation.get('validity')!='valid_prepared_diagnostic'
                or preparation.get('status')!='diagnostic_from_failed_conditional_shape'
                or preparation.get('E97_shape_supported') is not False
                or preparation.get('neutral_disk_exact') is not True
                or preparation.get('fitted_edges')!=1536
                or preparation.get('fit_parameters')!=590
                or preparation.get('resource_limit_met') is not True
                or e97.get('shape_screen',{}).get('supported') is not False):
            raise ValueError('E98 must preserve the reviewed E97 failed shape screen')
        fitted=np.load(source/'calibration.npz')
        previous=np.load(root/'e95-rm-e2-premotor-source/calibration.npz')
        for key in previous.files:
            if key not in ('values','theta','theta_source_ids'):
                exact(previous[key],fitted[key],'E98 unchanged E95 calibration '+key)
        exact(previous['theta'],fitted['legacy_theta_e89'],'E98 legacy47 theta')
        exact(previous['theta_source_ids'],fitted['legacy_theta_source_ids_e89'],
            'E98 legacy47 source order')
        coeff=np.load(root/'e97-softplus-training-native-recruitment/fit-coefficients.npz')
        exact(coeff['theta'],fitted['theta'],'E98 E97 frozen theta')
        exact(coeff['source_ids'],fitted['theta_source_ids'],'E98 48 source order')
        if len(fitted['tau_ids'])!=38:
            raise ValueError('E98 original38 tau values required')
        path=json.loads((root/'e96-observed-native-recruitment/parent-LF-existing-tarsal-paths.json').read_text())
        if (graph.nodes[60038]['cell_type']!='IN19B003'
                or str(graph.nodes[60038]['root_id'])!=str(path['top_existing_positive_E2_twohop_paths'][0]['node']['root_id'])
                or str(graph.nodes[60038]['root_id'])!='720575941511186508'):
            raise ValueError('Original LF IN19B003 identity required')
        original_w,_=rate_weights(graph)
        for post,pre,expected in ((60038,29608,7.44),(99007,60038,.03),
                (134148,60038,.15),(157447,60038,.27)):
            if not np.isclose(float(original_w[post,pre]),expected,rtol=1e-6,atol=0):
                raise ValueError('Original LF tarsal anatomical path required')
    if coupling_phase_body:
        prior=json.loads((source/'result.json').read_text())
        review=json.loads((source/'parent-second-review.json').read_text())
        if (prior.get('validity')!='valid_conditional_diagnostic'
                or prior.get('screen') is not True
                or prior.get('float64_screen') is not True
                or prior.get('runtime_float32_screen') is not True
                or prior.get('neutral_disk_exact') is not True
                or len(prior.get('rhythms',[]))!=6
                or len(prior.get('LR_by_window',{}))!=9
                or not all(item['screen'] for item in prior['rhythms'])
                or not all(item['screen'] for item in prior['LR_by_window'].values())
                or not all(item['screen'] for item in prior['runtime_rhythms'])
                or not all(item['screen'] for item in prior['runtime_LR_by_window'].values())
                or review.get('code_correction_accepted') is not True
                or review.get('remaining_blocking_defects')!=[]
                or review.get('predeclared_resource_compliance') is not False):
            raise ValueError('Reviewed E93 conditional candidate and resource deviation required')
        fitted=np.load(source/'calibration.npz')
        baseline=np.load(root/'e89-own-leg-native-recruitment/calibration.npz')
        selected=np.load(source/'fixed-inputs.npz')
        if (len(fitted['tau_ids'])!=38 or len(baseline['tau_ids'])!=32
                or len(selected['edge_ids'])!=6
                or not np.array_equal(fitted['tau_ids'][:32],baseline['tau_ids'])
                or not np.array_equal(fitted['tau_ids'][32:],baseline['intrinsic_tonic_ids'])):
            raise ValueError('E93 six coupling plus six tonic tau provenance required')
        for key in baseline.files:
            if key not in ('values','tau_ids','tau_values'):
                exact(baseline[key],fitted[key],'Fixed E89 calibration '+key)
        edge_positions=selected['selected_cal_edge_positions']
        tau_positions=selected['selected_cal_tau_positions']
        other_edges=np.ones(len(baseline['values']),dtype=bool)
        other_edges[edge_positions]=False
        other_tau=np.ones(32,dtype=bool)
        other_tau[tau_positions]=False
        exact(baseline['values'][other_edges],fitted['values'][other_edges],
            'All nonselected E89 output weights')
        exact(baseline['tau_values'][other_tau],fitted['tau_values'][:32][other_tau],
            'All other E89 tau values')
        exact(fitted['values'][edge_positions],
            np.load(source/'selected-runtime-float32.npz')['local_weights'][
                selected['target_rows'],selected['relay_columns']].astype(np.float32),
            'Six selected runtime float32 output edges')
        exact(fitted['tau_values'][tau_positions].astype(np.float32),
            np.load(source/'selected-runtime-float32.npz')['tau'][24:30].astype(np.float32),
            'Six native float32 coupling taus')
        exact(fitted['tau_values'][32:].astype(np.float32),
            np.load(source/'selected-runtime-float32.npz')['tau'][30:].astype(np.float32),
            'Six native float32 tonic taus')
    if rm_e2_body:
        preparation=json.loads((source/'result.json').read_text())
        if (preparation.get('validity')!='valid_prepared_existing_RM_source'
                or preparation.get('neutral_disk_exact') is not True
                or preparation.get('selected_source_count')!=48
                or preparation.get('actual_neural_s')!=0
                or preparation.get('actual_physical_s')!=0):
            raise ValueError('Exact E95 prepared zero-advance candidate required')
        fitted=np.load(source/'calibration.npz')
        previous=np.load(root/'e93-native-coupling-phase-v2/calibration.npz')
        hypothesis=json.loads((source/'source-hypothesis.json').read_text())
        if (len(fitted['tau_ids'])!=38 or hypothesis['relay_id']!=54014
                or hypothesis['source_id']!=43618
                or len(fitted['edge_ids'])-len(previous['edge_ids'])
                    !=hypothesis['incoming_existing_edges']):
            raise ValueError('One existing RM source receiver row and fixed38taus required')
        for key in previous.files:
            if key not in ('edge_ids','source_values','values',
                    'input_ids','zero_adaptation_ids'):
                exact(previous[key],fitted[key],'Unchanged E93 calibration '+key)
        for key in ('edge_ids','source_values','values','input_ids'):
            exact(previous[key],fitted[key][:len(previous[key])],
                'Unchanged E93 calibration prefix '+key)
        exact(np.union1d(previous['zero_adaptation_ids'],[54014]),
            fitted['zero_adaptation_ids'],'E95 one new gain0 cell')
    if native_all_leg_body or own_leg_body:
        prior=json.loads((source/'result.json').read_text())
        if (prior.get('validity')!='valid_conditional_fit'
                or (native_all_leg_body and not prior.get('reconstruction_exact'))
                or (own_leg_body and (prior.get('reconstruction_exact') is not False
                    or prior.get('shape_screen',{}).get('development_screen') is not True))
                or prior.get('fit_parameters')!=575 or prior.get('fitted_edges')!=1502
                or prior.get('source_lr_repaired') is not False):
            raise ValueError('Valid recorded E87/E89 source and fit evidence required')
        if own_leg_body:
            review=json.loads((source/'parent-review.json').read_text())
            if review.get('blocking_defects')!=[] or not review.get('checks',{}).get('source_time_and_rates_unaltered'):
                raise ValueError('Accepted E89 parent source review required')
        fitted=np.load(source/'calibration.npz')
        source_cal=np.load(root/'e86-native-source-cadence/calibration.npz')
        for key in source_cal.files:
            if key not in ('values','theta'):
                exact(source_cal[key],fitted[key],'Fixed E86 calibration '+key)
        if (len(fitted['tau_ids'])!=32 or len(fitted['theta_source_ids'])!=47
                or fitted['theta'].shape!=(96,47)):
            raise ValueError('Exact E86 kinetics and E87 source parameter shape required')
    initial=StateStore.load(source/'candidate-neutral')
    original=StateStore.load(root/'e32-core-adaptation/candidate/start')
    exact(original['body'],initial['body'],'Original M2 neutral physical initial state')
    for key in ('rate','adaptation','drive','output_mask','tick'):
        exact(original['network'][key],initial['network'][key],'Original M2 neural initial '+key)
    s=IntrinsicRecruitmentTrial.from_checkpoint(graph,initial)
    if s.network.device!='cuda':
        s.close()
        raise ValueError('Actual CUDA required')
    if coupling_phase_body or rm_e2_body or e98_body or e100_body or e102_body or e105_body:
        base_weights,_=rate_weights(graph)
        base_weights=BancTargetEncoder(graph,forward_drive=5000).calibrate_weights(
            base_weights,fitted['motor_ids'])
        expected_weights=patch_weights(base_weights,fitted['edge_ids'],
            fitted['source_values'],fitted['values'])
        actual_weights=source_matrix(s.network)
        try:
            exact(expected_weights.indptr,actual_weights.indptr,'E94 full CSR indptr')
            exact(expected_weights.indices,actual_weights.indices,'E94 full CSR indices')
            exact(expected_weights.data,actual_weights.data,'E94 full current weights')
        except Exception:
            s.close()
            raise
    cal=np.load(source/'calibration.npz')
    source_ids=np.r_[cal['core_ids'],cal['intrinsic_tonic_ids'],cal['coupling_ids']]
    extra_source=np.setdiff1d(cal['input_ids'],source_ids)
    observed=np.r_[source_ids,extra_source,cal['motor_ids']]
    dns=np.array(graph.resolve(['flywire:banc:888:720575941500851362','flywire:banc:888:720575941626500746']))
    observed=np.r_[observed,dns]
    if len(np.unique(observed))!=len(observed):
        s.close()
        raise ValueError('Distinct observed anatomical cells required')
    if all_leg_development:
        exact(np.unique(np.r_[cal['input_ids'],cal['intrinsic_tonic_ids'],cal['coupling_ids']]),
            np.sort(np.r_[source_ids,extra_source]),'All47 E86 source cells in body observation')
        expected_source_count=49 if e100_body or e102_body or e105_body else (48 if rm_e2_body or e98_body else 47)
        if (len(np.unique(np.r_[source_ids,extra_source]))!=expected_source_count
                or len(observed)!=expected_source_count+391+2):
            s.close()
            raise ValueError('Actual selected source+391motor+2DNg100 observation required')
    recruitment_ids = None
    recruitment_signals = []
    if recruitment_observation:
        relay_roots=('720575941526736740','720575941518807468',
            '720575941390358788')
        if coupling_phase_body or rm_e2_body or e98_body or e100_body or e102_body or e105_body:
            relay_roots+=('720575941504444354','720575941626267786',
                '720575941613448668','720575941595919217')
        if e98_body or e100_body or e102_body or e105_body:
            relay_roots+=('720575941511186508',)
        relay_ids = np.array(graph.resolve(['flywire:banc:888:' + rid
            for rid in relay_roots]))
        if coupling_phase_body or rm_e2_body or e98_body or e100_body or e102_body or e105_body:
            selected_rm=relay_ids[-5:-1] if e98_body or e100_body or e102_body or e105_body else relay_ids[-4:]
            exact(selected_rm,np.array([54014,136078,131125,122032],
                dtype=relay_ids.dtype),
                'Four existing RM relay graph identities')
        if e98_body or e100_body or e102_body or e105_body:
            exact(relay_ids[-1:],np.array([60038],dtype=relay_ids.dtype),
                'LF IN19B003 observed identity')
        recruitment_ids = np.r_[observed, np.setdiff1d(relay_ids, observed)]
        if (e100_body or e102_body or e105_body) and len(recruitment_ids)!=446:
            s.close()
            raise ValueError('E100 requires exactly446 unique detailed observed cells')
        if e102_body or e105_body:
            previous_meta=np.load(root/'e100-lf-tarsal-native-body-v2/recruitment-metadata.npz')
            previous_basic=np.load(root/'e100-lf-tarsal-native-body-v2/signals.npz')
            exact(recruitment_ids,previous_meta['observed_ids'],
                'E102 same446 detailed graph IDs and order as E100')
            exact(observed,previous_basic['observed_ids'],
                'E102 same442 basic graph IDs and order as E100')
        if len(np.unique(recruitment_ids)) != len(recruitment_ids):
            s.close()
            raise ValueError('New relay observations must not duplicate existing observations')
        recruitment_sources = np.r_[observed[:-len(cal['motor_ids'])-2], dns,
            np.setdiff1d(relay_ids, observed)]
        if (e100_body or e102_body or e105_body) and (len(recruitment_sources)!=55
                or len(np.unique(recruitment_sources))!=55):
            s.close()
            raise ValueError('E100 requires55 unique source/DN/relay metadata identities')
        if e102_body or e105_body:
            exact(recruitment_sources,previous_meta['source_ids'],
                'E102 same55 source metadata graph IDs and order as E100')
        weights = source_matrix(s.network)
        np.savez_compressed(out/'recruitment-metadata.npz', observed_ids=recruitment_ids,
            motor_ids=cal['motor_ids'], source_ids=recruitment_sources,
            source_to_motor=weights[cal['motor_ids']][:,recruitment_sources].toarray(),
            **{key:getattr(s.network,key).cpu().numpy()[recruitment_ids].copy()
               for key in ('tau','a','threshold','cap','_cell_gain')})
    body=s.body
    native_step=body._step_physics
    shadow=copy.copy(body)
    shadow.d=body.mj.MjData(body.m)
    integration_type=body.mj.mjtState.mjSTATE_INTEGRATION
    integration_state=np.empty(body.mj.mj_stateSize(body.m,integration_type))
    native=[]
    def observe_step():
        native_step()
        body.mj.mj_getState(body.m,body.d,integration_state,integration_type)
        body.mj.mj_setState(body.m,shadow.d,integration_state,integration_type)
        body.mj.mj_forward(body.m,shadow.d)
        feet=contact_diagnostics(shadow)
        _,rotation,_=shadow.pose()
        native.append((float(body.physics_time()),feet['normal_load_bw'],
            [np.nan if v is None else v for v in feet['slip_mm_s']],
            bool(shadow.nonfoot_contact()),float(rotation[2,2])))
    body._step_physics=observe_step
    rows,signals,dn_signals=[],[],[]
    model_s=0.;future=None;future_pass=False;reason='duration'
    try:
        StateStore.save(out/'start',s.snapshot())
        rows.append(capture_control(s))
        with (out/'trace.jsonl').open('w',encoding='utf8') as stream:
            stream.write(json.dumps(rows[0])+'\n')
            for tick in range(1,controls+1):
                if time.perf_counter()-began>(175 if source_basis_observation else 280):
                    reason='wall_limit';break
                old_native=len(native)
                s.advance(1);model_s+=.005
                rows.append(capture_control(s));stream.write(json.dumps(rows[-1])+'\n')
                signals.append(s.network.readout(observed))
                current = s.network._current(s.network.rate)
                dn_signals.append({'drive':s.network.drive[dns].cpu().numpy().copy(),
                    'recurrent_current':current[dns].cpu().numpy().copy(),
                    'adaptation':s.network.adaptation[dns].cpu().numpy().copy()})
                if recruitment_observation:
                    recruitment_signals.append({'rate':s.network.readout(recruitment_ids),
                        'recurrent_current':current[recruitment_ids].cpu().numpy().copy(),
                        'drive':s.network.drive[recruitment_ids].cpu().numpy().copy(),
                        'adaptation':s.network.adaptation[recruitment_ids].cpu().numpy().copy(),
                        'muscle_activation':s.adapter.activation.copy()})
                if len(native)-old_native!=50 or s.network.tick!=tick*20:
                    raise ValueError('Native neural/physical clocks and50substeps required')
                if s.fault or body.fault or any(row[3] or row[4]<.9 for row in native[old_native:]):
                    reason='health_guard';break
                if tick==200:
                    StateStore.save(out/'checkpoint-1s',s.snapshot())
                    clone=IntrinsicRecruitmentTrial.from_checkpoint(graph,StateStore.load(out/'checkpoint-1s'))
                    try:
                        clone.advance(20);model_s+=.1;future=clone.snapshot()
                    finally:
                        clone.close()
                if tick==220:
                    exact(future,s.snapshot(),'Intrinsic movingbody disk future')
                    future_pass=True
                if tick%200==0:
                    print(json.dumps({'time_s':tick*.005,'signed_forward_mm':s.signed_forward_mm,
                        'upright':rows[-1]['upright'],'wall_s':time.perf_counter()-began}),flush=True)
        if not s.fault and not body.fault:
            StateStore.save(out/'final',s.snapshot())
        physical={'controls':len(rows)-1,'physical_s':(len(rows)-1)*.005,'signed_forward_mm':s.signed_forward_mm,
            'minimum_upright':min((r[4] for r in native),default=rows[0]['upright']),
            'native_nonfoot_samples':sum(r[3] for r in native),'native_samples':len(native),
            'fault':s.fault,'body_fault':body.fault,'stop_reason':reason,
            'moving_disk_future_exact':future_pass,'neural_physical_model_s':model_s,
            'device':s.network.device,'wall_s':time.perf_counter()-began}
        write_json(out/'physical-result.json',physical)
    finally:
        write_json(out/'trace.json',{'schema':'flylab.m3-intrinsic-body-trace.v1',
            'controller':'Full BANC source intrinsic neural dynamics with actualbodyfeedback; no teacher',
            'graph_hash':graph.hash,'model_hash':s.network.identity,'body_hash':body.model_hash,
            'adapter_hash':s.adapter.identity,'parameters':{**initial['parameters'],'diagnostic':spec},'samples':rows})
        if signals:
            np.savez_compressed(out/'signals.npz',time_s=np.arange(1,len(signals)+1)*.005,
                observed_ids=observed,rate=signals,DN_ids=dns,
                **{'DN_'+k:np.array([row[k] for row in dn_signals]) for k in dn_signals[0]})
        if native:
            np.savez_compressed(out/'native-contact.npz',time_s=[r[0] for r in native],
                normal_load_bw=[r[1] for r in native],slip_mm_s=[r[2] for r in native],
                nonfoot_contact=[r[3] for r in native],upright=[r[4] for r in native])
        if recruitment_signals:
            np.savez_compressed(out/'recruitment-signals.npz',
                time_s=np.arange(1,len(recruitment_signals)+1)*.005,
                observed_ids=recruitment_ids,
                **{k:np.array([row[k] for row in recruitment_signals]) for k in recruitment_signals[0]})
        s.close()
    if exact_recruitment_observation:
        comparison=root/'e73-front-knee-delivery-body'
        baseline=json.loads((comparison/'trace.json').read_text())['samples']
        for tick,row in enumerate(rows):
            for key in ('qpos','qvel','ctrl','joint_targets_rad','tendon_inputs'):
                exact(baseline[tick][key],row[key],f'E73control{tick} {key}')
        for name in ('signals.npz','native-contact.npz'):
            previous=np.load(comparison/name)
            actual=np.load(out/name)
            for key in actual.files:
                expected=previous[key]
                if key not in ('observed_ids','DN_ids'):
                    expected=expected[:len(actual[key])]
                exact(expected,actual[key],f'E73prefix {name} {key}')
        for entry in recruitment_signals:
            if any(not np.isfinite(value).all() for value in entry.values()):
                raise ValueError('Finite recruitment observations required')
        write_json(out/'exact-prefix.json',{'validity':'valid','controls':len(rows)-1,
            'observed_rate_cells':len(observed),'native_samples':len(native),
            'all_physical_source_DN_and_native_arrays_bit_exact':True,
            'recruitment_values_finite':True,'reference':str(comparison)})
    quality=None
    if len(rows)>102:
        if all_leg_development:
            try:
                quality,_,_=analyze(load_model(out/'trace.json'))
            except (ValueError,KeyError,IndexError,TypeError) as error:
                write_json(out/'quality-analysis-error.json',{'error':type(error).__name__,
                    'message':str(error),'all_numeric_screens':False})
        else:
            quality,_,_=analyze(load_model(out/'trace.json'))
    if all_leg_development:
        times=np.array([row['time_s'] for row in rows])
        positions=np.array([row['body']['position'] for row in rows])
        window=(times>=.5)&(times<10)
        forward=np.asarray(rows[0]['body']['basis'],dtype=float)[[0,2],0]
        forward/=np.linalg.norm(forward)
        if window.sum()>=2:
            planar=positions[window][:,[0,2]]
            delta=planar[-1]-planar[0]
            path=float(np.linalg.norm(np.diff(planar,axis=0),axis=1).sum())
            net=float(np.linalg.norm(delta))
            fixed_forward=float(delta@forward)
        else:
            path=net=fixed_forward=0.
        ratio=net/path if path>0 else 0.
        health=(reason=='duration' and len(rows)==2001 and len(native)==100000
            and sum(r[3] for r in native)==0 and physical['minimum_upright']>=.9
            and not s.fault and not body.fault)
        events=quality.get('events',{}) if isinstance(quality,dict) else {}
        pairs=quality.get('pair_phase',{}) if isinstance(quality,dict) else {}
        knees=quality.get('knee_interior_deg',{}) if isinstance(quality,dict) else {}
        legs=('lf','lm','lh','rf','rm','rh')
        pair_names=('lf-rf','lm-rm','lh-rh')
        cycles={leg:events.get(leg,{}).get('complete_cycles') for leg in legs}
        cvs={leg:events.get(leg,{}).get('period_cv') for leg in legs}
        phase={key:{field:pairs.get(key,{}).get(field) for field in
            ('mean_cycles','resultant_length','samples')} for key in pair_names}
        widths={leg:(float(knees[leg]['p95']-knees[leg]['p05'])
            if isinstance(knees.get(leg),dict) and knees[leg].get('p95') is not None
            and knees[leg].get('p05') is not None else None) for leg in legs}
        f3=bool(health and fixed_forward>=10 and ratio>=.5)
        f4=bool(health and all(cycles[leg] is not None and cycles[leg]>=10 for leg in legs)
            and all(phase[key]['mean_cycles'] is not None
                and phase[key]['resultant_length'] is not None
                and abs((phase[key]['mean_cycles']-.5+.5)%1-.5)<=.15
                and phase[key]['resultant_length']>=.5 for key in pair_names))
        f5=bool(health and all(widths[leg] is not None and widths[leg]>=limit
            for leg,limit in {'lf':25,'rf':30,'lh':40,'rh':35}.items())
            and all(cvs[leg] is not None and cvs[leg]<=.3 for leg in legs))
        quality_summary={'fixed_initial_heading_net_forward_mm':fixed_forward,
            'net_displacement_mm':net,'planar_path_mm':path,'net_over_path':ratio,
            'complete_10s_native_health':health,
            'native_nonfoot_samples':physical['native_nonfoot_samples'],
            'minimum_native_upright':physical['minimum_upright'],
            'forward_eligible_complete_cycles':cycles,'physical_LR':phase,
            'knee_width_deg':widths,'period_cv':cvs,
            'development_screens':{'F3_numeric':f3,'F4_numeric':f4,'F5_numeric':f5},
            'F6_unused_poses_tested':False,'final_adoption':False}
        write_json(out/'quality-summary.json',quality_summary)
    source_phase_summary=None
    if coupling_phase_body or rm_e2_body or e98_body or e100_body or e102_body or e105_body:
        try:
            sample=np.load(out/'recruitment-signals.npz')
            source_rate=np.vstack((np.zeros((1,len(recruitment_ids))),sample['rate']))
            source_time=np.arange(len(source_rate))*.005
            e1=np.array([int(cal['core_ids'][(cal['core_groups']==group)
                & (cal['core_cell_types']==0)][0]) for group in range(6)])
            e1_columns=np.array([int(np.flatnonzero(recruitment_ids==cell)[0])
                for cell in e1])
            completed=complete_source_trace(sample['time_s'],sample['rate'],
                sample['observed_ids'],recruitment_ids,physical['controls'],reason)
            period=float(np.load(root/'e85-space-compatible-cadence-reference/commands.npz')['period_s'])
            source_rhythms,source_pairs=source_phase_screens(
                source_time,source_rate,e1_columns,period)
            source_phase_summary={'E1_ids':e1.tolist(),
                'rhythms':source_rhythms,'LR_by_window':source_pairs,
                'complete_10s_neural_trace':completed,
                'partial_window_screen':bool(all(item['screen'] for item in source_rhythms)
                    and all(item['screen'] for item in source_pairs.values())),
                'source_screen':bool(completed
                    and all(item['screen'] for item in source_rhythms)
                    and all(item['screen'] for item in source_pairs.values())),
                'sample_dt_s':.005,'native_neural_dt_s':.00025,
                'interpretation':'Actual whole-network/body source diagnostic; separate from physical F screens'}
        except (ValueError,KeyError,IndexError,TypeError,FileNotFoundError) as error:
            source_phase_summary={'source_screen':False,
                'complete_10s_neural_trace':False,
                'error':type(error).__name__,'message':str(error)}
        write_json(out/'source-phase-summary.json',source_phase_summary)
    lf_source_summary=None
    if e100_body or e102_body or e105_body:
        try:
            sample=np.load(out/'recruitment-signals.npz')
            metadata=np.load(out/'recruitment-metadata.npz')
            warm=np.load(root/'e89-own-leg-native-recruitment/warm-start.npz')
            anatomy=json.loads((source/'muscle-provenance.json').read_text())
            ids=sample['observed_ids']
            source_roster=metadata['source_ids']
            selected_sources=np.unique(np.r_[cal['input_ids'],cal['intrinsic_tonic_ids'],
                cal['coupling_ids']])
            if (len(ids)!=446 or len(source_roster)!=55
                    or len(np.unique(source_roster))!=55
                    or not np.all(np.isin(selected_sources,source_roster))
                    or warm['warm_muscles'].shape!=(84,96)
                    or anatomy['muscles'][14]['leg']!='lf'
                    or anatomy['muscles'][14]['muscle']!='tarsus_depressor_muscle'
                    or anatomy['muscles'][16]['leg']!='lf'
                    or anatomy['muscles'][16]['muscle']!='tarsus_levator_muscle'):
                raise ValueError('E100 55 source metadata and frozen LF muscles required')
            lookup={int(cell):int(np.flatnonzero(ids==cell)[0])
                for cell in (60038,11088,29608,105382,53861)}
            full_time=np.r_[0.,sample['time_s']]
            rate=np.vstack((np.zeros((1,len(ids))),sample['rate']))
            for cell in lookup:
                if initial['network']['rate'][cell]!=0:
                    raise ValueError('E100 LF source t0 must be actual original zero')
            lf_peaks=quadratic_peaks(full_time,rate[:,lookup[53861]])
            lf_phase=phase_trace(full_time,lf_peaks)
            target=np.full((len(full_time),2),np.nan)
            bracketed=np.isfinite(lf_phase)
            bins=warm['reference_phase_bins']
            for column,muscle in enumerate((14,16)):
                inverse=warm['warm_muscles'][:,muscle]
                target[bracketed,column]=np.interp(np.mod(lf_phase[bracketed],1),
                    np.r_[bins,1],np.r_[inverse,inverse[0]])
            actual_pair=rate[:,[lookup[60038],lookup[11088]]]
            actual_presyn=rate[:,[lookup[29608],lookup[105382]]]
            np.savez_compressed(out/'actual-LF-source-target-raw.npz',time_s=full_time,
                source_ids=np.array([60038,11088]),source_rate=actual_pair,
                presynaptic_ids=np.array([29608,105382]),presynaptic_rate=actual_presyn,
                LF_E1_id=53861,LF_E1_peak_times_s=lf_peaks,LF_phase=lf_phase,
                inverse_target_muscle_ids=np.array([14,16]),inverse_target=target)
            complete=bool(source_phase_summary['source_screen']
                and source_phase_summary['complete_10s_neural_trace']
                and quality_summary['complete_10s_native_health']
                and len(full_time)==2001)
            if not complete:
                raise ValueError('E100 incomplete native body or full source trajectory')
            period=float(np.load(root/'e85-space-compatible-cadence-reference/commands.npz')['period_s'])
            source_checks={
                'depressor_60038':lf_source_screen(full_time,actual_pair[:,0],
                    actual_presyn[:,0],target[:,0],period,sign=1),
                'levator_11088':lf_source_screen(full_time,actual_pair[:,1],
                    actual_presyn[:,1],target[:,1],period,sign=-1)}
            native_physiology={}
            for cell in (60038,11088):
                column=lookup[cell]
                gain=float(metadata['_cell_gain'][column])
                tau=float(metadata['tau'][column])
                drive_zero=bool(np.all(sample['drive'][:,column]==0))
                native_physiology[str(cell)]={'gain':gain,'tau_s':tau,'direct_drive_zero':drive_zero,
                    'screen':bool(gain==0 and tau==float(np.float32(.02)) and drive_zero)}
            lf_source_summary={'complete_full_source_and_native_health':complete,
                'screens':source_checks,'native_physiology':native_physiology,
                'source_preparation_screen':bool(all(x['screen'] for x in source_checks.values())
                    and all(x['screen'] for x in native_physiology.values())),
                'E97_shape_supported':False,'physical_F_separate':True,
                'target_alignment':'This actual run LF E1 quadratic peaks to fixed E89 84-bin inverse muscles14/16',
                'interpretation':'Actual source preparation screen only; not gait or adoption'}
        except (ValueError,KeyError,IndexError,TypeError,FileNotFoundError) as error:
            lf_source_summary={'source_preparation_screen':False,
                'error':type(error).__name__,'message':str(error),
                'E97_shape_supported':False}
        write_json(out/'actual-LF-source-summary.json',lf_source_summary)
    rm_source_summary=None
    if rm_e2_body:
        try:
            sample=np.load(out/'recruitment-signals.npz')
            metadata=np.load(out/'recruitment-metadata.npz')
            warm_start=np.load(root/'e89-own-leg-native-recruitment/warm-start.npz')
            own_phase=np.load(root/'e89-own-leg-native-recruitment/own-leg-phase.npz')
            anatomy=json.loads((source/'muscle-provenance.json').read_text())
            if (anatomy['muscles'][67]['leg']!='rm'
                    or anatomy['muscles'][67]['muscle']!='sternal_adductor_muscle'
                    or anatomy['muscles'][67]['columns']!=[364]
                    or int(np.flatnonzero(cal['motor_ids']==144018)[0])!=364
                    or warm_start['warm_muscles'].shape!=(84,96)
                    or len(warm_start['reference_phase_bins'])!=84
                    or own_phase['leg_order'][4]!='rm'
                    or own_phase['offsets'][4]!=0):
                raise ValueError('Original RM sternal target muscle provenance required')
            ids=sample['observed_ids']
            relay_col=int(np.flatnonzero(ids==54014)[0])
            e2_col=int(np.flatnonzero(ids==43618)[0])
            mask=(sample['time_s']>=2.5)&(sample['time_s']<9.5)
            relay=sample['rate'][mask,relay_col]
            e2=sample['rate'][mask,e2_col]
            rm_e1_col=int(np.flatnonzero(ids==97727)[0])
            full_times=np.r_[0.,sample['time_s']]
            full_rm_e1=np.r_[0.,sample['rate'][:,rm_e1_col]]
            bins=warm_start['reference_phase_bins']
            warm_muscles=warm_start['warm_muscles']
            warm,_,_=own_rm_inverse_at_actual_phase(
                full_times,full_rm_e1,np.r_[False,mask],bins,
                warm_muscles[:,67],own_phase['offsets'][4])
            def correlation(first,second):
                if len(first)<3 or np.std(first)==0 or np.std(second)==0:
                    return None
                return float(np.corrcoef(first,second)[0,1])
            width=float(np.percentile(relay,95)-np.percentile(relay,5))
            points=quadratic_peaks(sample['time_s'],sample['rate'][:,relay_col])
            within=points[(points>=2.5)&(points<9.5)]
            intervals=np.diff(within)
            mean=float(intervals.mean()) if len(intervals) else None
            cv=float(intervals.std()/intervals.mean()) if len(intervals) else None
            period=float(np.load(root/'e85-space-compatible-cadence-reference/commands.npz')['period_s'])
            drive_zero=bool(np.all(sample['drive'][:,relay_col]==0))
            native_gain=float(metadata['_cell_gain'][relay_col])
            native_tau=float(metadata['tau'][relay_col])
            rm_e2_r=correlation(relay,e2)
            target_r=correlation(relay,warm)
            full=bool(source_phase_summary['source_screen']
                and source_phase_summary['complete_10s_neural_trace']
                and quality_summary['complete_10s_native_health'])
            supported=bool(full and width>=20 and rm_e2_r is not None
                and rm_e2_r>=.8 and len(intervals)>=12 and cv is not None
                and cv<=.3 and mean is not None and abs(mean/period-1)<=.1
                and target_r is not None and target_r>=.5 and drive_zero
                and native_gain==0 and native_tau==float(np.float32(.02)))
            rm_source_summary={'full_source_health':full,
                'new_relay_width_p95_p05':width,'RM_E2_Pearson':rm_e2_r,
                'own_RM_sternal_inverse_target_Pearson':target_r,
                'complete_cycles':len(intervals),'period_mean_s':mean,
                'period_CV':cv,'direct_drive_exact_zero':drive_zero,
                'native_gain':native_gain,'native_tau_s':native_tau,
                'source_preparation_screen':supported,
                'target_muscle_index':67,'mask_samples':int(mask.sum()),
                'window_s':[2.5,9.5],
                'target_alignment':'E95 actual RM E1 quadratic peaks to E89 fixed84-bin inverse with E85 RM offset0',
                'interpretation':'Actual source preparation only; physical F and adoption separate'}
        except (ValueError,KeyError,IndexError,TypeError,FileNotFoundError) as error:
            rm_source_summary={'source_preparation_screen':False,
                'error':type(error).__name__,'message':str(error)}
        write_json(out/'rm-source-summary.json',rm_source_summary)
    e105_startup=None
    if e105_body:
        predicted=np.load(root/'e104-constrained-startup-recruitment/prediction.npz')
        transfer,times99,q99,ltm99,dq,dltm=e105_startup_diagnostic(rows,native,predicted)
        np.savez_compressed(out/'actual-startup-vs-E104-raw.npz',time_s=times99,
            actual_command=q99,actual_LTM=ltm99,
            predicted_command=predicted['command'][1:len(q99)+1],
            predicted_LTM=predicted['tendon'][1:len(ltm99)+1],
            command_difference=dq,LTM_difference=dltm)
        first100=bool(len(rows)>=101 and len(native)>=5000
            and np.array_equal(np.asarray([row['time_s'] for row in rows[1:101]]),
                np.arange(1,101)*.005))
        first_native=native[:5000]
        startup_health=bool(first100 and all(not row[3] and row[4]>=.9
            for row in first_native) and all(not row['fault'] for row in rows[1:101]))
        e105_startup={'complete_first_half_second_native_health':startup_health,
            'complete_first_100_controls':first100,
            'first_half_second_native_samples':len(first_native),
            'first_half_second_nonfoot_samples':sum(row[3] for row in first_native),
            'first_half_second_minimum_upright':min((row[4] for row in first_native),default=None),
            'actual_99_command_LTM_transfer':transfer,
            'E104_startup_supported':True,'E104_steady_supported':False,
            'E104_local_supported':False,'full_10s_native_health_separate':quality_summary['complete_10s_native_health'],
            'original_physical_F3_F5_separate':quality_summary['development_screens']}
        write_json(out/'startup-diagnostic.json',e105_startup)
    result={'experiment':spec['experiment'],'validity':'valid_full_BANC_physical_feedback_diagnosis',
        'hypothesis_verdict':'requires_full_quality_review' if reason=='duration' else 'not_supported',
        'adoption':'not_applicable','milestone_outcome':'pending','runtime_external_teacher':False,
        'physical':physical,'quality':quality}
    if all_leg_development:
        result.update(quality_summary=quality_summary,actual_neural_s=model_s,
            actual_physical_s=model_s,
            hypothesis_verdict='one_development_start_numeric_F3_F5' if all(
                quality_summary['development_screens'].values()) else 'not_supported')
    if coupling_phase_body:
        result['source_phase_summary']=source_phase_summary
    if e98_body:
        result['source_phase_summary']=source_phase_summary
        result['E97_shape_supported']=False
        result['preparation_status']='diagnostic_from_failed_conditional_shape'
        result['LF_IN19B003_observed_id']=60038
        result['diagnostic_cells']=len(recruitment_ids)
        result['direct_physical_discriminator']='all_F3_F4_F5_numeric' if all(
            quality_summary['development_screens'].values()) else 'not_all_F3_F4_F5_numeric'
    if e100_body:
        result['source_phase_summary']=source_phase_summary
        result['actual_LF_source_summary']=lf_source_summary
        result['E97_shape_supported']=False
        result['preparation_status']='reviewed_E99_existing_LF_sources'
        result['diagnostic_cells']=len(recruitment_ids)
        result['prior_E99_shared_wall_s']=e99_prior_wall
        result['prior_E100_failed_execution_wall_s']=prior_execution_wall_s
        result['body_wall_s']=time.perf_counter()-began
        result['shared_E99_E100_wall_s']=e99_prior_wall+prior_execution_wall_s+result['body_wall_s']
        result['resource_limit_met']=bool(result['body_wall_s']<=285
            and result['shared_E99_E100_wall_s']<=375 and model_s<=10.1000001)
        if not result['resource_limit_met']:
            result['validity']='resource_limit_incomplete'
    if e102_body:
        result['source_phase_summary']=source_phase_summary
        result['actual_LF_source_summary']=lf_source_summary
        result['E101_conditional_shape_supported']=True
        result['diagnostic_cells']=len(recruitment_ids)
        result['prior_E101_fit_wall_s']=e101_prior_wall
        result['prior_E102_failed_execution_wall_s']=prior_execution_wall_s
        result['body_wall_s']=time.perf_counter()-began
        result['shared_E101_E102_wall_s']=e101_prior_wall+prior_execution_wall_s+result['body_wall_s']
        result['resource_limit_met']=bool(result['body_wall_s']<=285
            and result['shared_E101_E102_wall_s']<=375 and model_s<=10.1000001)
        if not result['resource_limit_met']:
            result['validity']='resource_limit_incomplete'
    if e105_body:
        result['source_phase_summary']=source_phase_summary
        result['actual_LF_source_summary']=lf_source_summary
        result['startup_diagnostic']=e105_startup
        result['E104_shape_supported']=False
        result['E104_local_supported']=False
        result['preparation_status']='valid_prepared_startup_diagnostic'
        result['diagnostic_cells']=len(recruitment_ids)
        result['prior_E104_wall_s']=41.41753739997512
        result['prior_E105_failed_execution_wall_s']=prior_execution_wall_s
        result['E105_preparation_wall_s']=e105_prior_wall
        result['body_wall_s']=time.perf_counter()-began
        result['shared_E104_E105_wall_s']=(result['prior_E104_wall_s']+
            e105_prior_wall+prior_execution_wall_s+result['body_wall_s'])
        result['resource_limit_met']=bool(result['body_wall_s']<=285
            and result['shared_E104_E105_wall_s']<=375 and model_s<=10.1000001)
        if not result['resource_limit_met']:
            result['validity']='resource_limit_incomplete'
    if rm_e2_body:
        result['source_phase_summary']=source_phase_summary
        result['RM_source_summary']=rm_source_summary
    write_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='quality'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','root','graph','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--experiment',default='M3-E64-intrinsic-source-moving-body')
    parser.add_argument('--prior-execution-wall-s',type=float,default=0.)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing experiment output: '+str(args.out))
    execution_started=time.perf_counter()
    try:
        run(args.source,args.root,args.graph,args.out,experiment=args.experiment,
            prior_execution_wall_s=args.prior_execution_wall_s)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'validity':'incomplete','error':type(error).__name__,
                'message':str(error),'hypothesis_verdict':'not_tested',
                'execution_wall_s':time.perf_counter()-execution_started})
        raise
