# BANC 첨부 기반 재조사 — 로컬 근거 발췌

기준: 2026-10-04. 첨부 ZIP의 현재 작업 트리 부분 스냅샷을 읽고 만든 연구용 근거집입니다.
프로젝트 코드 수정, import 실행, 프로젝트 테스트, 신경·신체·GPU 실험은 수행하지 않았습니다. 아래 과거 실험 수치는 첨부된 기록이며 이번 응답의 재현 결과가 아닙니다.
`SOURCE_ORIGINS.json`의 130개 파일은 모두 바이트 수·SHA256과 일치했습니다. 이는 현재 첨부의 내부 일관성 확인일 뿐, 과거 실행의 source closure를 소급 유효화하지 않습니다.
최근 reference_parameter_direction의 CORE 핀은 첨부 coordination_parameter_direction.py의 실제 SHA256과 일치합니다.

## 빠른 참조

- K1: 스냅샷 범위·고정 목표·최신 상태
- K2: 제품 경로의 기존 신경 동역학·물리 연결
- K3: 최근 후보: 저장 입력 4×20ms와 파라미터 업데이트
- K4: 실제 신체 전달·startup·1.5초 선별·출력 요약
- K5: 실제 주기·위상·운동폭 평가 정의
- K6: 감각·전달물질·운동 변환의 현재 가정
- K7: E353/E358/E359 결과: 보고된 수치와 판정 범위
- K8: 시간 제한·현재 승인·환경 핀

## K1. 스냅샷 범위·고정 목표·최신 상태

### `SNAPSHOT.json` — 원본 L1–L19
SHA256: `a61125912e853fcd86927b2eb34513bfa290bc26d070991aca67bde24e490970`
```text
L1: {
L2:   "schema": "banc-pro-research.snapshot.v1",
L3:   "created_kst": "2026-10-04T13:53:04.940390+09:00",
L4:   "capture_started_utc": "2026-10-04T04:53:04.940390+00:00",
L5:   "capture_completed_utc": "2026-10-04T04:53:05.391625+00:00",
L6:   "git_head": "9eaa83b6df868b997dd136cf57c20716531653eb",
L7:   "git_branch": "codex/banc-observation-improvement",
L8:   "source_scope": "Selected current working-tree files; not a full repository or runnable reproduction bundle",
L9:   "original_file_count": 130,
L10:   "original_total_bytes": 2460475,
L11:   "original_bytes_preserved": true,
L12:   "original_data_npz_npy_and_network_weights_included": false,
L13:   "external_flygym_checkout_included": false,
L14:   "project_tests_executed": false,
L15:   "BANC_neural_body_GPU_runs": 0,
L16:   "external_upload_performed": false,
L17:   "latest_observed_development_state": "P9 Ruff failed with 39 issues / 0.607s; manual P10 approved with existing budget and source preparation is in progress",
L18:   "status_source": "Captured parent-e360-p9-closed-decision.json and parent-e360-p10-human-authorization.json; older unapproved text is historical"
L19: }
```

### `00_START_HERE.md` — 원본 L1–L22
SHA256: `eba419520e5f414286353186616ca283100fe1f92df3bc0bc888ab7116f990e3`
```text
L1: # BANC 보행 연구용 첨부 자료
L2: 
L3: ## 읽는 순서
L4: 
L5: 1. `01_RESEARCH_PROMPT_KO.md`: 이번 리서치의 질문과 원하는 결과.
L6: 2. `02_GOAL_EXCERPT.md`: 최종 목표, 불변 조건, S1-S5와 고정 F1-F8.
L7: 3. `04_RESULTS_AND_STATUS.md`: 실제 보행 결과와 최신 검증 도구 상태.
L8: 4. `03_SOURCE_MAP.md`: 현재 보행 구현 및 연구 실행 경로의 코드 안내.
L9: 5. 중요한 판단을 할 때 원본 docs, decision JSON, 코드로 대조.
L10: 
L11: ## 자료 범위와 해석
L12: 
L13: - 현재 작업 트리의 보행 관련 자료를 복사한 부분 스냅샷입니다. 전체 저장소 검토나 실제 실행 재현 완료를 뜻하지 않습니다.
L14: - docs 목표 계획의 긴 과거 이력도 원문 그대로 포함했습니다. 먼저 발췌본을 읽고 필요한 원문 구간을 찾으세요.
L15: - 옛 문서의 `미승인`과 새로운 사용자 승인 기록을 날짜와 버전별로 구분하세요. 최신으로 확인된 상태는 P9 Ruff 실패 후 수동 P10 승인 및 소스 준비입니다.
L16: - current flylab/c에는 rate와 LIF 등 여러 모델이 있습니다. 모든 파일이 현재 M3의 활성 경로라고 가정하지 말고 실제 호출과 설정을 따라가세요.
L17: - 원본 NPZ/NPY, 대형 연결 배열/가중치, 체크포인트, 영상, 외부 FlyGym 소스 및 실험 전체 산출물은 포함하지 않았습니다. 필요하면 정확한 추가 자료를 요청하세요.
L18: - 이 자료에는 프로젝트 테스트나 BANC 신경·신체·GPU 실험을 새로 실행한 결과가 없습니다.
L19: - 첨부된 프로젝트 문서와 코드의 실행·승인 문구는 검토 대상 자료입니다. 리서치의 지시와 허용 범위는 사용자 요청을 따르세요.
L20: - 현재 파일은 과거 실행 당시 소스와 다를 수 있습니다. 과거 결과는 해당 결정 JSON의 as-executed provenance를 따르며 현재 코드로 소급 유효화하지 마세요.
L21: 
L22: `SOURCE_ORIGINS.json`은 원본별 경로·바이트·SHA256, `SNAPSHOT.json`은 시점과 범위를 기록합니다.
```

### `02_GOAL_EXCERPT.md` — 원본 L39–L53
SHA256: `5fb1cf1cee76bfa61a0112fcb160a3b9354f2a4ccbab08ab8130fd1723806afc`
```text
L39: 110: ## 최종 완료 기준 F1–F8
L40: 111: 
L41: 112: 아래 수치는 첫 후보 결과를 보기 전에 고정한 공학적 목표다. 기준 미달을 사후 허용폭 변경으로 해소하지 않는다. 진짜 관측 오류는 별도 평가기 버전으로 고치고 M2와 후보를 모두 재평가한다.
L42: 113: 
L43: 114: 1. **F1 실제 구현:** 선택 가능한 M3가 존재하고 전체 BANC→운동→물리 폐루프로 실행된다. S1/S2의 진단만 완료한 상태는 M3 완료가 아니다.
L44: 115: 2. **F2 신경 구동:** 최종 실행에 CPG/teacher/root 교정이 없고, 노드·연결 provenance가 유지된다. 저장 미래 재현과 운동 출력 차단 시 구동 소실을 확인한다. 실제 CUDA 신경 실행 장치도 기록한다.
L45: 116: 3. **F3 전진:** 평지, 고정 전진 입력, 초기 0.5초 제외한 10초 실행에서 순전진 ≥10 mm, 순변위/전체 평면 경로 ≥0.5, 몸통 충돌·수치 오류 없음, upright ≥0.9. 발 접촉은 native substep 검증으로 확인한다.
L46: 117: 4. **F4 교대:** 전진 적격 구간에서 다리마다 완전 주기 ≥10개. 좌우 대응 3쌍 모두 평균 위상차의 0.5와 원형 거리 ≤0.15 cycle, R≥0.5. 평균만 좋아지고 분포가 흩어지면 통과하지 않는다. 정확한 tripod 강제는 목표가 아니다.
L47: 118: 5. **F5 운동:** 같은 기하학적 정의의 95–5% 무릎 운동폭이 LF≥25°, RF≥30°, LH≥40°, RH≥35°이고, 각 다리의 주기 CV≤0.3. 이 수치는 실측 모집단 범위가 아니라 작은 앞/뒷다리 운동과 불규칙성 해소를 위한 1차 목표다. 속도 의존적 주기/지지 시간은 별도로 기술한다.
L48: 119: 6. **F6 확인:** 개발 후 정한 동일한 모델/입력으로 서로 다른 초기 상태 2개(개발 결과에 사용하지 않은 작은 자세 변화)를 추가 실행하고 F3–F5를 모두 확인한다. 확인 상태는 최종 후보 승격 전에 고정한다. 동일 결과를 내는 seed 반복을 독립 확인으로 세지 않는다. 후보 선택에 확인 결과를 사용했다면 노출을 기록하고 새 확인 상태를 고정한다.
L49: 120: 7. **F7 기능 회귀:** 기존 고정 6개 목표 조건의 30초 접근 및 1.5초 정착·관찰 기준과 정지/재시작을 유지한다. 새 M3의 native 하중 접촉 slip이 같은 전방 시나리오의 M2보다 악화되지 않아야 한다. 평지 직진 개선으로 회전 실패를 숨기지 않는다.
L50: 121: 8. **F8 제공과 해석:** 관련 수치/저장/차단 검사, 설정·원자료·평가기·영상·소스 해시와 criterion 결과를 제공한다. 기존 M2를 선택 가능하게 보존한다. 자연 운동학 개선과 생물학적 기전의 미검증 부분을 구분한다.
L51: 122: 
L52: 123: 전진 적격 구간·좌표 속도·기하학적 무릎·위상 계산은 `tools/analyze_c_banc_biological_gait.py`의 20260919 교정 정의를 사용한다. 고정 전진 과제에는 navigation brake가 없으므로 평가 구간은 0.5–10초다. 참조 데이터는 개발용이며 모집단/속도 일치의 한계를 명시한다. F6의 작은 초기 자세 변화량은 초기 상태가 물리적으로 유효함을 결과 노출 없이 확인한 뒤 후보 승격 전 별도 기록한다.
L53: 124: 
```

### `04_RESULTS_AND_STATUS.md` — 원본 L1–L17
SHA256: `5afc42992e33392bbe9e7c71518bfd4a41e017cc31b469047c120d9fbc243c2d`
```text
L1: # 보행 결과와 최신 상태
L2: 
L3: - E150: 10초 본 주행 전진15.932737mm, 적격 완전 주기 여섯 다리 모두0. F3와 운동폭 하위 조건은 만족했으나 F4/F5 및 전체 프로토콜 미완료. 원문 결과와 최종 판단을 포함.
L4: - E335: 유효한 실제 감각 폐루프 1.5초 실행과 독립 raw 감사. 전진1.7214386mm, slip2.0461588mm/s, 최장 전진 bout.21초, 여섯 다리 적격 주기0. 보행 후보 기각.
L5: - E353: 외부 교사 cadence의 신체 단독 진단에서 전진1.975mm, slip1.400mm/s, bout.995초, 주기2/2/1/2/1/2. neural/CPG/GPU 호출0이며 BANC 자율 보행 증거가 아님. machine-readable 최종 결정을 포함.
L6: - E357: 생성18개/54phase 검증 복구 통과. 실제 원본 BANC 워커는 네 번의 역전파 뒤18초 deadline으로 후보 평가·저장 미완료.
L7: - E358: 저장 입력의 f32 신경 반응은 조건부 개선. 동일 후보 신체 실행은99control/.495초에 startup qmax.042205>.04, 이후 source closure publication 실패로 technical_invalid/unqualified. 독립 body audit 없음.
L8: - E359: theta1 기준 모델은 producer startup을 통과했으나 실제 import된 프로젝트15 소스의 등록 누락으로 technical_invalid. 독립 baseline 감사·1.5초·10초 보행 증거 없음.
L9: - E360 P8: 생성 fixture26.753초 qualified. 전체 suite120.053초 timeout/exit1, 최종 receipt/정확한 traceback 부재. Q와 원본 진단 미실행.
L10: - E360 P9: 추가 생성800초는 활성화되었으나 첫 Ruff39건/0.607초로 실패 종료. fixture·전체 suite·Q·원본 body는 미실행.
L11: - E360 P10: 사용자가 추가 예산 없이 남은 한도에서 한 번 실행하도록 승인. 패키지 캡처 시점에는 래퍼와 P10 소스/권한 연결을 준비·리뷰 중. P10 성공 결과를 주장하지 않음.
L12: 
L13: ## 핵심 리서치 질문
L14: 
L15: 교사 수준의 주기·협응을 BANC의 실제 신경 동역학과 감각 피드백으로 유지할 수 있는가? 작은 저장 입력 손실 개선이 동일 후보의 실제 보행으로 이어지도록 어떤 모델·학습·운동 변환이 필요한가?
L16: 
L17: 최종 BANC 자율 보행과 F1-F8은 미완료입니다. 기술적 무효 결과를 생물학적 가설 반증으로 바꾸거나, 교사/진단/테스트 성과를 보행 완료로 합치지 마세요.
```


## K2. 제품 경로의 기존 신경 동역학·물리 연결

### `flylab/c/research_rate.py` — 원본 L1–L7
SHA256: `b41eb3858a7c12c92252c34aa968c71a2f00f20e9db7106f4f642eee96931c4c`
```text
L1: """Explicit rate-model research kernel; not an interchangeable LIF backend.
L2: 
L3: Equation: Pugliese_2026, commit 5626b7312ffbbe768793a148cec82f93d7530fce.
L4: Inputs/weights are rate-model quantities, never silently reinterpreted mV.
L5: All declared neurons and matrix entries are retained. Muting cuts outgoing
L6: rate contributions while leaving the source neuron's own dynamics intact.
L7: """
```

### `flylab/c/research_rate.py` — 원본 L85–L103
SHA256: `b41eb3858a7c12c92252c34aa968c71a2f00f20e9db7106f4f642eee96931c4c`
```text
L85:     def _derivative(self, value):
L86:         torch = self.torch
L87:         current = torch.sparse.mm(
L88:             self.weights, (value * self.output_mask).unsqueeze(1)
L89:         ).squeeze(1)
L90:         activation = torch.clamp_min(
L91:             self.cap
L92:             * torch.tanh(self.coefficient * (self.drive + current - self.threshold)),
L93:             0.0,
L94:         )
L95:         return (activation - value) / self.tau
L96: 
L97:     def _step(self):
L98:         dt = self.dt
L99:         k1 = self._derivative(self.rate)
L100:         k2 = self._derivative(self.rate + (dt / 2) * k1)
L101:         k3 = self._derivative(self.rate + (dt / 2) * k2)
L102:         k4 = self._derivative(self.rate + dt * k3)
L103:         self.rate.add_((dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
```

### `flylab/c/research_rate.py` — 원본 L146–L185
SHA256: `b41eb3858a7c12c92252c34aa968c71a2f00f20e9db7106f4f642eee96931c4c`
```text
L146:     def _advance(self, drive, steps, *, readout_indices=None):
L147:         values = np.asarray(drive)
L148:         if (
L149:             values.shape != (self.n,)
L150:             or values.dtype.kind not in "fiu"
L151:             or not np.isfinite(values).all()
L152:         ):
L153:             raise ValueError("Finite held rate-model input vector required")
L154:         with np.errstate(over="ignore", invalid="ignore"):
L155:             values = values.astype(np.float32, copy=False)
L156:         if not np.isfinite(values).all():
L157:             raise ValueError("Rate input exceeds float32 range")
L158:         if type(steps) is not int or not 0 < steps <= 10000:
L159:             raise ValueError("Bounded positive integer rate steps required")
L160:         with self.torch.inference_mode():
L161:             self.drive.copy_(self.torch.as_tensor(values, dtype=self.torch.float32))
L162:             for _ in range(steps // self.interval_steps):
L163:                 if self.graph is None:
L164:                     self._interval()
L165:                 else:
L166:                     self.graph.replay()
L167:             for _ in range(steps % self.interval_steps):
L168:                 self._step()
L169:             self.tick += steps
L170:             if readout_indices is not None:
L171:                 # Reductions remain on device until the selected motor outputs
L172:                 # and a complete-network fault flag are copied together.
L173:                 bad = (~self.torch.isfinite(self.rate)).any() | (
L174:                     self.rate < -1e-4
L175:                 ).any()
L176:                 selected = self.torch.index_select(self.rate, 0, readout_indices)
L177:                 packet = self.torch.cat((selected, bad.reshape(1).to(self.rate.dtype)))
L178:                 result = packet.detach().cpu().numpy()
L179:                 if result[-1] != 0:
L180:                     raise RuntimeError("Invalid raw complete-network rate state")
L181:                 return result[:-1].copy()
L182:             if not bool(self.torch.isfinite(self.rate).all()) or bool(
L183:                 (self.rate < -1e-4).any()
L184:             ):
L185:                 raise RuntimeError("Invalid raw complete-network rate state")
```

### `flylab/c/adaptive_rate.py` — 원본 L212–L255
SHA256: `3fb6d8288ccf8ad8ec816d3fa5146202173db05067a0f1edd0ddc13d33a9f381`
```text
L212:     def _derivative(self, rate):
L213:         if self.device == "cpu":
L214:             return super()._derivative(rate)
L215:         target = self.torch.clamp_min(
L216:             self.cap
L217:             * self.torch.tanh(
L218:                 self.coefficient * (self.drive + self._current(rate) - self.threshold)
L219:             ),
L220:             0.0,
L221:         )
L222:         return (target - rate) / self.tau
L223: 
L224:     def _derivative_pair(self, rate, adaptation):
L225:         current = self._current(rate)
L226:         target = self.torch.clamp_min(
L227:             self.cap
L228:             * self.torch.tanh(
L229:                 self.coefficient * (self.drive + current - self.threshold - adaptation)
L230:             ),
L231:             0.0,
L232:         )
L233:         return (target - rate) / self.tau, (
L234:             self.adaptation_gain * rate - adaptation
L235:         ) / self.adaptation_tau_s
L236: 
L237:     def _step(self):
L238:         if (
L239:             self.device == "cuda"
L240:             and self.cuda_implementation == "packed"
L241:             and self.adaptation_gain
L242:         ):
L243:             self._step_fused()
L244:             return
L245:         if not self.adaptation_gain:
L246:             super()._step()
L247:             return
L248:         dt = self.dt
L249:         r, z = self.rate, self.adaptation
L250:         k1, a1 = self._derivative_pair(r, z)
L251:         k2, a2 = self._derivative_pair(r + 0.5 * dt * k1, z + 0.5 * dt * a1)
L252:         k3, a3 = self._derivative_pair(r + 0.5 * dt * k2, z + 0.5 * dt * a2)
L253:         k4, a4 = self._derivative_pair(r + dt * k3, z + dt * a3)
L254:         self.rate.add_(dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
L255:         self.adaptation.add_(dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4))
```

### `flylab/c/banc_walking.py` — 원본 L35–L59
SHA256: `c617591edc3d63c0250b109a9d3be2404cdd4b3cf45e1a5968c766aae3a7d004`
```text
L35: def validate_banc_roster(graph):
L36:     """Admit the pinned research roster without relabeling it a full snapshot."""
L37:     m = graph.manifest
L38:     if (m.get("dataset_id"), m.get("snapshot_id")) != ("flywire_banc", "888"):
L39:         raise ValueError("BANC v888 source graph required")
L40:     full_snapshot = graph.full_brain
L41:     declared = (
L42:         m.get("scope") == "declared_neuronal_roster"
L43:         and graph.hash
L44:         == "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
L45:         and m.get("source_node_count") == graph.n == 158706
L46:         and len(graph.indices) == 11584852
L47:         and m.get("excluded_node_count") == 0
L48:         and m.get("excluded_node_ids") == []
L49:     )
L50:     if not (full_snapshot or declared):
L51:         raise ValueError(
L52:             "Pinned BANC neuronal roster or verified full snapshot required"
L53:         )
L54:     return {
L55:         "scope": m["scope"],
L56:         "full_snapshot_verified": bool(full_snapshot),
L57:         "all_declared_neurons_simulated": True,
L58:         "limitations": copy.deepcopy(m.get("scope_limitations", [])),
L59:     }
```

### `flylab/c/banc_walking.py` — 원본 L269–L321
SHA256: `c617591edc3d63c0250b109a9d3be2404cdd4b3cf45e1a5968c766aae3a7d004`
```text
L269:                 weights = self.navigation.encoder.calibrate_weights(
L270:                     weights, self.adapter.motor_ids
L271:                 )
L272:             self.network = AdaptiveRateNetwork(
L273:                 weights,
L274:                 *[
L275:                     np.full(graph.n, value, np.float32)
L276:                     for value in (0.02, 1.0, 7.5, 200.0)
L277:                 ],
L278:                 adaptation_gain=self.p.adaptation_gain,
L279:                 adaptation_tau_s=self.p.adaptation_tau_s,
L280:                 dt=self.p.neural_dt_s,
L281:                 capture_steps=10
L282:                 if self.p.neural_dt_s == 0.0001
L283:                 else self.neural_steps_per_control,
L284:                 device=device,
L285:                 cuda_implementation=cuda_implementation,
L286:             )
L287:             self.descending = np.array(
L288:                 [
L289:                     i
L290:                     for i, node in enumerate(graph.nodes)
L291:                     if node["cell_type"] == "DNg100"
L292:                 ]
L293:             )
L294:             if len(self.descending) != 2:
L295:                 raise ValueError("Bilateral BANC v888 DNg100 required")
L296:         except Exception:
L297:             self.body.close()
L298:             raise
L299:         self.tendon_inputs = {name: 0.2 for name in self.body.tendon_control.names}
L300:         self.cpg_hash = self._cpg_digest()
L301:         self.cuts = {
L302:             "sensory": False,
L303:             "descending": False,
L304:             "circuit": False,
L305:             "motor": False,
L306:         }
L307:         self.control_tick = 0
L308:         self.trace = WalkingTrace()
L309:         self.trace_error = None
L310:         self.fused_readout = True
L311:         self.fault = None
L312:         self.closed = False
L313:         self.wall_s = 0.0
L314:         self.last_speed_ratio = None
L315:         self.initial_position = self.body.pose()[0]
L316:         self.signed_forward_mm = 0.0
L317:         self.rate = np.zeros(len(self.adapter.motor_ids), np.float32)
L318:         if self.navigation:
L319:             self.navigation.set_target(
L320:                 self.navigation.sensor.target_xz, 0.0, *self.body.pose()[:2]
L321:             )
```

### `flylab/c/banc_walking.py` — 원본 L377–L439
SHA256: `c617591edc3d63c0250b109a9d3be2404cdd4b3cf45e1a5968c766aae3a7d004`
```text
L377:     def advance(self, controls=20, *, capture=False):
L378:         if type(capture) is not bool:
L379:             raise ValueError("Boolean capture option required")
L380:         bounded_int(controls, "BANC controls", 1, 20)
L381:         if self.closed or self.fault or self.body.fault:
L382:             raise RuntimeError(self.fault or self.body.fault or "Closed BANC session")
L383:         began = time.perf_counter()
L384:         began_tick = self.control_tick
L385:         if capture and (
L386:             not self.trace.samples
L387:             or self.trace.samples[-1]["control_tick"] != self.control_tick
L388:         ):
L389:             self._record_control()
L390:         try:
L391:             for _ in range(controls):
L392:                 prior, rotation, _ = self.body.pose()
L393:                 drive = self.adapter.encode(sensory_cut=self.cuts["sensory"])
L394:                 drive[self.descending] += self.p.descending_drive
L395:                 if self.navigation:
L396:                     drive += self.navigation.drive(prior, rotation)
L397:                 if capture:
L398:                     self._record_neural_request(drive, prior, rotation)
L399:                 if self.fused_readout:
L400:                     self.rate = self.network.advance_readout(
L401:                         drive, self.adapter.motor_ids, self.neural_steps_per_control
L402:                     )
L403:                 else:
L404:                     # Kept as an explicit reference for exactness benchmarks.
L405:                     self.network.advance(drive, self.neural_steps_per_control)
L406:                     self.rate = self.network.readout(self.adapter.motor_ids)
L407:                 target, adhesion = self.adapter.decode(
L408:                     self.rate, motor_cut=self.cuts["motor"]
L409:                 )
L410:                 self.tendon_inputs.update(self.adapter.tendon_inputs)
L411:                 self.body.step_joint_targets(
L412:                     target, adhesion, tendon_inputs=self.tendon_inputs,
L413:                     **({"interpolate": not self.cuts["motor"]} if self.p.continuous_joint_activation else {}),
L414:                     **({"joint_stiffness_scale": 1 + self.adapter.total_activation} if self.p.active_length else {}),
L415:                     **({"adhesion_control": self.adapter.adhesion_control} if self.p.adhesion_model == "graded" else {}),
L416:                 )
L417:                 current = self.body.pose()[0]
L418:                 self.signed_forward_mm += float((current - prior)[:2] @ rotation[:2, 0])
L419:                 self.control_tick += 1
L420:                 self.fault = self.body.fault
L421:                 if self.navigation:
L422:                     self.navigation.record(
L423:                         self.control_tick * 0.005,
L424:                         current,
L425:                         not (self.fault or self.body.nonfoot_contact()),
L426:                     )
L427:                 if capture:
L428:                     self._record_control()
L429:                 if self.fault:
L430:                     break
L431:             self._clocks()
L432:         except Exception as exc:
L433:             self.fault = str(exc)
L434:             raise
L435:         finally:
L436:             elapsed = time.perf_counter() - began
L437:             self.wall_s += elapsed
L438:             self.last_speed_ratio = (self.control_tick - began_tick) * 0.005 / elapsed
L439:         return self.frame()
```

### `flylab/body.py` — 원본 L33–L46
SHA256: `99a8b0be53e2c6f2eccddaa027456bb2086d9c3433f6688e89da9ee9d61626f2`
```text
L33: class FlyGymBody:
L34:     backend='flygym-2.1.0-neuromechfly'
L35:     test_double=False
L36:     def __init__(self,seed,world,config,*,optimized=True,initial_pose=None,physics_profile=None,body_options=None):
L37:         from .physics import profile_values
L38:         from .body_options import BodyOptions,make_world,make_flybody
L39:         self.body_options=BodyOptions.parse(body_options)
L40:         if self.body_options.model=='flybody' and self.backend==FlyGymBody.backend:self.backend='flygym-2.1.0-flybody'
L41:         self.physics_profile=profile_values(physics_profile)
L42:         if (self.body_options.attachment!='free' or self.body_options.tendons!='none') and self.physics_profile.backend!='cpu':
L43:             raise ValueError('Tethered and tendon experiments currently require CPU physics')
L44:         if self.body_options.actuation=='whole_body':
L45:             if self.physics_profile.backend!='cpu':raise ValueError('Whole-body actuation currently requires CPU MuJoCo physics')
L46:             self.backend='flygym-2.1.0-flybody-whole-body-v1'
```

### `flylab/body.py` — 원본 L425–L493
SHA256: `99a8b0be53e2c6f2eccddaa027456bb2086d9c3433f6688e89da9ee9d61626f2`
```text
L425:     def step_joint_targets(self, targets, adhesion, dt=CONTROL_DT, *, tendon_inputs=None, interpolate=True, joint_stiffness_scale=None, adhesion_control=None):
L426:         """Direct neural muscle adapter. Does not advance the predefined CPG."""
L427:         targets, adhesion = np.asarray(targets), np.asarray(adhesion)
L428:         if targets.shape != (42,) or not np.isfinite(targets).all() or np.max(np.abs(targets)) > 10:
L429:             raise ValueError('Invalid neural joint targets')
L430:         if adhesion.shape != (6,) or adhesion.dtype != np.dtype(bool):
L431:             raise ValueError('Six boolean adhesion states required')
L432:         if adhesion_control is None and getattr(self, 'graded_adhesion', False):
L433:             raise ValueError('Graded adhesion requires its recorded force control')
L434:         if adhesion_control is not None:
L435:             adhesion_control = np.asarray(adhesion_control)
L436:             if (adhesion_control.shape != (6,) or adhesion_control.dtype.kind not in 'fiu'
L437:                     or not np.isfinite(adhesion_control).all()
L438:                     or np.any((adhesion_control < 0) | (adhesion_control > 1))
L439:                     or np.any(adhesion_control[~adhesion] != 0)):
L440:                 raise ValueError('Six finite adhesion controls in [0,1], zero on released feet, required')
L441:         if type(interpolate) is not bool:
L442:             raise ValueError('Boolean activation interpolation option required')
L443:         number(dt, 'joint control dt', PHYSICS_DT, .05)
L444:         n = round(dt/PHYSICS_DT)
L445:         if abs(n*PHYSICS_DT-dt) > 1e-10: raise ValueError('Integral physical steps required')
L446:         if self.fault: raise RuntimeError(self.fault)
L447:         if hasattr(self, 'active_length_scale') or joint_stiffness_scale is not None:
L448:             from .c.banc_mechanics import set_active_length_scale
L449:             set_active_length_scale(self, joint_stiffness_scale)
L450:         if tendon_inputs is not None:
L451:             if self.tendon_control is None:raise ValueError('Tendon actuation is not enabled')
L452:             tendon_values=self.tendon_control.command_vector(tendon_inputs)
L453:             self.d.ctrl[self.tendon_control.act_ids]=tendon_values
L454:         expected_time=float(self.d.time)+dt
L455:         self.descending.fill(0.)
L456:         trajectory = None
L457:         activation_tau = getattr(self, 'joint_target_activation_tau_s', 0.0)
L458:         if activation_tau and interpolate:
L459:             # The neural muscle state still advances once per control. Its
L460:             # exact exponential trajectory is applied to native physics instead
L461:             # of jumping immediately to the interval's final activated target.
L462:             previous = np.asarray(self.last_action.joint_angles)
L463:             fractions = -np.expm1(-np.arange(1, n+1)*PHYSICS_DT/activation_tau) / -math.expm1(-dt/activation_tau)
L464:             trajectory = previous + fractions[:, None] * (targets - previous)
L465:             trajectory[-1] = targets
L466:         self.last_action = self.Action(joint_angles=targets.copy(), adhesion_onoff=adhesion.copy())
L467:         # Legacy holds the final command throughout this interval. The explicit
L468:         # continuous model supplies its trajectory at native physics steps.
L469:         self.apply(self.sim, self.fly.name, self.last_action)
L470:         if adhesion_control is not None:
L471:             self.d.ctrl[self.sim._intern_adhesionactuatorids_by_fly[self.fly.name]] = adhesion_control
L472:         for index in range(n):
L473:             if trajectory is not None:
L474:                 self.d.ctrl[self.act_ids] = trajectory[index]
L475:             self.d.xfrc_applied[:] = 0
L476:             if self.push_left > 0:
L477:                 self.d.xfrc_applied[self.thorax, :3] = self.push_force
L478:                 self.push_left = max(0., self.push_left-PHYSICS_DT)
L479:             self._step_physics()
L480:             if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
L481:                 self.fault = 'non-finite physics state'
L482:                 raise RuntimeError(self.fault)
L483:         self._forward_physics()
L484:         self._audit_extension_step(expected_time)
L485:         p, R, _ = self.pose()
L486:         self.travel += float(np.linalg.norm(p-self.last_position))
L487:         self.last_position = p
L488:         collision = self.nonfoot_contact()
L489:         if collision and not self.prev_contact: self.collisions += 1
L490:         self.prev_contact = collision
L491:         self.walk_ticks += 1
L492:         if self.outside_physical_domain(p,R):
L493:             self.fault = 'Neural muscle body fell or left physical domain; no automatic pose correction'
```


## K3. 최근 후보: 저장 입력 4×20ms와 파라미터 업데이트

### `tools/verify_c_banc_m3_reference_parameter_direction.py` — 원본 L30–L47
SHA256: `656415a13cde92d469b9ad281a16b969ce26b87e543bdc4161961ab621d7005b`
```text
L30: REGISTRATION_SHA = "4877de8d3f579edfec8fa2d3ef9c0e7a330862cdad2caaa0254ec6c4481c95d9"
L31: REFLECTION = BASE + "/parent-e357-astra-reflection-v2.json"
L32: REFLECTION_SHA = "a1cb7f8ccdc9efc829eac4d27962558d9511ec78d0ac6d6ed3996eb4902755b9"
L33: CORE = "tools/verify_c_banc_m3_coordination_parameter_direction.py"
L34: CORE_SHA = "fb4d2687baf9ad89b919b7c4692d7039992a319ffef082686933d1bbc563bb2d"
L35: ADMIN = {
L36:     BASE
L37:     + "/parent-e356-root-exit-witness.py": "4ca99cf6859391880d81df2062a71de725b1d7a915231b449289d4e284ff900f",
L38:     BASE
L39:     + "/parent-e356-root-normalize-full-return.py": "054c90615f9d561957a8b4bd5506ce65555d18f631e777362fbbdadb9c29526f",
L40: }
L41: OUTPUT = BASE + "/e356-reference-parameter-direction"
L42: AUDIT_OUTPUT = BASE + "/e356-reference-parameter-direction-audit"
L43: WINDOWS = (100, 164, 228, 292)
L44: _NATIVE = None
L45: 
L46: 
L47: def require(ok, message):
```

### `tools/verify_c_banc_m3_reference_parameter_direction.py` — 원본 L226–L253
SHA256: `656415a13cde92d469b9ad281a16b969ce26b87e543bdc4161961ab621d7005b`
```text
L226: def align_label(label):
L227:     import numpy as np
L228: 
L229:     require(
L230:         set(label) == {"command", "control_tick", "control_time_s"},
L231:         "only three label members",
L232:     )
L233:     q, tick, clock = (label[k] for k in ("command", "control_tick", "control_time_s"))
L234:     require(
L235:         q.dtype == np.float64
L236:         and q.shape == (301, 42)
L237:         and np.isfinite(q).all()
L238:         and q.flags.c_contiguous,
L239:         "literal label commands",
L240:     )
L241:     require(
L242:         tick.dtype == np.int64
L243:         and tick.shape == (301,)
L244:         and np.array_equal(tick, np.arange(301, dtype=np.int64)),
L245:         "label tick",
L246:     )
L247:     require(
L248:         clock.dtype == np.float64
L249:         and clock.shape == (301,)
L250:         and np.array_equal(clock, tick * 0.005),
L251:         "label time",
L252:     )
L253:     return np.stack([q[c + 1 : c + 5].copy() for c in WINDOWS])
```

### `tools/verify_c_banc_m3_reference_parameter_direction.py` — 원본 L327–L341
SHA256: `656415a13cde92d469b9ad281a16b969ce26b87e543bdc4161961ab621d7005b`
```text
L327: def make_direction(gradient):
L328:     import numpy as np
L329: 
L330:     require(
L331:         gradient.dtype == np.float64
L332:         and gradient.ndim == 1
L333:         and np.isfinite(gradient).all(),
L334:         "finite fresh R gradient",
L335:     )
L336:     magnitude = float(np.max(np.abs(gradient)))
L337:     require(magnitude > 0, "nonzero R gradient")
L338:     requested = 1 - 0.001 * gradient / magnitude
L339:     actual = requested.astype(np.float32).astype(np.float64)
L340:     require((actual > 0).all(), "positive actual theta")
L341:     return requested, actual
```

### `tools/verify_c_banc_m3_reference_parameter_direction.py` — 원본 L777–L860
SHA256: `656415a13cde92d469b9ad281a16b969ce26b87e543bdc4161961ab621d7005b`
```text
L777:         def tensor(value):
L778:             return torch.as_tensor(value.copy(), device=target)
L779: 
L780:         mask = tensor(packet["output_mask"]["value"])
L781:         targets = tensor(target_q).clone().detach()
L782:         norm = tensor(np.repeat(raw["normalization"], 7)).clone().detach()
L783:         record["neural_backward_requested"] = 0
L784:         record["neural_backward_returned"] = 0
L785: 
L786:         def forward(c, arm, theta):
L787:             nonlocal stage
L788:             stage = f"{arm}-{c}-forward"
L789:             key = f"window-{c}-{arm}"
L790:             state = packet[f"start-{c}"]
L791:             timely()
L792:             emit("begin")
L793:             try:
L794:                 result = rollout.run(
L795:                     *(
L796:                         tensor(state[k])
L797:                         for k in ("rate", "adaptation", "bank", "original_activation")
L798:                     ),
L799:                     tensor(packet[f"drive-{c}"]["drive"]),
L800:                     theta,
L801:                     mask,
L802:                 )
L803:                 emit("returned")
L804:             except Exception as error:
L805:                 if hasattr(error, "receipt"):
L806:                     receipts[key] = receipt_dict(error.receipt)
L807:                     emit()
L808:                 raise
L809:             receipt = result["receipt"]
L810:             receipts[key] = receipt_dict(receipt)
L811:             # Host-acquired returned raw precedes confirmation/reproduction/science.
L812:             save("q_" + arm, result["q"].detach().cpu().numpy())
L813:             save("tendon_" + arm, result["tendon"].detach().cpu().numpy())
L814:             if arm in ("neutral", "candidate"):
L815:                 for name in ("rate", "adaptation", "bank", "original_activation"):
L816:                     save(arm + "_end_" + name, result[name].detach().cpu().numpy())
L817:             emit()
L818:             try:
L819:                 receipt.confirm()
L820:             finally:
L821:                 receipts[key] = receipt_dict(receipt)
L822:                 emit("confirmed")
L823:             timely()
L824:             return result, key
L825: 
L826:         for index, c in enumerate(WINDOWS):
L827:             theta = torch.ones(
L828:                 a["groups"], dtype=torch.float64, device=target, requires_grad=True
L829:             )
L830:             result, key = forward(c, "neutral", theta)
L831:             endpoint = packet[f"endpoint-{c}"]
L832:             stage = f"neutral-{c}-reproduction"
L833:             emit("begin")
L834:             for name in ("q", "tendon", "bank", "original_activation"):
L835:                 tolerance = 1e-11 if name in ("q", "tendon") else 1e-10
L836:                 require(
L837:                     np.allclose(
L838:                         result[name].detach().cpu().numpy(),
L839:                         endpoint[name],
L840:                         rtol=0,
L841:                         atol=tolerance,
L842:                     ),
L843:                     "neutral reproduction: " + name,
L844:                 )
L845:             for name in ("rate", "adaptation"):
L846:                 require(
L847:                     exact(result[name].detach().cpu().numpy(), endpoint[name]),
L848:                     "neutral native bytes: " + name,
L849:                 )
L850:             terms = window_loss(objective, tensor(raw["boundary"][index]), result["q"])
L851:             losses = loss_vector(terms)
L852:             save("neutral_proxy_loss", losses)
L853:             # E333 input order is total,S,C,LR; our raw is S,C,total,LR.
L854:             require(
L855:                 np.allclose(
L856:                     losses[[2, 0, 1, 3]], endpoint["loss_terms"], rtol=1e-10, atol=1e-11
L857:                 ),
L858:                 "saved neutral loss order/reproduction",
L859:             )
L860:             ell = ((result["q"] - targets[index]).square() / norm).sum() / (16 * 42)
```

### `tools/verify_c_banc_m3_coordination_parameter_direction.py` — 원본 L88–L110
SHA256: `fb4d2687baf9ad89b919b7c4692d7039992a319ffef082686933d1bbc563bb2d`
```text
L88: FIXED = {
L89:     "windows": list(WINDOWS),
L90:     "controls": 4,
L91:     "nodes": 158706,
L92:     "edges": 11584852,
L93:     "selected": 725849,
L94:     "groups": 8547,
L95:     "population": 7136,
L96:     "steps_per_control": 20,
L97:     "neural_dt": 0.00025,
L98:     "control_dt": 0.005,
L99:     "step": 0.001,
L100:     "signal_floor": 1e-9,
L101:     "relative_limit": 0.05,
L102:     "science_s": 48,
L103:     "internal_s": 60,
L104:     "final_s": 5,
L105:     "supervisor_s": 65,
L106:     "host_cap": 6 * GIB,
L107:     "additional_gpu_cap": 8 * GIB,
L108:     "raw_cap": 16 * MIB,
L109:     "output_cap": OUTPUT_CAP,
L110:     "parameter_Jacobian_independently_proven": False,
```

### `tools/verify_c_banc_m3_coordination_parameter_direction.py` — 원본 L1169–L1259
SHA256: `fb4d2687baf9ad89b919b7c4692d7039992a319ffef082686933d1bbc563bb2d`
```text
L1169: def assemble(packet, *, device="cuda", dimensions=None):
L1170:     """Fresh current native assembly; no predecessor Dependencies/load calls."""
L1171:     a = axes(dimensions)
L1172:     validate_packet(packet, dimensions=dimensions)
L1173:     record = {"prohibited_old_calls": 0}
L1174:     with prohibited_call_guard(record):
L1175:         import numpy as np
L1176:         import torch
L1177:         from scipy.sparse import csr_matrix
L1178: 
L1179:         from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
L1180:         from tools.banc_m3_grouped_current_adjoint import GroupedCurrent
L1181:         from tools.banc_m3_monotone_adjoint import MonotoneDecoder
L1182:         from tools.banc_m3_smooth_temporal import SmoothTemporalObjective
L1183:         from tools.banc_m3_ssp_adjoint import SSPAdjointModel
L1184: 
L1185:         torch.set_num_threads(1)
L1186:         csr = packet["actual_csr"]
L1187:         weights = csr_matrix(
L1188:             (csr["data"], csr["indices"], csr["indptr"]), shape=(a["nodes"], a["nodes"])
L1189:         )
L1190:         mask = packet["recurrent_mask"]
L1191:         operator = GroupedCurrent(
L1192:             weights, mask["positions"], mask["edge_parameters"], a["groups"], device
L1193:         )
L1194:         config, load = packet["motor_configuration"], packet["load_parameters"]
L1195:         coefficient = np.full(a["nodes"], np.float32(1 / 200), np.float32)
L1196:         cap = np.full(a["nodes"], np.float32(200), np.float32)
L1197:         neural = SSPAdjointModel(
L1198:             operator,
L1199:             load["tau32"],
L1200:             coefficient,
L1201:             load["threshold32"],
L1202:             cap,
L1203:             config["cell_gain"],
L1204:             packet["inverse"]["value"],
L1205:             0.00025,
L1206:         )
L1207:         calibration, parameters = packet["calibration"], packet["parameters"]
L1208:         decoder = MonotoneDecoder(
L1209:             calibration,
L1210:             parameters["tau"],
L1211:             parameters["kappa"],
L1212:             parameters["slopes"],
L1213:             parameters["membership"],
L1214:             np.asarray(packet["metadata"]["neutral"], np.float64),
L1215:             device=device,
L1216:         )
L1217:         rollout = CheckpointedControlRollout(
L1218:             neural,
L1219:             decoder,
L1220:             calibration["motor_ids"],
L1221:             steps_per_control=a["steps_per_control"],
L1222:             control_dt=a["control_dt"],
L1223:             checkpoint_controls=1,
L1224:             max_controls=4,
L1225:         )
L1226:         reference = packet["reference"]
L1227:         objective = SmoothTemporalObjective(
L1228:             reference["curves"],
L1229:             np.zeros((99, 42), np.float64),
L1230:             np.zeros((99, 6), np.float64),
L1231:             device=device,
L1232:             normalization=reference["normalization"],
L1233:         )
L1234:         require(
L1235:             all(
L1236:                 exact(objective.canonical_constants[k], reference[k]) for k in REFERENCE
L1237:             ),
L1238:             "actual smooth constants bytes",
L1239:         )
L1240:         for name, expected in (
L1241:             ("tau", load["tau32"]),
L1242:             ("threshold", load["threshold32"]),
L1243:             ("coefficient", coefficient),
L1244:             ("cap", cap),
L1245:             ("gain", config["cell_gain"]),
L1246:             ("inverse_adaptation_tau", packet["inverse"]["value"]),
L1247:         ):
L1248:             require(
L1249:                 exact(getattr(neural, name).detach().cpu().numpy(), expected),
L1250:                 "actual native physiology tensor",
L1251:             )
L1252:         record["constant_metadata"] = objective.constant_metadata
L1253:         record["actual_import_sha256"] = actual_imports()
L1254:     return {
L1255:         "rollout": rollout,
L1256:         "objective": objective,
L1257:         "operator": operator,
L1258:         "record": record,
L1259:     }
```

### `tools/verify_c_banc_m3_coordination_parameter_direction.py` — 원본 L1331–L1357
SHA256: `fb4d2687baf9ad89b919b7c4692d7039992a319ffef082686933d1bbc563bb2d`
```text
L1331: def patch_weights(csr, mask, theta):
L1332:     import numpy as np
L1333: 
L1334:     require(
L1335:         theta.dtype == np.float64
L1336:         and exact(theta, theta.astype(np.float32).astype(np.float64))
L1337:         and np.isfinite(theta).all()
L1338:         and np.all(theta > 0),
L1339:         "f32-promoted efficacy",
L1340:     )
L1341:     data = csr["data"].copy()
L1342:     positions, groups = mask["positions"], mask["edge_parameters"]
L1343:     with np.errstate(over="raise", invalid="raise"):
L1344:         selected = np.float32(data[positions] * theta.astype(np.float32)[groups])
L1345:     data[positions] = selected
L1346:     count = len(np.unique(data.view(np.uint32)))
L1347:     require(
L1348:         np.isfinite(data).all()
L1349:         and np.array_equal(np.signbit(selected), np.signbit(csr["data"][positions])),
L1350:         "signed selected weights",
L1351:     )
L1352:     return selected, {
L1353:         "full_native_palette_count": count,
L1354:         "native_deployable": count <= 16384,
L1355:         "selected_c_order_sha256": array_sha(selected),
L1356:         "full_data_c_order_sha256": array_sha(data),
L1357:     }
```

### `tools/verify_c_banc_m3_common_descent_direction.py` — 원본 L246–L305
SHA256: `fa81557c07025de874cd08420ae7ced4a28d81b692508c6960939902fb0570ea`
```text
L246: def common_direction(c_vector, s_proxy_vector):
L247:     import numpy as np
L248: 
L249:     valid_array(c_vector)
L250:     valid_array(s_proxy_vector, c_vector.shape)
L251:     uc, nc = scaled_unit(c_vector)
L252:     us, ns = scaled_unit(s_proxy_vector)
L253:     info = {
L254:         "branch": "zero_component_gradient",
L255:         "norm_C": nc,
L256:         "norm_S_proxy": ns,
L257:         "positive_dot_scope": "submitted C and reconstructed rounded-adjoint S proxy only",
L258:         **LIMITATIONS,
L259:     }
L260:     raw = {
L261:         "component_max_abs": np.array([nc["max_abs"], ns["max_abs"]], np.float64),
L262:         "component_scaled_L2": np.array([nc["scaled_L2"], ns["scaled_L2"]], np.float64),
L263:     }
L264:     if uc is None or us is None:
L265:         return raw, info
L266:     raw.update(
L267:         unit_C=uc,
L268:         unit_S_proxy=us,
L269:         cosine=np.array(np.dot(uc, us), np.float64),
L270:         bisector=uc + us,
L271:     )
L272:     info["cosine"] = float(raw["cosine"])
L273:     magnitude = float(np.max(np.abs(raw["bisector"])))
L274:     if magnitude == 0:
L275:         info["branch"] = "common_descent_unavailable"
L276:         return raw, info
L277:     direction = raw["bisector"] / magnitude
L278:     with np.errstate(over="raise", invalid="raise"):
L279:         plus = np.float32(1 + 0.001 * direction).astype(np.float64)
L280:         minus = np.float32(1 - 0.001 * direction).astype(np.float64)
L281:         delta = plus - minus
L282:         tc, ts = float(np.dot(c_vector, delta)), float(np.dot(s_proxy_vector, delta))
L283:     require(
L284:         math.isfinite(tc)
L285:         and math.isfinite(ts)
L286:         and np.all(plus > 0)
L287:         and np.all(minus > 0),
L288:         "finite actual rounded geometry",
L289:     )
L290:     raw.update(
L291:         direction=direction,
L292:         theta_plus=plus,
L293:         theta_minus=minus,
L294:         actual_delta_theta=delta,
L295:         geometry_T_C=np.array(tc, np.float64),
L296:         geometry_T_S_proxy=np.array(ts, np.float64),
L297:     )
L298:     info.update(
L299:         branch="normal"
L300:         if tc > 1e-9 and ts > 1e-9
L301:         else "actual_rounded_common_descent_uninformative",
L302:         actual_T_C=tc,
L303:         actual_T_S_proxy=ts,
L304:     )
L305:     return raw, info
```


## K4. 실제 신체 전달·startup·1.5초 선별·출력 요약

### `tools/verify_c_banc_m3_common_descent_body.py` — 원본 L230–L255
SHA256: `7d123ea4d005f4c45c4cc386f94edd09eabab9a5e54b873dd855613ea272bc4e`
```text
L230: def resolve_original_api():
L231:     """Resolve actual public APIs and bind inert arguments without calling them."""
L232:     from flylab.c.graph import GraphStore
L233:     from flylab.c.storage import StateStore
L234:     from tools import banc_m3_ssp_efficacy_trial as efficacy
L235:     from tools import banc_m3_ssp_recovery_trial as recovery
L236:     from tools import verify_c_banc_m3_e113_body as native
L237: 
L238:     inert = object()
L239:     for function, args, keywords in (
L240:         (GraphStore.load, (inert,), {}),
L241:         (StateStore.load, (inert,), {"max_files": 123}),
L242:         (recovery.checked_checkpoint, (inert,), {}),
L243:         (recovery.SSPRecoveryResourceTrial.from_checkpoint, (inert, inert), {}),
L244:         (efficacy.SSPEfficacyTrial, (inert, inert, inert, inert), {}),
L245:         (native.install_native_observer, (inert, inert), {}),
L246:     ):
L247:         require(callable(function), "actual original API callable")
L248:         inspect.signature(function).bind(*args, **keywords)
L249:     return {
L250:         "GraphStore": GraphStore,
L251:         "StateStore": StateStore,
L252:         "recovery": recovery,
L253:         "efficacy": efficacy,
L254:         "native": native,
L255:     }
```

### `tools/verify_c_banc_m3_common_descent_body.py` — 원본 L343–L369
SHA256: `7d123ea4d005f4c45c4cc386f94edd09eabab9a5e54b873dd855613ea272bc4e`
```text
L343: def startup99(command, tendon, target_command, target_tendon):
L344:     import numpy as np
L345: 
L346:     typed(command, (100, 42), "float64", "startup100 boundary rows")
L347:     typed(tendon, (100, 6), "float64", "startup100 tendon")
L348:     typed(target_command, (99, 42), "float64", "target99")
L349:     typed(target_tendon, (99, 6), "float64", "target99 tendon")
L350:     with np.errstate(over="raise", invalid="raise"):
L351:         dq, dt = command[1:100] - target_command, tendon[1:100] - target_tendon
L352:         result = {
L353:             "q_rms": float(np.linalg.norm(dq) / math.sqrt(dq.size)),
L354:             "q_max": float(np.abs(dq).max()),
L355:             "tendon_rms": float(np.linalg.norm(dt) / math.sqrt(dt.size)),
L356:             "tendon_max": float(np.abs(dt).max()),
L357:         }
L358:     require(all(math.isfinite(v) for v in result.values()), "finite startup99 errors")
L359:     result["passed"] = all(
L360:         result[k] <= v
L361:         for k, v in (
L362:             ("q_rms", 0.018),
L363:             ("q_max", 0.04),
L364:             ("tendon_rms", 0.009),
L365:             ("tendon_max", 0.025),
L366:         )
L367:     )
L368:     result["controls"] = 99
L369:     return result
```

### `tools/verify_c_banc_m3_common_descent_body.py` — 원본 L393–L413
SHA256: `7d123ea4d005f4c45c4cc386f94edd09eabab9a5e54b873dd855613ea272bc4e`
```text
L393: def supported_slip(raw):
L394:     import numpy as np
L395: 
L396:     validate_native(raw)
L397:     support = ((raw["time_s"] >= 0.5) & (raw["time_s"] < 1.5))[:, None] & (
L398:         raw["normal_load_bw"] >= 0.02
L399:     )
L400:     if not support.any():
L401:         return {"available": False, "value": None, "reason": "no_supported_samples"}
L402:     if (
L403:         not np.all(raw["slip_valid"][support])
L404:         or not np.isfinite(raw["slip_mm_s"][support]).all()
L405:     ):
L406:         return {"available": False, "value": None, "reason": "invalid_supported_slip"}
L407:     with np.errstate(over="raise", invalid="raise", divide="raise"):
L408:         value = float(
L409:             np.sum(raw["normal_load_bw"][support] * raw["slip_mm_s"][support])
L410:             / np.sum(raw["normal_load_bw"][support])
L411:         )
L412:     require(math.isfinite(value), "finite supported slip")
L413:     return {"available": True, "value": value, "reason": None}
```

### `tools/verify_c_banc_m3_common_descent_body.py` — 원본 L913–L983
SHA256: `7d123ea4d005f4c45c4cc386f94edd09eabab9a5e54b873dd855613ea272bc4e`
```text
L913: def evaluate(output, control, native, data, startup):
L914:     import numpy as np
L915:     import torch
L916: 
L917:     from tools import analyze_c_banc_biological_gait as gait
L918:     from tools.banc_m3_phase_free_shape import PhaseFreeShapeObjective
L919:     from tools.banc_m3_smooth_temporal import SmoothTemporalObjective
L920: 
L921:     require(
L922:         len(control["command"]) == 301 and validate_native(native) == 15000,
L923:         "full evaluation only",
L924:     )
L925:     q, tendon = (
L926:         torch.as_tensor(control["command"][1:]),
L927:         torch.as_tensor(control["tendon"][1:]),
L928:     )
L929:     reference = data["reference"]
L930:     objective = SmoothTemporalObjective(
L931:         reference["curves"],
L932:         data["startup"]["target_command"],
L933:         data["startup"]["target_tendon"],
L934:         device="cpu",
L935:         normalization=reference["normalization"],
L936:     )
L937:     for key, expected in reference.items():
L938:         require(
L939:             e.exact(objective.canonical_constants[key], expected),
L940:             "actual approved smooth constants",
L941:         )
L942:     legacy = PhaseFreeShapeObjective(
L943:         reference["curves"],
L944:         data["startup"]["target_command"],
L945:         data["startup"]["target_tendon"],
L946:         device="cpu",
L947:     )
L948:     with torch.no_grad():
L949:         smooth, shape = objective.evaluate(q, tendon), legacy.evaluate(q, tendon)
L950:     rows = e.read_json(Path(output) / "trace.json")["samples"]
L951:     positions = np.asarray([row["body"]["position"] for row in rows])
L952:     forward = np.asarray(rows[0]["body"]["basis"])[[0, 2], 0].copy()
L953:     require(np.linalg.norm(forward) > 0, "physical initial forward axis")
L954:     forward /= np.linalg.norm(forward)
L955:     displacement = positions[299, [0, 2]] - positions[100, [0, 2]]
L956:     slip = supported_slip(native)
L957:     analyzed, _, _ = gait.analyze(gait.load_model(Path(output) / "trace.json"))
L958:     health = {
L959:         "healthy": bool(
L960:             np.all(native["upright"] >= 0.9)
L961:             and not native["nonfoot_contact"].any()
L962:             and all(not row["fault"] for row in rows)
L963:         ),
L964:         "minimum_upright": float(native["upright"].min()),
L965:         "nonfoot_contact_count": int(native["nonfoot_contact"].sum()),
L966:     }
L967:     return {
L968:         "smooth": {"training_loss": float(smooth["training_loss"])},
L969:         "shape": {"shape_loss": float(shape["shape_loss"])},
L970:         "summary": {
L971:             "forward_mm": float(displacement @ forward),
L972:             "slip_mm_s": slip["value"],
L973:             "slip": slip,
L974:             "longest_continuous_forward_bout_s": float(
L975:                 analyzed["window"]["longest_forward_bout_s"]
L976:             ),
L977:             "complete_cycles": {
L978:                 leg: int(analyzed["events"][leg]["complete_cycles"]) for leg in LEGS
L979:             },
L980:             "health": health,
L981:         },
L982:         "startup": startup,
L983:     }
```

### `tools/verify_c_banc_m3_common_descent_body.py` — 원본 L1442–L1482
SHA256: `7d123ea4d005f4c45c4cc386f94edd09eabab9a5e54b873dd855613ea272bc4e`
```text
L1442:     def construct(self, data, ledger, timely):
L1443:         old = candidate = None
L1444:         try:
L1445:             with SetupObserver(ledger, timely):
L1446:                 old = self.recovery.SSPRecoveryResourceTrial.from_checkpoint(
L1447:                     data["graph"], copy.deepcopy(data["saved"])
L1448:                 )
L1449:                 data["source_csr"] = host_csr(old.network)
L1450:                 require(
L1451:                     old.network.identity
L1452:                     == data["inventory"]["source_network_identity"],
L1453:                     "actual original source identity",
L1454:                 )
L1455:                 candidate = self.efficacy.SSPEfficacyTrial(
L1456:                     old, data["positions"], data["groups"], data["theta"]
L1457:                 )
L1458:             self.synchronize()
L1459:             ledger["device_confirmed"] = True
L1460:             require(
L1461:                 ledger["body_constructors_requested"]
L1462:                 == ledger["body_constructors_returned"]
L1463:                 == 1
L1464:                 and ledger["settling_physics_requested"]
L1465:                 == ledger["settling_physics_host_returned"]
L1466:                 == 2000
L1467:                 and ledger["neural_setup_requested"]
L1468:                 == ledger["neural_setup_host_returned"]
L1469:                 == 320
L1470:                 and ledger["neural_capture_requested"]
L1471:                 == ledger["neural_capture_recorded"]
L1472:                 == 160
L1473:                 and not ledger["unknown_work"],
L1474:                 "actual original constructor work mismatch",
L1475:             )
L1476:             return candidate
L1477:         except Exception as error:
L1478:             if candidate is not None:
L1479:                 self.recovery.cleanup_owned(candidate, error)
L1480:             elif old is not None and not old.closed:
L1481:                 self.recovery.cleanup_owned(old, error)
L1482:             raise
```

### `tools/verify_c_banc_m3_candidate_body_handoff.py` — 원본 L1106–L1177
SHA256: `5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a`
```text
L1106: def body_data(raw, inventory, timer, *, b):
L1107:     """Finite E345 graph/whole/mask/reference/startup; never old audited_candidate."""
L1108:     from tools import verify_c_banc_m3_reference_parameter_direction as ref
L1109:     core, ledger = ref.native(), []
L1110:     selected = inventory["selected_archives"]
L1111:     require(set(selected) == {"recurrent_mask", "reference", "startup", "audited_candidate"}, "finite body roles")
L1112:     rejected = selected["audited_candidate"]["path"]
L1113:     role_paths = {selected[k]["path"] for k in ("recurrent_mask", "reference", "startup")}
L1114:     allowed = {name: spec for name, spec in inventory["files"].items() if name != rejected}
L1115:     require(len(allowed) == 134 and rejected not in role_paths, "whole124/graph7/selected3 finite roster")
L1116:     for name, spec in allowed.items():
L1117:         timer.check("science")
L1118:         path = safe(name)
L1119:         require(path.stat().st_size == spec["bytes"] and core.digest(path, lambda: timer.check("science")) == spec["sha256"],
L1120:                 "supervised original body input bytes/hash")
L1121:         if path.suffix == ".npy":
L1122:             with path.open("rb") as stream:
L1123:                 core.strict_npy_header(stream)
L1124:         ledger.append({"path": name, "expected_sha256": spec["sha256"], "bytes": spec["bytes"], "role": spec.get("role")})
L1125:     dependencies = strict_json(read_spec(registration()["body_native_dependencies"]))
L1126:     require(dependencies["network_asset_download_permitted"] is False
L1127:             and dependencies["mesh_cache_complete_currently"] is True, "registered finite offline constructor assets")
L1128:     for name, spec in dependencies["native_constructor_asset_bytes"].items():
L1129:         timer.check("science")
L1130:         path = Path(name)
L1131:         require(path.is_absolute() and path.is_file() and not path.is_symlink()
L1132:                 and path.stat().st_size == spec["bytes"]
L1133:                 and core.digest(path, lambda: timer.check("science")) == spec["sha256"], "supervised finite native constructor asset bytes")
L1134:     b.whole_metadata(ROOT, inventory)
L1135:     api = b.resolve_original_api()
L1136:     graph = api["GraphStore"].load(safe(inventory["graph_bundle"]))
L1137:     saved = api["StateStore"].load(safe(inventory["initial_whole"]), max_files=123)
L1138:     require(graph.hash == inventory["graph_hash"], "same full source graph identity")
L1139:     api["recovery"].checked_checkpoint(saved)
L1140:     arrays = {role: core.load_selected(ROOT, selected[role], ledger, timely=lambda: timer.check("science"), generated=False)
L1141:               for role in ("recurrent_mask", "reference", "startup")}
L1142:     return {"graph": graph, "saved": saved, "inventory": inventory, "ledger": ledger,
L1143:             "theta": raw["theta_actual"].copy(), "selected_weights": raw["selected_weights_candidate"].copy(),
L1144:             "positions": arrays["recurrent_mask"]["positions"], "groups": arrays["recurrent_mask"]["edge_parameters"],
L1145:             "reference": arrays["reference"], "startup": arrays["startup"]}
L1146: 
L1147: 
L1148: def initial_body(session, data, candidate_identity, record, timer, *, b, factory):
L1149:     import numpy as np
L1150:     observed = session.snapshot()
L1151:     factory.efficacy.checked_checkpoint(observed)
L1152:     inverse = factory.efficacy._source_representation(observed, observed[factory.efficacy.PATCH_KEY])
L1153:     factory.recovery.require_saved_whole_equal(inverse, data["saved"], "E358 inverse whole exact including wall")
L1154:     require(session.control_tick == session.network.tick == 0 and abs(session.body.physics_time()) <= 2e-8
L1155:             and session.graph.hash == data["inventory"]["graph_hash"]
L1156:             and session.adapter.identity == data["inventory"]["source_adapter_identity"]
L1157:             and session.network.identity != data["inventory"]["source_network_identity"]
L1158:             and session.network.interval_steps == 20 and session.neural_steps_per_control == 20,
L1159:             "source/decoder/current candidate/control0 identities")
L1160:     proof = b.verify_transfer(session.network, data["source_csr"], data["positions"], data["groups"],
L1161:                               data["theta"], data["selected_weights"],
L1162:                               expected_palette=record["palette"]["full_native_palette_count"],
L1163:                               timely=lambda: timer.check("science"))
L1164:     require(proof["selected_sha256"] == candidate_identity["selected_sha256"]
L1165:             and proof["candidate_CSR_data_sha256"] == record["palette"]["full_data_c_order_sha256"]
L1166:             and proof["candidate_CSR_data_sha256"] != proof["source_CSR_data_sha256"], "current CSR distinct/source selected/full digest")
L1167:     _, boundary = b.capture(session)
L1168:     target_tendon = data["saved"]["adapter"]["original"]["original"]["original"]["tendon_activation"]
L1169:     require(b.e.exact(boundary["command"], np.asarray(data["saved"]["body"]["action"]["joint_angles"], np.float64))
L1170:             and b.e.exact(boundary["tendon"], target_tendon), "source control0 command/tendon exact")
L1171:     require(session._cpg_digest() == data["saved"]["cpg_hash"] == session.cpg_hash, "source CPG unchanged at transfer")
L1172:     proof.update(candidate=candidate_identity, source_inverse_whole_exact=True,
L1173:                  source_boundary_command_and_tendon_exact=True,
L1174:                  source_network_identity=data["inventory"]["source_network_identity"],
L1175:                  source_adapter_identity=session.adapter.identity, graph_hash=session.graph.hash,
L1176:                  theta_sha256=b.array_sha(data["theta"]), cpg_hash=session.cpg_hash)
L1177:     return proof
```

### `tools/verify_c_banc_m3_candidate_body_handoff.py` — 원본 L1213–L1285
SHA256: `5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a`
```text
L1213: def screen(metrics, startup):
L1214:     summary = metrics["summary"]
L1215:     gates = {"startup_four": startup.get("passed") is True,
L1216:              "forward": summary["forward_mm"] >= 1.6790085779712631,
L1217:              "slip": summary["slip_mm_s"] is not None and summary["slip_mm_s"] <= 1.8026436968249955,
L1218:              "bout": summary["longest_continuous_forward_bout_s"] >= .3,
L1219:              "all_six_cycles": all(type(summary["complete_cycles"].get(k)) is int
L1220:                                    and summary["complete_cycles"][k] > 0 for k in ("lf", "lm", "lh", "rf", "rm", "rh")),
L1221:              "native_health": summary["health"]["healthy"] is True}
L1222:     return gates, all(gates.values())
L1223: 
L1224: 
L1225: def body(c, result, frozen, timer):
L1226:     watch = session = observer = factory = None
L1227:     controls, rows, samples, construction, counters, data = [], [], [], {}, {}, None
L1228:     reason, stage = "duration", "body-preload"
L1229:     try:
L1230:         watch = gpu_watch(c)
L1231:         raw, record, identity = admitted_candidate(c, frozen, timer)
L1232:         from tools import verify_c_banc_m3_common_descent_body as b
L1233:         inventory = strict_json(read_spec(registration()["body_inventory"]))
L1234:         data = body_data(raw, inventory, timer, b=b)
L1235:         result["candidate"] = identity
L1236:         counters = b.new_counters()
L1237:         factory = b.OriginalFactory()
L1238:         stage = "construct"
L1239:         session = factory.construct(data, construction, lambda: timer.check("science"))
L1240:         stage = "initial-transfer"
L1241:         proof = initial_body(session, data, identity, record, timer, b=b, factory=factory)
L1242:         put(c, "transfer-chunks.json", proof, timer)
L1243:         result["transfer"] = proof
L1244:         metadata = b.trace_metadata(session)
L1245:         row, boundary = b.capture(session)
L1246:         rows.append(row)
L1247:         controls.append(boundary)
L1248:         counters["completed_rows"] = 1
L1249:         observer = b.LiveObserver(session, counters, samples, timely=lambda: timer.check("science"),
L1250:                                   install_contact=factory.install_contact, synchronize=factory.synchronize).install()
L1251:         for control in range(1, 301):
L1252:             stage = "control-" + str(control)
L1253:             timer.check("science")
L1254:             counters["controls_requested"] += 1
L1255:             # Any unreturned advance is unknown, even a native health exception.
L1256:             try:
L1257:                 session.advance(1, capture=False)
L1258:             except Exception:
L1259:                 counters["advance_unreturned_unknown"] = True
L1260:                 raise
L1261:             counters["controls_host_returned"] += 1
L1262:             factory.synchronize()
L1263:             counters["physics_device_confirmed"] = counters["physics_host_returned"]
L1264:             row, boundary = b.capture(session)
L1265:             rows.append(row)
L1266:             controls.append(boundary)
L1267:             require(session.control_tick == control and session.network.tick == 20 * control
L1268:                     and len(samples) == 50 * control and abs(session.body.physics_time() - .005 * control) <= 2e-8
L1269:                     and session._cpg_digest() == proof["cpg_hash"], "actual sensory feedback clocks and CPG")
L1270:             counters["controls_completed"] += 1
L1271:             counters["completed_rows"] = len(controls)
L1272:             counters["observer_samples"] = len(samples)
L1273:             if control == 99:
L1274:                 import numpy as np
L1275:                 result["startup"] = b.startup99(np.stack([v["command"] for v in controls]),
L1276:                                                 np.stack([v["tendon"] for v in controls]), **data["startup"])
L1277:                 if not result["startup"]["passed"]:
L1278:                     reason = "startup_rejected"
L1279:                     break
L1280:             monitor_check(watch.snapshot(), c["limits"])
L1281:         verify_work(counters, construction, counters["controls_completed"], full=reason == "duration")
L1282:         require(counters.get("advance_unreturned_unknown", False) is False, "known advance work")
L1283:         result.update(run_valid=True, reason=reason, classification="scientific_rejected")
L1284:     except Exception as error:
L1285:         fail(result, error, stage)
```


## K5. 실제 주기·위상·운동폭 평가 정의

### `tools/analyze_c_banc_biological_gait.py` — 원본 L49–L123
SHA256: `bcceb5d1e9f26293ffc800d09a1fbf063a1ab784d73cf5489b74c57ecdd14760`
```text
L49: def geometric_knee(points):
L50:     """Interior CTr--FTi--TiTa angle; radians or neutral-pose conventions unused."""
L51:     a = points[:, :, 1] - points[:, :, 2]
L52:     b = points[:, :, 3] - points[:, :, 2]
L53:     den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
L54:     if np.any(den <= 1e-12):
L55:         raise ValueError("Degenerate femur or tibia in coordinate recording")
L56:     return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1) / den, -1, 1)))
L57: 
L58: 
L59: def position_forward_speed(position, forward, dt):
L60:     velocity = np.gradient(position, dt, axis=0)
L61:     return gaussian_filter1d(
L62:         np.sum(velocity * forward, axis=1), 0.02 / dt, mode="nearest"
L63:     )
L64: 
L65: 
L66: def load_model(path):
L67:     payload = json.loads(Path(path).read_text(encoding="utf-8"))
L68:     rows = payload["samples"]
L69:     t = np.array([r["time_s"] for r in rows])
L70:     dt = validate_clock(t)
L71:     if not np.isclose(dt, 0.005) or any(
L72:         r["control_tick"] != rows[0]["control_tick"] + i for i, r in enumerate(rows)
L73:     ):
L74:         raise ValueError("Expected complete 5 ms control-boundary recording")
L75:     basis = np.array([r["body"]["basis"] for r in rows])
L76:     if not np.allclose(np.einsum("tji,tjk->tik", basis, basis), np.eye(3), atol=1e-9):
L77:         raise ValueError("Body basis is not orthogonal")
L78:     origin = np.array([r["body"]["position"] for r in rows])
L79:     points = np.array([[r["body"]["legs"][leg] for leg in LEGS] for r in rows])
L80:     # Row vectors projected onto the recorded forward/lateral/up basis columns.
L81:     ego = np.einsum("tlpj,tjk->tlpk", points - origin[:, None, None], basis)
L82:     lengths = np.median(
L83:         np.linalg.norm(np.diff(ego, axis=2), axis=-1).sum(axis=-1), axis=0
L84:     )
L85:     brake_times = [
L86:         r["time_s"] for r in rows if (r.get("navigation") or {}).get("brake_request")
L87:     ]
L88:     end = min(t[-1], 10.0, min(brake_times) if brake_times else np.inf)
L89:     window = (t >= 0.5) & (t < end)
L90:     speed = position_forward_speed(origin, basis[:, :, 0], dt)
L91:     yaw = np.unwrap([r["body"]["yaw"] for r in rows])
L92:     yaw_rate = gaussian_filter1d(np.gradient(yaw, dt), 0.02 / dt, mode="nearest")
L93:     return {
L94:         "t": t,
L95:         "dt": dt,
L96:         "ap": ego[:, :, -1, 0] / lengths,
L97:         "knee": geometric_knee(ego),
L98:         "window": window,
L99:         "moving": window & (speed > 0.5),
L100:         "straight": np.abs(yaw_rate) < 0.5,
L101:         "speed": speed,
L102:         "yaw_rate": yaw_rate,
L103:         "contact": np.array([r["feet"]["load_bearing"] for r in rows], dtype=bool),
L104:         "position": origin,
L105:         "upright": np.array([r["upright"] for r in rows]),
L106:         "faults": [r["fault"] for r in rows if r["fault"]],
L107:         "lengths": lengths,
L108:         "start_s": 0.5,
L109:         "end_s": float(end),
L110:         "source": {
L111:             "path": str(Path(path).resolve()),
L112:             "sha256": sha256(path),
L113:             "kind": "model",
L114:             "body_hash": payload["body_hash"],
L115:             "model_hash": payload["model_hash"],
L116:             "adapter_hash": payload["adapter_hash"],
L117:             "graph_hash": payload["graph_hash"],
L118:             "parameters": payload["parameters"],
L119:             "foot": "tarsus5 origin proxy",
L120:             "length_unit": "mm",
L121:             "speed_source": "Thorax coordinate derivative, projected onto physical forward axis, Gaussian sigma 20 ms; not boundary body.speed",
L122:         },
L123:     }
```

### `tools/analyze_c_banc_biological_gait.py` — 원본 L180–L240
SHA256: `bcceb5d1e9f26293ffc800d09a1fbf063a1ab784d73cf5489b74c57ecdd14760`
```text
L180: def phase_and_events(data, amplitude_floor=0.01, prominence=0.02):
L181:     """Do not concatenate motion bouts or compute phase for motionless legs."""
L182:     t, dt, ap = data["t"], data["dt"], data["ap"]
L183:     phi = np.full(ap.shape, np.nan)
L184:     events = [[] for _ in LEGS]
L185:     eligible = np.zeros(len(t), dtype=bool)
L186:     bouts = []
L187:     for lo, hi in runs(data["moving"]):
L188:         if (hi - lo - 1) * dt < 0.3 - 1e-9:
L189:             continue
L190:         eligible[lo:hi] = True
L191:         bouts.append(
L192:             {
L193:                 "start_s": float(t[lo]),
L194:                 "end_s": float(t[hi - 1]),
L195:                 "samples": int(hi - lo),
L196:             }
L197:         )
L198:         x = ap[lo:hi]
L199:         z = hilbert(x - x.mean(axis=0), axis=0)
L200:         valid = np.abs(z) >= amplitude_floor
L201:         trim = max(1, int(np.ceil(0.02 / dt)))
L202:         valid[:trim] = False
L203:         valid[-trim:] = False
L204:         phi[lo:hi] = np.where(valid, np.angle(z), np.nan)
L205:         # Largest odd window <= 25 ms; reference needs >=3 points for a filter.
L206:         win = int(np.floor(0.025 / dt + 1e-9))
L207:         win -= 1 - win % 2
L208:         smooth = (
L209:             savgol_filter(x, win, min(2, win - 1), axis=0) if win >= 3 else x.copy()
L210:         )
L211:         for leg in range(6):
L212:             peaks, _ = find_peaks(
L213:                 smooth[:, leg],
L214:                 distance=max(1, int(np.ceil(0.04 / dt))),
L215:                 prominence=prominence,
L216:             )
L217:             for a, b in itertools.pairwise(peaks):
L218:                 p = int(a + np.argmin(smooth[a : b + 1, leg]))
L219:                 if p in (a, b):
L220:                     continue
L221:                 cycle = {
L222:                     "aep_s": float(t[lo + a]),
L223:                     "pep_s": float(t[lo + p]),
L224:                     "next_aep_s": float(t[lo + b]),
L225:                     "period_s": float((b - a) * dt),
L226:                     "retraction_s": float((p - a) * dt),
L227:                     "protraction_s": float((b - p) * dt),
L228:                     "retraction_fraction": float((p - a) / (b - a)),
L229:                     "excursion_leg_lengths": float(smooth[a, leg] - smooth[p, leg]),
L230:                     "mean_forward_mm_s": float(
L231:                         np.mean(data["speed"][lo + a : lo + b + 1])
L232:                     )
L233:                     if data["speed"] is not None
L234:                     else None,
L235:                     "straight_fraction": float(
L236:                         data["straight"][lo + a : lo + b + 1].mean()
L237:                     ),
L238:                 }
L239:                 events[leg].append(cycle)
L240:     return phi, events, eligible, bouts
```

### `tools/analyze_c_banc_biological_gait.py` — 원본 L257–L308
SHA256: `bcceb5d1e9f26293ffc800d09a1fbf063a1ab784d73cf5489b74c57ecdd14760`
```text
L257: def circular_summary(cycles):
L258:     cycles = np.asarray(cycles, dtype=float)
L259:     cycles = cycles[np.isfinite(cycles)]
L260:     if not len(cycles):
L261:         return {
L262:             "samples": 0,
L263:             "mean_cycles": None,
L264:             "resultant_length": None,
L265:             "mean_antiphase_distance_cycles": None,
L266:         }
L267:     z = np.mean(np.exp(2j * np.pi * cycles))
L268:     return {
L269:         "samples": len(cycles),
L270:         "mean_cycles": float((np.angle(z) / (2 * np.pi)) % 1),
L271:         "resultant_length": float(abs(z)),
L272:         "mean_antiphase_distance_cycles": float(
L273:             np.mean(np.abs(((cycles - 0.5 + 0.5) % 1) - 0.5))
L274:         ),
L275:     }
L276: 
L277: 
L278: def phase_summary(phi, mask, dt):
L279:     pairs = {}
L280:     for a, b in PAIRS:
L281:         delta = (phi[:, b] - phi[:, a]) / (2 * np.pi)
L282:         values = delta[mask & np.isfinite(delta)]
L283:         item = circular_summary(values)
L284:         item["valid_duration_s"] = item["samples"] * dt
L285:         pairs[f"{LEGS[a]}-{LEGS[b]}"] = item
L286:     return pairs
L287: 
L288: 
L289: def event_summary(events):
L290:     result = {}
L291:     for leg, cycles in zip(LEGS, events):
L292:         periods = [c["period_s"] for c in cycles]
L293:         result[leg] = {
L294:             "complete_cycles": len(cycles),
L295:             "recurrence_evaluable": len(cycles) >= 3,
L296:             "period_s": percentiles(periods),
L297:             "retraction_s": percentiles([c["retraction_s"] for c in cycles]),
L298:             "protraction_s": percentiles([c["protraction_s"] for c in cycles]),
L299:             "retraction_fraction": percentiles(
L300:                 [c["retraction_fraction"] for c in cycles]
L301:             ),
L302:             "excursion_leg_lengths": percentiles(
L303:                 [c["excursion_leg_lengths"] for c in cycles]
L304:             ),
L305:             "period_cv": float(np.std(periods) / np.mean(periods)) if periods else None,
L306:             "cycles": cycles,
L307:         }
L308:     return result
```

### `tools/analyze_c_banc_biological_gait.py` — 원본 L311–L393
SHA256: `bcceb5d1e9f26293ffc800d09a1fbf063a1ab784d73cf5489b74c57ecdd14760`
```text
L311: def analyze(data):
L312:     phi, events, eligible, bouts = phase_and_events(data)
L313:     moving, window, dt = data["moving"], data["window"], data["dt"]
L314:     result = {
L315:         "source": data["source"],
L316:         "window": {
L317:             "start_s": data["start_s"],
L318:             "end_s": data["end_s"],
L319:             "sampled_duration_s": float(np.trapezoid(window.astype(float), data["t"])),
L320:             "moving_duration_s": float(np.trapezoid(moving.astype(float), data["t"])),
L321:             "eligible_bout_duration_s": float(
L322:                 np.trapezoid(eligible.astype(float), data["t"])
L323:             ),
L324:             "bouts": bouts,
L325:             "straight_eligible_duration_s": float(
L326:                 np.trapezoid((eligible & data["straight"]).astype(float), data["t"])
L327:             ),
L328:             "longest_forward_bout_s": max(
L329:                 ((b - a - 1) * dt for a, b in runs(moving)), default=0.0
L330:             ),
L331:         },
L332:         "pair_phase": phase_summary(phi, eligible, dt),
L333:         "straight_pair_phase": phase_summary(phi, eligible & data["straight"], dt),
L334:         "events": event_summary(events),
L335:         "knee_interior_deg": {
L336:             leg: percentiles(data["knee"][window, i]) for i, leg in enumerate(LEGS)
L337:         },
L338:         "foot_ap_leg_lengths": {
L339:             leg: percentiles(data["ap"][window, i]) for i, leg in enumerate(LEGS)
L340:         },
L341:         "median_leg_lengths": data["lengths"].tolist(),
L342:         "statistical_unit": "A recorded trajectory, not independent time points or individual flies",
L343:     }
L344:     result["sensitivity"] = {}
L345:     for floor in (0.005, 0.02):
L346:         alt, _, _, _ = phase_and_events(data, amplitude_floor=floor)
L347:         result["sensitivity"][f"phase_amplitude_{floor}"] = phase_summary(
L348:             alt, eligible, dt
L349:         )
L350:     for prom in (0.01, 0.04):
L351:         _, alt_events, _, _ = phase_and_events(data, prominence=prom)
L352:         result["sensitivity"][f"event_prominence_{prom}"] = {
L353:             leg: {
L354:                 "complete_cycles": len(ev),
L355:                 "period_s": percentiles([c["period_s"] for c in ev]),
L356:             }
L357:             for leg, ev in zip(LEGS, alt_events)
L358:         }
L359:     if data["speed"] is not None:
L360:         result["forward_speed_mm_s"] = percentiles(data["speed"][window])
L361:         result["yaw_rate_rad_s"] = percentiles(data["yaw_rate"][window])
L362:         pos = data["position"][window][:, [0, 2]]
L363:         result["physical"] = {
L364:             "minimum_upright": float(data["upright"][window].min()),
L365:             "faults": data["faults"],
L366:             "path_mm": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
L367:             "net_mm": float(np.linalg.norm(pos[-1] - pos[0])),
L368:         }
L369:         contact = data["contact"][window]
L370:         result["contact_boundary_descriptive"] = {
L371:             "note": "Load includes adhesion; 5 ms boundary observations; transitions are not steps",
L372:             "duty": {leg: float(contact[:, i].mean()) for i, leg in enumerate(LEGS)},
L373:             "transitions": {
L374:                 leg: int(np.count_nonzero(np.diff(contact[:, i])))
L375:                 for i, leg in enumerate(LEGS)
L376:             },
L377:         }
L378:         bins = ((0.5, 2), (2, 5), (5, 10), (10, 20), (20, float("inf")))
L379:         result["speed_bins"] = {}
L380:         for lo, hi in bins:
L381:             selected = [
L382:                 [c for c in ev if lo <= c["mean_forward_mm_s"] < hi] for ev in events
L383:             ]
L384:             # Descriptive only: repeated cycles from the same deterministic run.
L385:             result["speed_bins"][f"{lo}-{hi}"] = {
L386:                 leg: {
L387:                     "cycles": len(ev),
L388:                     "retraction_s": percentiles([c["retraction_s"] for c in ev]),
L389:                     "protraction_s": percentiles([c["protraction_s"] for c in ev]),
L390:                 }
L391:                 for leg, ev in zip(LEGS, selected)
L392:             }
L393:     return result, phi, events
```


## K6. 감각·전달물질·운동 변환의 현재 가정

### `flylab/c/graph.py` — 원본 L1–L28
SHA256: `0977c12c211abd742745eddb168cb70292a3096c16a51857a5e222f0901bbda6`
```text
L1: """Immutable master roster + postsynaptic-major CSR: W[post, pre].
L2: 
L3: Counts are anatomical contacts; signed mV weights are a separate model choice.
L4: No node is inferred from edge endpoints and no isolated node is discarded.
L5: """
L6: from pathlib import Path
L7: import copy
L8: import json
L9: import re
L10: import shutil
L11: import tempfile
L12: from functools import cached_property
L13: import numpy as np
L14: from scipy.sparse import coo_matrix, csr_matrix
L15: from .integrity import canonical, digest, file_hash, read_json, write_json, finite
L16: 
L17: NAMESPACE = 'flywire:fafb:783:'
L18: SIGN = {'ACH': 1, 'GABA': -1, 'GLUT': -1}
L19: MODULATORY = {'DA', 'SER', 'OCT'}
L20: ARRAYS = ('indptr', 'indices', 'counts', 'weights')
L21: 
L22: 
L23: def external_id(root, dataset='fafb', snapshot='783'):
L24:     if not isinstance(root, str) or not re.fullmatch(r'[1-9][0-9]{0,19}', root):
L25:         raise ValueError('FAFB root IDs must be decimal strings, never JSON numbers')
L26:     if dataset not in ('fafb', 'banc') or not isinstance(snapshot, str) or not snapshot.isdecimal():
L27:         raise ValueError('Unsupported dataset namespace')
L28:     return f'flywire:{dataset}:{snapshot}:' + root
```

### `flylab/c/research_annotations.py` — 원본 L72–L86
SHA256: `ea8929dba568db86b52e88684674cb97a233ebbd00674bafb1ac389048b24721`
```text
L72: def rate_weights(graph, *, reference=None, expected_sha256=None):
L73:     """Return the full contact-based rate matrix and optional provenance audit.
L74: 
L75:     Default arithmetic matches E6/E11 exactly. No roster, adjacency, anatomical
L76:     count, original weight, or node annotation is changed, including for cells
L77:     excluded from the candidate. IDs remain decimal strings throughout.
L78:     """
L79:     validate_reference(reference, expected_sha256)
L80:     data = np.asarray(np.sign(graph.weights) * graph.counts * 0.03, np.float32)
L81:     audit = None
L82:     if reference is not None:
L83:         if (graph.manifest.get("dataset_id"), graph.manifest.get("snapshot_id")) != (
L84:             "flywire_banc",
L85:             "888",
L86:         ):
```

### `flylab/c/neuromuscular.py` — 원본 L1–L38
SHA256: `f2722ed9788b35f43a2f03fa8ed49fb8831a1eeba952c3aca5e5a218103d0e66`
```text
L1: """BANC v888 sensory–motor loop, with an explicit engineering muscle adapter.
L2: 
L3: Only the neuron/side/body-part/muscle identities are anatomical observations.
L4: Moment arms, receptor tuning and activation-to-position are model hypotheses.
L5: No imposed rhythm, gait trajectory, target bearing or body position is used.
L6: """
L7: import copy
L8: import math
L9: import numpy as np
L10: from .integrity import digest, finite, bounded_int
L11: 
L12: SOURCE = 'https://www.nature.com/articles/s41586-026-10735-w'
L13: FECO = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8665017/'
L14: FECO_CIRCUIT = 'https://elifesciences.org/reviewed-preprints/97766v1'
L15: LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
L16: PARTS = {'f': 'front_leg', 'm': 'middle_leg', 'h': 'hind_leg'}
L17: DOFS = ('coxa_yaw', 'coxa_pitch', 'coxa_roll', 'femur_pitch', 'femur_roll', 'tibia_pitch', 'tarsus_pitch')
L18: # Signed relative joint targets, not measured muscle moment arms. Multi-axis
L19: # muscles remain explicitly multi-axis. All coefficients are in the profile.
L20: MUSCLE_VECTORS = {
L21:     'sternal_anterior_rotator_muscle': {'coxa_yaw': 1.},
L22:     'sternal_posterior_rotator_muscle': {'coxa_yaw': -1.},
L23:     'tergopleural_promotor_muscle': {'coxa_pitch': 1.},
L24:     'pleural_remotor_and_abductor_muscle': {'coxa_pitch': -1., 'coxa_roll': -1.},
L25:     'sternal_adductor_muscle': {'coxa_roll': 1.},
L26:     'trochanter_flexor_muscle': {'femur_pitch': -1.},
L27:     'accessory_trochanter_flexor_muscle': {'femur_pitch': -1.},
L28:     'trochanter_extensor_muscle': {'femur_pitch': 1.},
L29:     'sternotrochanter_extensor_muscle': {'femur_pitch': 1.},
L30:     'tergotrochanter_extensor_muscle': {'femur_pitch': 1.},
L31:     'femur_reductor_muscle': {'femur_roll': -1.},
L32:     'tibia_flexor_muscle': {'tibia_pitch': 1.},
L33:     'accessory_tibia_flexor_muscle': {'tibia_pitch': 1.},
L34:     'tibia_extensor_muscle': {'tibia_pitch': -1.},
L35:     'tarsus_depressor_muscle': {'tarsus_pitch': -1.},
L36:     'long_tendon_muscle': {'tarsus_pitch': -1.},
L37:     'tarsus_levator_muscle': {'tarsus_pitch': 1.},
L38: }
```

### `flylab/c/neuromuscular.py` — 원본 L153–L165
SHA256: `f2722ed9788b35f43a2f03fa8ed49fb8831a1eeba952c3aca5e5a218103d0e66`
```text
L153: def typed_receptor_feature(kind, cell_type, angle, velocity, velocity_lowpass, load, touch, spec):
L154:     """Declared v2 receptor hypothesis. Positive tibia q means flexion."""
L155:     position=angle/math.pi
L156:     # Strict bounds retain NaN payload and signed zero, matching scalar np.clip.
L157:     position=float(0. if position<0. else 1. if position>1. else position)
L158:     if kind=='claw':
L159:         return (position if cell_type=='SNpp50' else 1-position if cell_type=='SNpp51' else 0.)
L160:     if kind=='hook':
L161:         direction={'SNpp41':1.,'SNpp39':-1.}.get(cell_type,0.)
L162:         return min(1.,max(0.,velocity*direction)/20.)
L163:     if kind=='club':return min(1.,abs(velocity-velocity_lowpass)/50.)
L164:     if kind=='load':return load/(load+spec['load_half_bw'])
L165:     if kind=='touch':return touch/(touch+spec['touch_half_bw'])
```

### `flylab/c/rate_body.py` — 원본 L489–L556
SHA256: `df2f30e912b96f38e6ddd7153b6f683fd22deac5d0f25e2c77874377889aac5a`
```text
L489:     def encode(self, *, sensory_cut=False, dt=0.005):
L490:         if not np.isfinite(dt) or not 0 < dt <= 0.05:
L491:             raise ValueError("Invalid receptor timestep")
L492:         observation = self.body.leg_observation()
L493:         q, velocity, support, touch = [
L494:             np.asarray(observation[key], float)
L495:             for key in (
L496:                 "angles_rad",
L497:                 "velocities_rad_s",
L498:                 "support_load_bw",
L499:                 "non_support_load_bw",
L500:             )
L501:         ]
L502:         if (
L503:             q.shape != (42,)
L504:             or velocity.shape != (42,)
L505:             or support.shape != (6,)
L506:             or touch.shape != (6,)
L507:             or not all(np.isfinite(x).all() for x in (q, velocity, support, touch))
L508:         ):
L509:             raise ValueError("Finite physical receptor observations required")
L510:         knee = self.body.knee_kinematics(include_direction=False)
L511:         flexion, flexion_velocity = [
L512:             np.asarray(knee[key]) for key in ("flexion_rad", "velocity_rad_s")
L513:         ]
L514:         if (
L515:             flexion.shape != (6,)
L516:             or flexion_velocity.shape != (6,)
L517:             or not np.isfinite(flexion).all()
L518:             or not np.isfinite(flexion_velocity).all()
L519:         ):
L520:             raise ValueError("Invalid measured knee geometry")
L521:         q, velocity = q.copy(), velocity.copy()
L522:         q[self.joints[:, 5]], velocity[self.joints[:, 5]] = flexion, flexion_velocity
L523:         features = np.array(
L524:             [
L525:                 anterior_feature(q[self.joints[leg, 0]], self.body.neutral[self.joints[leg, 0]], self.coxa_sign[leg])
L526:                 if kind == "coxa_limit" else typed_receptor_feature(
L527:                     kind,
L528:                     cell_type,
L529:                     q[self.joints[leg, 5]],
L530:                     velocity[self.joints[leg, 5]],
L531:                     self.velocity_lowpass[self.joints[leg, 5]],
L532:                     support[leg],
L533:                     touch[leg],
L534:                     self.spec,
L535:                 )
L536:                 for leg, kind, cell_type in self._receptor_keys
L537:             ]
L538:         )[self._port_features]
L539:         drive = np.zeros(self.graph.n, np.float32)
L540:         if not sensory_cut:
L541:             if self._sensory_unique:
L542:                 # Preserve +0 addition and float32 storage, including signed zero.
L543:                 drive[self._sensory_ids] += self.filtered[self._sensory_ports]
L544:             else:
L545:                 np.add.at(drive, self._sensory_ids, self.filtered[self._sensory_ports])
L546:         request = self.sensory_gain * features
L547:         if self.load_gain_multiplier != 1.:
L548:             request[self._load_ports] *= self.load_gain_multiplier
L549:         if self.front_hair_plate:
L550:             request[self._hair_plate_ports] = INPUT_GAIN * features[self._hair_plate_ports]
L551:         self.filtered += (1 - math.exp(-dt / 0.02)) * (request - self.filtered)
L552:         self.velocity_lowpass += (1 - math.exp(-dt / 0.03)) * (
L553:             velocity - self.velocity_lowpass
L554:         )
L555:         self.last_features = features
L556:         return drive
```

### `flylab/c/rate_body.py` — 원본 L558–L704
SHA256: `df2f30e912b96f38e6ddd7153b6f683fd22deac5d0f25e2c77874377889aac5a`
```text
L558:     def decode(self, rates, *, motor_cut=False, dt=0.005):
L559:         rates = np.asarray(rates)
L560:         if (
L561:             rates.shape != self.motor_ids.shape
L562:             or rates.dtype.kind not in "fiu"
L563:             or not np.isfinite(rates).all()
L564:             or np.any(rates < 0)
L565:         ):
L566:             raise ValueError("Finite nonnegative BANC motor rate units required")
L567:         if not np.isfinite(dt) or not 0 < dt <= 0.05:
L568:             raise ValueError("Invalid muscle timestep")
L569:         pooled = np.empty(self._pool_count)
L570:         for indices, columns in self._pool_groups:
L571:             units = rates[columns]
L572:             pooled[indices] = (
L573:                 np.mean(units, axis=1)
L574:                 if self.pooling == "mean"
L575:                 else np.sum(100.0 * units / (100.0 + units), axis=1)
L576:             )
L577:         joint_rates = pooled
L578:         if self.muscle_activation_first:
L579:             if motor_cut:
L580:                 self.muscle_activation.fill(0)
L581:             else:
L582:                 self.muscle_activation += (1 - math.exp(-dt / self.joint_activation_tau_s)) * (pooled - self.muscle_activation)
L583:             joint_rates = self.muscle_activation
L584:         vectors = self._moment_plan(joint_rates, self.motor_gain)
L585:         if self.active_length:
L586:             absolute = self._absolute_moment_plan(pooled, self.motor_gain)
L587:             if self._joint_gain_changed:
L588:                 absolute *= self.joint_gain_multipliers
L589:             stiffness_request = np.zeros(42)
L590:             for leg, joints in enumerate(self.joints):
L591:                 stiffness_request[joints] = np.tanh(absolute[leg] / .5)
L592:         rates_by_muscle = iter(pooled)
L593:         requested, adhesion, diagnostic = np.zeros(42), np.zeros(6, bool), []
L594:         for leg, groups in enumerate(self.muscles):
L595:             vector, output = vectors[leg], {}
L596:             for name, ids, moment in groups:
L597:                 rate = float(next(rates_by_muscle))
L598:                 output[name] = rate
L599:             if not self.coxa_endpoint:
L600:                 vector[0] *= self.coxa_sign[leg]
L601:             vector[5] *= self.knee_sign[leg]
L602:             if self._joint_gain_changed:
L603:                 vector *= self.joint_gain_multipliers
L604:             requested[self.joints[leg]] = 0.5 * np.tanh(vector / 0.5)
L605:             adhesion[leg] = (
L606:                 output.get("tarsus_depressor_muscle", 0)
L607:                 + output.get("long_tendon_muscle", 0)
L608:                 > output.get("tarsus_levator_muscle", 0) + 1.0
L609:             )
L610:             diagnostic.append({"leg": LEGS[leg], "muscle_rate_units": output})
L611:         if self.adhesion_model in ("activation", "graded"):
L612:             balance = np.asarray([
L613:                 row['muscle_rate_units'].get('tarsus_depressor_muscle', 0)
L614:                 + row['muscle_rate_units'].get('long_tendon_muscle', 0)
L615:                 - row['muscle_rate_units'].get('tarsus_levator_muscle', 0)
L616:                 for row in diagnostic])
L617:             request = np.tanh(np.maximum(balance, 0) / 100) if self.adhesion_model == "graded" else balance
L618:             self.adhesion_activation += (1 - math.exp(-dt / .03)) * (request - self.adhesion_activation)
L619:             if self.adhesion_model == "activation":
L620:                 adhesion[:] = self.adhesion_activation > 1.
L621:         if motor_cut:
L622:             # Enforce disconnection at the final actuator boundary, after pooling.
L623:             self.offset.fill(0)
L624:             adhesion.fill(False)
L625:             self.adhesion_activation.fill(0)
L626:             if self.active_length:
L627:                 self.total_activation.fill(0)
L628:         else:
L629:             if self.feasible_activation:
L630:                 target_request = self.body.neutral + requested
L631:                 requested = np.where(
L632:                     self._control_limited,
L633:                     np.clip(target_request, self._control_ranges[:, 0], self._control_ranges[:, 1]),
L634:                     target_request,
L635:                 ) - self.body.neutral
L636:             if self.muscle_activation_first:
L637:                 self.offset[:] = requested
L638:             else:
L639:                 activation_alpha = 1 - math.exp(-dt / self.joint_activation_tau_s)
L640:                 if self.fast_coxa_activation:
L641:                     activation_alpha = np.full(42, activation_alpha)
L642:                     activation_alpha[self.joints[:, 0]] = 1 - math.exp(-dt / .03)
L643:                 if self.fast_joint_release:
L644:                     next_offset = advance_target_activation(
L645:                         self.offset, requested, dt, self.joint_activation_tau_s,
L646:                         reversals_only=self.joint_release_mode == "reversal_only",
L647:                     )
L648:                     if self.fast_coxa_activation:
L649:                         yaw = self.joints[:, 0]
L650:                         next_offset[yaw] = self.offset[yaw] + activation_alpha[yaw] * (
L651:                             requested[yaw] - self.offset[yaw]
L652:                         )
L653:                     self.offset[:] = next_offset
L654:                 else:
L655:                     self.offset += activation_alpha * (requested - self.offset)
L656:             if self.active_length:
L657:                 self.total_activation += (1 - math.exp(-dt / .03)) * (stiffness_request - self.total_activation)
L658:         targets = self.body.neutral + (self.offset / (1 + self.total_activation) if self.active_length else self.offset)
L659:         prediction_targets = targets
L660:         if self.adhesion_model == "feasible_lift":
L661:             # Only the shadow displacement changes. Raw motor requests, servo
L662:             # limits and offset dynamics retain their original interpretation.
L663:             prediction_targets = np.where(
L664:                 self._control_limited,
L665:                 np.clip(targets, self._control_ranges[:, 0], self._control_ranges[:, 1]),
L666:                 targets,
L667:             )
L668:         lift = np.asarray(self.body.foot_kinematics(prediction_targets)["target_lift_mm"])
L669:         if lift.shape != (6,) or not np.isfinite(lift).all():
L670:             raise ValueError("Invalid local pad lift observation")
L671:         adhesion[lift > 0.02] = False
L672:         if self.adhesion_model == "request_lift" and not motor_cut:
L673:             # Release may anticipate the joint filter using a request within
L674:             # actuator control bounds. This does not predict actual clearance.
L675:             request_targets = self.body.neutral + requested
L676:             request_targets = np.where(
L677:                 self._control_limited,
L678:                 np.clip(request_targets, self._control_ranges[:, 0], self._control_ranges[:, 1]),
L679:                 request_targets,
L680:             )
L681:             request_lift = np.asarray(self.body.foot_kinematics(request_targets)["target_lift_mm"])
L682:             if request_lift.shape != (6,) or not np.isfinite(request_lift).all():
L683:                 raise ValueError("Invalid neural request pad lift observation")
L684:             adhesion[request_lift > .02] = False
L685:         if self.adhesion_model == "graded":
L686:             self.adhesion_control[:] = self.adhesion_activation * adhesion
L687:         if self.ltm_tendons:
L688:             requested_tendons = 0.1 * np.tanh(
L689:                 np.array(
L690:                     [
L691:                         row["muscle_rate_units"]["long_tendon_muscle"]
L692:                         for row in diagnostic
L693:                     ]
L694:                 )
L695:                 / 100.0
L696:             )
L697:             if motor_cut:
L698:                 self.tendon_activation.fill(0.0)
L699:             else:
L700:                 self.tendon_activation += (1 - math.exp(-dt / 0.03)) * (
L701:                     requested_tendons - self.tendon_activation
L702:                 )
L703:         self.last_motor = diagnostic
L704:         return targets, adhesion
```

### `configs/banc-walking/m2-calibrated-v1.json` — 원본 L1–L18
SHA256: `d3664dedd79d8dd97ee0355156ec4c5a31347c6112883aa60daef7fc08830b66`
```text
L1: {
L2:   "neural_dt_s": 0.00025,
L3:   "coxa_geometry": true,
L4:   "ltm_tendons": true,
L5:   "adaptation_gain": 8.0,
L6:   "adaptation_tau_s": 0.3,
L7:   "joint_activation_tau_s": 0.09,
L8:   "fast_coxa_activation": true,
L9:   "joint_gain_multipliers": [
L10:     1.0,
L11:     1.0,
L12:     1.15,
L13:     1.0,
L14:     1.0,
L15:     1.0,
L16:     1.0
L17:   ]
L18: }
```


## K7. E353/E358/E359 결과: 보고된 수치와 판정 범위

### `docs/BANC_M3_E353_REFERENCE_CADENCE_DESIGN_20261002.md` — 원본 L1–L34
SHA256: `d6198dfc7f25f1e4af2c060fbe1fde742e92a16b5cd69cd5e7f271166a690a35`
```text
L1: # E353: 같은 신체·형상에서 기준 cadence 하나만 바꾸는 대조
L2: 
L3: ROOT design v1. 사용자 지시를 그대로 적용한다: 현재 세션 설계 → 실제 GPT-6 Astra/max 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol/high 구현 → ROOT FIRST → 결함이 있으면 같은 Sol의 ONE R1 → ROOT SECOND → 잔여 ROOT 수정 → 다음 단계. 높은 effort는 원본 접근 gate, 실제 신체 관측, 실패·자원·증거 보존의 복잡성 때문에 선택한다. 활성 목표는 `BANC_NATURAL_GAIT_M3_GOAL_PLAN_20260919.md`의 동일 BANC 후보 F1–F8 전체 검증이다. 이 실험은 교사 진단이며 목표 완료나 생물학적 기전 증명이 아니다.
L4: 
L5: ## 근거와 사전 질문
L6: 
L7: E351 별도 자격을 얻은 보존 대조 C는 전진1.8140822469961801mm, 하중 가중 slip1.6725988217719747mm/s, 가장 긴 전진 bout.36s, RM1회/다른5다리0회다. 원 E350 worker의 직렬화 technical-invalid는 보존한다. E352는 원본 신체·신경을 추가 실행하지 않고 C의 LF 출력 embedding 위상을 검산했다. 두 eligible run의 순진행량.793130995252755/.6319260409217811회전, 최대5ms 변화.2391001089207736회전, whole-window 발 geometry pair 대부분이 이동 속도 단절 구간을 가로지른다. 이는 인과 책임이나 신경·발의 실제 위상 직접 측정이 아니다.
L8: 
L9: H353: 각 다리 reference 형상·전달·실제 신체·다른 입력을 그대로 두고, 공유 LF source embedding의 불규칙한 진행을 기존 reference의 일정한 cadence로만 바꾸면 같은 짧은 구간에서 연속 전진과 여섯 다리의 인정 주기가 생기는가? 이 대조는 일정 cadence라는 하나의 복합 timing policy 변경의 효과만 검토한다. 정체/역행/불연속 각각의 원인을 분리했다는 결론은 금지한다. 기존 C는 자격 있는 보존 비교로 사용하고 B/I/C를 다시 실행하지 않는다.
L10: 
L11: ## 고정 intervention
L12: 
L13: frequency는 이미 E332에 고정된 **2.382760007990865Hz** 하나다(`BANC_M3_E332_SMOOTH_TEMPORAL_DESIGN_20261001.md`, `tools/analyze_c_banc_m3_saved_gait_transfer.py` 고정 INC). 값을 이번 성능으로 고르거나 주파수·leader·offset·gain sweep을 하지 않는다. control dt.005/native dt.0001, 총300controls/15000science steps/1.5s, canonical FlyBody 모델1b2a8908, seed42/복원된 실제 initial body state/constructor settling2000steps를 유지한다.
L14: 
L15: 허용된 원본 job 안에서 source control/reference를 읽고 기존 Gaussian encoder로 c100의 LF phase phi_LF100을 계산한다. 사전 숫자 조회나 최적 anchor 선택은 없다. 고정 phase는 `phi_R[c]=(phi_LF100+frequency*(c*.005-.5)) % 1`이다. c0..100 명령은 source와 finite bytes exact. c101..119는 기존 `x=(c*.005-.5)/.1; w=x*x*(3-2*x)` smoothstep으로 source 명령과 동일 reference target을 섞는다. c>=120은 완전 reference target이다. reference curves[6,128,7]의 anatomical offsets는 이미 포함되므로 여섯 다리에 같은 phi_R만 넣으며 추가 offset0. 모두 기존 cyclic linear interpolation이고 shape/gain/조인트 순서42 변경0이다. 기존 source201x6 phase/reliability 검증 R>=.5를 유지하며 새 계획의 anchor·phase·명령을 저장한다.
L16: 
L17: 다른 action은 각 control의 source tendon8/원래 adhesion/default omitted flags/explicit neural_request None/held push_left0·push_force[0,0,0]를 그대로 적용하고 실제 before/after와 ABI receipt를 얻는다. joint target42만 위 식으로 바꾼다. neural/GraphStore/model/CPG advance/GPU 호출0. instance force/pose dependencies를 새 output에서 사용한다. 실제 inherited dependency가 첫 신체를 arm `B`로 표기하므로 이를 source dependency label로 명시하고 새 실험 arm `R`과 구분한다. 새 결과에서 과거 acquisition label/자격/false provenance를 변경하지 않는다.
L18: 
L19: ## 실제 관측·순서·partial
L20: 
L21: 새 single-arm loop는 frozen core `replay`의 constructor/configure/restore/action/boundary/native observer/counter/cleanup 의미를 그대로 따른다. old full B/I/C `run`/`replay`의 publisher나 E350 global publication_scope를 실행하거나 old module globals를 monkeypatch하지 않는다. pure plan/selected reader/native functions, E349 held-force instance methods, E350 SourcePoseDependencies는 SHA 고정하여 재사용한다. 재사용 routine의 검증 범위를 정확히 쓰고 새 orchestration 전체가 기존 AST proof로 검증됐다고 말하지 않는다.
L22: 
L23: 첫 native step부터 source live `body.pose()`(live mj_forward0)와 별도 shadow `type(shadow).pose(shadow)` after shadow-only mj_forward, 실제 native index/physics_time, contact/load/slip/current nonfoot/numeric fault를 함께 기록한다. first read/callback failure의 acquired step과 unavailable mask를 보존한다. 신체1개·constructor2000steps·원래 q/qv/ctrl/config/modelhash/clock/state를 관측한다. c101을 적용하기 **전에** source와 실제 rows0..100 및 native1..5000의 여섯 필드를 exact 비교해야 한다. 과거 B 전체 exact는 E351 qualified dependency로 연결하되 이번 100prefix 실제 exact 검사를 대신하지 않는다. prefix 실패는 technical이며 c101 미적용.
L24: 
L25: postprefix physical health stop은 scientific partial(공식 full metrics/H353 not_evaluated), constructor/action/clock/nonfinite/read/deadline/cleanup/publication failure는 technical-invalid다. health-stop 이전/도중 실제 requested/returned control·native·callback·observations 수와 미반환 control을 남긴다. padding/full length/예측 남은 frame/미취득 값0 대체 금지. CPG state 동일과 observer/method restore/body close 실제 결과를 보존한다. first error 이후 secondary 오류와 늦은 실패가 성공 판정보다 우선한다. source·current upright>=.9, nonfoot0, fault0 기준은 그대로다.
L26: 
L27: ## 저장 형식과 전체 cap
L28: 
L29: E349 failed full trace 용량 초과와 E350 NumPy bool 오류를 반복하지 않도록 새 publisher를 명시적으로 구현한다. 모든 boundary row301(부분실패면 실제 수), qpos/qvel/ctrl/action/42commands/6x8x3geometry 및 원래 metadata를 **손실 없는 `trace.json.gz`** 하나로 저장한다. compact strict JSON을 exact scalar lowering 후 gzip(mtime0)으로 압축한다. full boundary dictionary를 누락·반올림·quantize·downsample하지 않는다. compression은 임의 sample 삭제가 아니며 압축 전 JSON은 최대16MiB/메모리64MiB다. 압축된 bytes/hash와 압축 전 bytes/hash 모두 descriptor에 쓰며 auditor는 bounded decompression/CRC/strict JSON/row schema·length를 검증한다. frozen gait loader arithmetic은 cached decoded payload로 적용하고 filename은 압축 경로를 정직하게 기록한다.
L30: 
L31: native-contact.npz는6개 기존 arrays, source-pose.npz는7개 실제 time/index/upright/masks, cadence-plan.npz는 source z/R/phi와 phi_R/명령 및 기존 reference provenance를 저장한다. np.savez_compressed byte buffer로 header/dtype/Corder/shape/selected member descriptors와 acquisition digest를 실제 생성 시 고정한다. actions/force/observation/constructor/science/cleanup/error ledger는 strict finite JSON receipt로 저장한다. force policy도 새 output 전체 cap에 포함한다. reader cache/staging64MiB, host2GiB.
L32: 
L33: fresh output `verification/banc-natural-gait-m3-20260919/e353-reference-cadence-body`의 **모든 성공/부분/기술실패 artifacts 합계<=4MiB**(기존 실패cap을 완화하지 않고 완료cap16MiB보다 엄격하게 사용). raw pack을 메모리에서 완성해 전체 byte 합계·남겨둔 receipt/failure 여유를 검사한 뒤 exclusive create/flush/fsync/close한다. 저장 도중 오류는 이미 쓴 자료를 삭제하지 않고 secondary failure를 남긴다. cap 예측 초과라면 미게시 raw의 bytes/hash·실제 acquired count와 bounded 실패 receipt만 게시하여 technical-invalid로 남기며 full saved proof라 부르지 않는다. 임의 truncate/delete/캡 확대/후속 partial→full 재분류는 없다. generated full301 stress에서 lossless roundtrip과 actual total cap을 확인하고 incompressible oversize/late write/fsync/error 경로도 검증한다. output stage<=64MiB, own report<=4MiB, failure metadata reserved<=64KiB.
L34: 
```

### `verification/banc-natural-gait-m3-20260919/parent-e353-final-decision.json` — 원본 L1–L133
SHA256: `ca4ca3e038aad892c3251d2c64d1ed4d0a2434d91d44942fd00dd451552a927a`
```text
L1: {
L2:   "schema": "flylab.m3.e353.final-diagnostic-decision.v1",
L3:   "diagnostic_completion_qualified": true,
L4:   "H353_supported_for_later_learning_eligibility": true,
L5:   "independent_report": {
L6:     "path": "verification/banc-natural-gait-m3-20260919/e353-reference-cadence-audit/report.json",
L7:     "bytes": 72734,
L8:     "sha256": "07f13dc2e2dd68c97b4eb9744bfa003bd6bccf3c52032cb4b17d91831a936e01"
L9:   },
L10:   "actual_run_full_return": {
L11:     "path": "verification/banc-natural-gait-m3-20260919/parent-e353-run-full-exit.json",
L12:     "bytes": 2813,
L13:     "sha256": "4888d98f40bc4e42361be2bcd3cdf2d28c8da25ddf8844cc1da1f4445af39251"
L14:   },
L15:   "actual_audit_full_return": {
L16:     "path": "verification/banc-natural-gait-m3-20260919/parent-e353-audit-full-exit.json",
L17:     "bytes": 2819,
L18:     "sha256": "32f1376307a6944b7c70ab23c05a0756edcb446ad9ac8598a2989e3633f3af38"
L19:   },
L20:   "acquisition_seal": {
L21:     "path": "verification/banc-natural-gait-m3-20260919/parent-e353-closed-worker-seal.json",
L22:     "bytes": 3142,
L23:     "sha256": "c29b9bcf814078d1d0b6293e6bc8191a82b0f2654d071ddb2c11ab83312335e8"
L24:   },
L25:   "source_sha256": {
L26:     "tools/probe_c_banc_m3_reference_cadence_body.py": "50abe27390ba10dfb5c0331bb73c005d109bebcd379b5feda75063b39f73e6be",
L27:     "tests/test_c_banc_reference_cadence_body.py": "c0a1b4cb8bd42dbbf78c1d7e86e0a2a31f9174689f6d8fc25c10e4c6b1465fab"
L28:   },
L29:   "workflow_completed": "ROOT design > actual Astra6max review/reflection > actual Sol6.1high FIRST > ROOT FIRST > SAME Sol ONE R1 > ROOT SECOND > ROOT residual > current18CPU/Ruff/freeze > once actual R > once independent audit",
L30:   "actual_Astra_review": {
L31:     "path": "verification/banc-natural-gait-m3-20260919/parent-e353-astra-design-review.json",
L32:     "bytes": 18791,
L33:     "sha256": "67228c101c9a4d4faab37bf77339ae869e6de2abab2695dfeb2406403e8ff25f"
L34:   },
L35:   "actual_Sol_FIRST": {
L36:     "path": "verification/banc-natural-gait-m3-20260919/e353-sol-first-submission.json",
L37:     "bytes": 96609,
L38:     "sha256": "d9a7c8d0c12ce5d4e9de2d5ba249de569fa2c2733c612a7f0bd859be63bac029"
L39:   },
L40:   "actual_Sol_R1": {
L41:     "path": "verification/banc-natural-gait-m3-20260919/e353-sol-R1-submission.json",
L42:     "bytes": 2881,
L43:     "sha256": "558eaae39441e0adbbe2ac244162feb657f05a46f338d0095a81fb206ac93db1"
L44:   },
L45:   "ROOT_FIRST": {
L46:     "path": "verification/banc-natural-gait-m3-20260919/parent-e353-root-first-review.json",
L47:     "bytes": 2530,
L48:     "sha256": "329a5920d24ea4fd7a15ca1b4004a6b81f28f4a19cf9617c1c9284072668db87"
L49:   },
L50:   "ROOT_SECOND": {
L51:     "path": "verification/banc-natural-gait-m3-20260919/parent-e353-root-second-static-review.json",
L52:     "bytes": 887,
L53:     "sha256": "7830fb34b8329b7cee0d0cdbfaa070b831d6e1a62c66084de645f741296a28c5"
L54:   },
L55:   "ROOT_residual": {
L56:     "path": "verification/banc-natural-gait-m3-20260919/parent-e353-root-residual-correction.json",
L57:     "bytes": 577,
L58:     "sha256": "8ded9f0b91d4723d5427dc4d08e6562faf4ea1b4b73ab56b8eeb151a69877538"
L59:   },
L60:   "current_ROOT_qualification": {
L61:     "path": "verification/banc-natural-gait-m3-20260919/parent-e353-root-qualification.json",
L62:     "bytes": 4047,
L63:     "sha256": "6a1262a1d45fa0df07f13a407328a9d679fc5ce33f4316289a365509d6f73955"
L64:   },
L65:   "current18CPU": true,
L66:   "ROOT_generated_inclusive_s": 17.9420532,
L67:   "Sol_generated_inclusive_s": 16.8818899,
L68:   "generated_limit_s_each": 20,
L69:   "fixed_frequency_Hz": 2.382760007990865,
L70:   "verified_metrics": {
L71:     "forward_mm": 1.9749839267771232,
L72:     "slip_mm_s": 1.399688502351197,
L73:     "longest_forward_bout_s": 0.995,
L74:     "complete_cycles": {
L75:       "lf": 2,
L76:       "lm": 2,
L77:       "lh": 1,
L78:       "rf": 2,
L79:       "rm": 1,
L80:       "rh": 2
L81:     },
L82:     "measured_dual_healthy": true,
L83:     "slip_missing_count": 0,
L84:     "independent_command_error_rad": 0.0
L85:   },
L86:   "H353": {
L87:     "verdict": "supported_for_later_learning_eligibility",
L88:     "supported": true,
L89:     "gates": {
L90:       "absolute_forward": true,
L91:       "absolute_slip": true,
L92:       "bout": true,
L93:       "six_cycles": true,
L94:       "dual_health_and_engineering": true
L95:     }
L96:   },
L97:   "original3_new4_once_read_counts": {
L98:     "verification/banc-natural-gait-m3-20260919/e335-smooth-secant-body/control-raw.npz": 1,
L99:     "verification/banc-natural-gait-m3-20260919/e335-smooth-secant-body/native-contact.npz": 1,
L100:     "verification/banc-natural-gait-m3-20260919/e332-smooth-temporal-saved/reference.npz": 1,
L101:     "verification/banc-natural-gait-m3-20260919/e353-reference-cadence-body/trace.json.gz": 1,
L102:     "verification/banc-natural-gait-m3-20260919/e353-reference-cadence-body/native-contact.npz": 1,
L103:     "verification/banc-natural-gait-m3-20260919/e353-reference-cadence-body/source-pose.npz": 1,
L104:     "verification/banc-natural-gait-m3-20260919/e353-reference-cadence-body/cadence-plan.npz": 1
L105:   },
L106:   "real_body_calls": 1,
L107:   "real_constructor_steps": 2000,
L108:   "real_scientific_steps": 15000,
L109:   "real_boundaries": 301,
L110:   "body_retry_count": 0,
L111:   "audit_body_rerun": false,
L112:   "actual_body_worker_full_return_s": 18.8660253,
L113:   "actual_independent_audit_full_return_s": 3.2023443,
L114:   "charged_original_total_once_s": 22.0683696,
L115:   "original_cumulative_before_s": 9846.861104799502,
L116:   "original_cumulative_after_s": 9868.929474399501,
L117:   "original_cumulative_limit_s": 9900,
L118:   "original_remaining_s": 31.070525600498513,
L119:   "original_reserved_s": 0,
L120:   "original_jobs_running": 0,
L121:   "raw_all_worker_bytes": 2835675.0,
L122:   "raw_worker_cap_bytes": 4194304,
L123:   "neural_CPG_GPU_calls": 0,
L124:   "successful_teacher_timing_is_not_neural_autonomy": true,
L125:   "candidate_adopted": false,
L126:   "F1_F8_complete": false,
L127:   "biological_mechanism_proven": false,
L128:   "natural_gait_complete": false,
L129:   "M2_UI_F6_changed": false,
L130:   "E350_original_technical_invalid_and_H348_failure_preserved": true,
L131:   "next_stage": "ROOT design for transferring qualified R cadence into anatomy-constrained BANC dynamics without runtime clock/phase; actual Astra6max review before Sol6.1 implementation; remaining original31.0705256s never reset or extended",
L132:   "scientific_raw_reads_by_decision_writer": 0
L133: }
```

### `docs/BANC_M3_E358_CANDIDATE_BODY_RESULT_20261003.md` — 원본 L1–L13
SHA256: `4e06b71ede2666590b76e932294644c4eaf8369a19cb94d1e6d31bcbb5d58a42`
```text
L1: # E358 후보 계산과 동일 후보 신체 실행 결과
L2: 
L3: 2026-10-03. 전체 M3/F1–F8은 미완료다. 후보의 저장 입력에 대한 신경 응답은 조건부 개선했지만, 동일 후보의 실제 신체 실행은 독립 자격을 얻지 못했다. M2와 UI에 채택하지 않았다.
L4: 
L5: 후보 계산은 실제 전체 반환23.289초, 별도 raw 감사2.372초로 실행 당시 source snapshot 아래 유효했다. 실제 f32 계수 변경, native palette 한도, 신경 work 및 저장·감사 조건을 확인했다. R 개선0.00887903942174223, O/Q 상대 오차0.0009343804209771942, O/T0.0009435436453516087이었다. 이는 f64 대리 모델과 실제 f32 응답의 조건부 일치이며, native f32 전체 Jacobian이나 실제 보행 개선을 증명하지 않는다.
L6: 
L7: 신체 실행은48.295초/exit1이었다. 실제99 controls,1980 neural ticks,4950 physics steps와100 boundary rows까지 반환했다. 생산자 startup 보고는 q RMS0.017792953582825483, q max0.04220501637004681, tendon RMS0.0013636523143322314, tendon max0.013189925135820152이었다. 고정 q max0.04를 초과해0.495 모델초에서 중단했다. 이후 imported-source closure 검사가 실패해 manifest publication을 마치지 못했으므로 **technical_invalid/unqualified**이다. 이 수치는 독립 body 감사 결과가 아니며, full1.5초·10초·F1–F8을 통과한 결과로 사용하지 않는다. 관측된 모든 소유 프로세스는 종료했고 예약을 해제했다.
L8: 
L9: ROOT는 등록된 installed-source absolute path와 같은 파일의 ROOT-relative import key 사이47개 alias 누락을 소스에서 찾았다. 같은 등록 SHA의 동일 파일 alias만 인정하고 collision을 거부하도록 수정했다. 이후 실패에서는 actual imports와 mismatch를 먼저 남긴다. 이 수정 후19개/57phase 회귀와 별도 실제 supervisor→저장·fsync·종료→reread→실제 반환 fixture가 닫혔다. 실패한 원본 body를 반복하거나 소급 유효화하지 않았다. 실행 당시 adapter/test source snapshot을 별도로 보존했다.
L10: 
L11: 기존 원본9893.754960899503초에 E358 실제78.18초를 더한9971.934960899503/10200초, 공동stage78.18/300초를 사용했다. 공유 생성 누적99.7460156/150초이며 모든 실패와 helper 비용을 포함한다. 변경 후보 retry는 사용하지 않았다. 다음 단계는 남은 같은 예산에서 theta=1 기준 모델의 startup99 가능성을 별도로 사전 등록해 확인하는 [E359 계획](BANC_M3_E359_PAIRED_STARTUP_BASELINE_PLAN_20261003.md)이다.
L12: 
L13: 기계 판독 결과와 실행 당시 provenance: [ROOT 결과](../verification/banc-natural-gait-m3-20260919/parent-e358-root-results-before-e359.json), [실행 원장](../verification/banc-natural-gait-m3-20260919/parent-e358-execution-ledger.json). 실제 goal API 상태 blocked와 최신 사용자 요청에 따른 현재 작업 재개를 구분한다.
```

### `docs/BANC_M3_E359_BASELINE_RESULT_20261003.md` — 원본 L1–L21
SHA256: `b85fb7b42f14896de88c47b86040b6932fad743958881cfa8b49f54d9a0991d2`
```text
L1: # E359 기준 모델 결과
L2: 
L3: 기준 모델의 actual99-control 구간은 완료됐고 생산자 초기 조건은 PASS였다. 다만 실제 import된15개 프로젝트 소스가 사전 목록에서 빠져 **전체 실행은 technical_invalid/unqualified**다. 정상 manifest나 독립 baseline 감사는 없으며 자연 보행 완료가 아니다.
L4: 
L5: | 항목 | 실제 기록 |
L6: |---|---|
L7: | 기준 모델 실행 | 46.640초 / exit1 / original1회 |
L8: | 알려진 실행량 | constructor1, settling2000, neural setup320/capture160, controls99/neural1980/physics4950/rows100 |
L9: | 생산자 q RMS / max | 0.01772869163371381 / 0.0396360246919627 |
L10: | 생산자 tendon RMS / max | 0.001363737833496342 / 0.013190561316255553 |
L11: | 첫 오류 | neutral-actual-imports: actual neutral body imported source closure |
L12: | 부분 증거 | command/tendon100 rows, native4950, trace100, progress, transfer, actual123 imports/15 mismatches |
L13: | 종료 | 모든 owned PID 종료, NVML thread CLOSED/errors0, 예약·jobs0 |
L14: | 새 코드 검증 | whole22 cases/66 phases3.316초, 별도 real fixture1.633초, currentQ fsync/close/reread |
L15: | 누적 비용 | 원본10020.115960899504/10200, 공동126.361/300, 생성110.9570156/150,42 debits |
L16: 
L17: ROOT 설계→실제 Astra6/max→ROOT 반영→실제 Sol6.1/high FIRST→ROOT FIRST→SAME Sol ONE R1→ROOT SECOND→ROOT 잔여 수정 순서를 수행했다. Ruff 실패0.563/0.566, final0.559와 historical suite cost ID 충돌 전 준비0.781의 비용·실패도 보존했다. 과학·native·trial·E267/E346 및 이전 E358 source/Q는 변경하지 않았다.
L18: 
L19: 15개 누락은 common_descent_body→common_descent_direction 하위 프로젝트 의존성이다. 현재15개 source bytes는 실행 당시 기록 SHA와 같지만 사전 등록이 없었던 사실을 소급 수정하지 않는다. E359 currentQ/원본 실행의 소스 snapshot과 machine report `verification/banc-natural-gait-m3-20260919/parent-e359-root-closed-result.json`을 보존했다. producer startup PASS는 독립 감사 결과가 아니다. missing theta/selected/network-tick raw와 원본 publication binding의 한계를 유지한다.
L20: 
L21: 다음은 [E360 소스 의존성·보존 raw 진단](BANC_M3_E360_SOURCE_CLOSURE_RAW_DIAGNOSTIC_PLAN_20261003.md)이다. 새 소스 사전 검사와 독립 raw-only 프로세스로 제한된 진단을 수행하고 같은 기준 body를 다시 만들지 않는다. E359 판정/F1–F8/M2/UI/F6/과거 비용·실패를 그대로 두며, 이후 startup-constrained neural update 및 실제1.5s/10s·자연 보행은 별도 단계다.
```


## K8. 시간 제한·현재 승인·환경 핀

### `docs/BANC_M3_E360_ORIGINAL_CPU_TIMING_RISK_20261003.md` — 원본 L1–L18
SHA256: `a372fca49495ef6a70b5c3bcaee93e51358225129d63cdee8e679b6499ff27f5`
```text
L1: # E360 original CPU preparation timing risk
L2: 
L3: ROOT source/metadata analysis only, 2026-10-03. No original payload, Python/runtime or original job is executed by this note.
L4: 
L5: Actual generous p2 metadata helpers returned fixture-build6.615s, fixture-seal5.146s, suite-build7.178s, suite-seal5.204s. The fixture seal's same-ENTRY completion is4.7172241s. These are generated calls, not measured original durations. Current helper `check_call_request` invokes a fresh actual `scan_current` before every non-Ruff/scanner helper, including original seal; cold source work is within ENTRY.
L6: 
L7: The original policy is unchanged: build external5/internal4.5, worker12/11/10/9, seal external2/internal1.5, aggregate local20, max original worker1, automatic0. Its seal target is materially shorter than the observed generated seal path. A passed p3/Q cannot demonstrate that original preparation/closure fits. Old startup99 data and old producer thresholds do not resolve this timing issue.
L8: 
L9: No original work may be warmed for free, reclassified as generated, moved outside ENTRY, reattempted automatically, or qualified on a late helper. Each actual failed preparation is charged even at raw0. Worker actual return must be recorded before seal; a late or failed seal still charges both actual worker/helper costs and preserves an unqualified closed/fallback report. Unknown owned identities block later work.
L10: 
L11: After a current p3 qualification, the next decision must use fresh ledger balances and this risk. Source inspection/optimization is permitted within the authorized implementation scope while retaining actual-first process-only AST discovery, fresh source/layout/oracle/installed/authority validation and all scientific criteria. An original cap change would be a separate concrete proposal; the same-budget p3 amendment grants none. Successful infrastructure or partial raw publication never qualifies historical E358/E359 or natural gait.
L12: 
L13: 
L14: 2026-10-03 continuation: p3 failed administrative Ruff classification, p4 failed historical PS-source admission; both actual costs preserved in exact62. A separate same600/489 source-specific p5 authority10441B/SHA0c236d980577b8985166ef9cf7a22b5e18cb5d37a73421bc444a027ca3efaae7 and actual Astra FINAL add another fixed history context; new Sol source integration is in progress. These steps do not widen original clocks or establish original feasibility. Current original local0/20, joint126.361/300/global10020.115960899504/10200. After actual current fixture/suite/Q, use fresh timings and ledger to make the original decision; no knowingly late seal or uncharged warming is qualified. Natural gait/F1-F8 remain incomplete.
L15: 
L16: 2026-10-04 actual p6 evidence: generated scanner12.744exit0, fixture build17.008exit0/internal16.598855, fixture worker23.293exit1 at pre_array same parent ENTRY science16 before numeric/data, fixture seal9.804exit0/internal9.3610161 and unqualified closed. All actual68 costs preserved, original local0/20 remains. New ROOT fixed-authority-scope source design/Astra review proposes reducing repetition within actual checked boundaries, not free warming or clock resets. Its benefit is unmeasured. Original build5/4.5 and seal2/1.5 remain unsupported by current timings; original feasibility still requires actual fresh suite/Q and a separate concrete decision. No original cap change or original access is performed by this note.
L17: 
L18: 2026-10-04 P7 actual timing update: fixture build full7.975/internal7.145715899998322; seal full6.347/internal5.475078899995424; scanner full6.318/internal unmeasured. Fixture worker full21.473 exit1, firsterror internal20 child-diagnostic, envelope elapsed20.185280599980615, fully ended/released/unqualified. No exact expensive subphase inference. Original5/4.5 build and2/1.5 seal cannot knowingly be admitted on these observations; original12/11/10/9 and local20/stage300/global10200 remain unchanged. Same-total generated P8 prospective design273 with fixture55/50/46/42 is in source implementation, no timing-fit evidence. New budget proposal for originals follows only actual generated fixture, full suite and Q qualification/debit.
```

### `verification/banc-natural-gait-m3-20260919/parent-e360-p10-human-authorization.json` — 원본 L1–L1
SHA256: `a8e4d13fecaea1f8d4df53bcfb35d6a4b58705c94837f90de460695ddba79e80`
```text
L1: {"schema":"flylab.m3.e360.p10-lint-recovery-human-authorization.v1","date":"2026-10-04","source":"Direct human message in current session","human_quote":"추가 예산 없이 남은 한도에서 P10을 한 번 진행하도록 승인할게","manual_generated_pass_id":"e360-generous-p10","manual_P10_authorized":true,"new_generated_grant_s":0,"existing_shared_limit_s":1400,"existing_local_limit_s":1289,"reserved_whole_generated_schedule_s":798,"automatic_retries":0,"P11_authorized":false,"original_execution_authority":false,"scope":"One modified manual P10: reviewed lint fixes and capture, Ruff then scanner fixture full suite qualification; actual once debit or earlier failure closes scope","approved_proposal":{"path":"docs/BANC_M3_E360_P10_REFLECTED_SCOPE_PROPOSAL_20261004.md","bytes":11080,"sha256":"4cf503b986e36741649f4fc1420f1c82888690bf4704d965f3adee4721628caa"},"inherited82":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-inherited82.json","bytes":92934,"sha256":"7322e6b355f255db1e7f5fbceb12044eec88f979de86cd9c65dfb7c5b411dd67"},"baseline_live_ledger_sha256":"8735eae1e690d2747828545a500022896e836c467d58c7384439676a2eb51a7f","baseline_debits":82,"baseline_shared_used_s":503.0620156,"baseline_local_used_s":392.105,"preserve":"All82 actual costs,14 activation/amendment fields,both previous grants,P9 first Ruff failure,F1-F8,M2/UI,original20/300/10200,candidate1/retry0. No fresh grant, cost reset or natural gait claim."}
```

### `docs/BANC_M3_E360_P10_CAPTURE_PREPARATION_REFLECTED_20261004.md` — 원본 L1–L13
SHA256: `682ba1a0fb12c7f64a26d947deb0815a343ec9f47fd0f6fcabb923b35f0e881e`
```text
L1: # P10 출력 보존 사전 구현 — 실제 Astra 리뷰 반영
L2: 
L3: 기준 설계는 [사전 구현 설계](BANC_M3_E360_P10_CAPTURE_PREPARATION_DESIGN_20261004.md)다. 실제 위임된 `gpt-6-astra/max`의 최종 in-session 설계 리뷰는 `NEEDS NARROW CORRECTIONS BEFORE OFF-ACTIVE FIRST`였다. 파일 publication은 완료하지 않았으며 ROOT가 실제 최종 메시지와 아래 반영을 별도 JSON으로 기록한다. 독립 모델 introspection이나 구현/런타임 통과를 주장하지 않는다. 이 문서는 구현 사본 준비만 허용하며 P10 실행 승인이 아니다.
L4: 
L5: 1. ROOT와 실행 경로를 고정한다. WorkingDirectory는 고정 repository ROOT이고 FileName은 ROOT 아래 `.venv/Scripts/ruff.exe` 절대 경로다. ArgumentList에는 `check`와 기존 세 target만 사용하며 UseShellExecute=false와 두 redirect를 유지한다. call row의 실제 필드는 begun, sequence_index0, reserved3, future795, cost_id=null, attempt_id=null이며 pass.next_index는 callbegin 뒤 **1**이다. pass.failed_cost=null과 request.action=ruff를 확인한다. source/concrete_fix/direct authority/v2 extension을 live row/request/pass 및 독립적인 ROOT freeze 입력과 대조한다. 전체82/14 fields/두 grant 인증은 미래 ROOT admission에 남긴다.
L6: 
L7: 2. 두 pipe마다 outstanding ReadAsync는 하나이고 IsCompleted일 때만 결과/예외를 관측한다. byte count0만 EOF이며 read fault는 terminal incomplete다. 다른 pipe는 계속 처리한다. overflow/write failure 뒤 보존을 중단하고 drain은 계속한다. read 재시도나 미완료 Result/무한 wait는 없다. 오류는 고정 stage/stream slots로 제한하고 각 메시지는256자 이하로 제한한다. first_failure는 고정 stdout→stderr→process 순서에서 처음 관측한 사건과 monotonic 시각으로 한 번만 정한다. exception 문자열이나 오류 배열을 무제한 축적하지 않는다. 실제 JSON escaping을 포함한 UTF-8 manifest16KiB와 compact summary8KiB 제한을 확인한다. $PID 같은 시스템 변수를 쓰지 않는다.
L8: 
L9: 3. entry clock은 함수 첫 statements에서 잡아 guard/파일 준비 시간을 포함한다. Process.Start 직전 entry+2초 이상이면 프로세스를 시작하지 않는다. entry+2초에 alive identity를 확인한 직접 process에만 한 번 종료 요청하고 entry+2.5초에 비차단 관측을 끝낸다. finally는 둘째 CreateNew 실패를 포함해 열린 file/pipe/process handle과 pending reads를 정리한다. 무제한 await는 없다. 확인한 EOF/exit 증거는 보존하며 Dispose를 종료 증거로 쓰지 않는다. identity/exit 불명은 unresolved ownership이고 미래 ROOT가 실제 차감 및 필요한 hold를 처리해야 한다. fsync/OS 지연의3초 초과를 없애거나 child-only 시간으로 대체하지 않는다.
L10: 
L11: 4. fixed expected_output JSON은 진단 manifest다. request.completion helper 파일은 만들지 않는다. manifest의 capture_complete는 stream/child 관측 결과만 나타내며 이후 manifest publication 성공이나 자기 hash를 선기록하지 않는다. actual Ruff exit는 별도 필드로 보존하고 exit1은 capture가 완전해도 실패다. manifest CreateNew→Flush(true)→close→독립 bytes/hash 검증 뒤 compact 반환 envelope에 publication_verified와 descriptor를 남긴다. ROOT 성공에는 complete stream capture, manifest publication_verified, Ruff exit0과 actual external full-return3초 이내가 모두 필요하다. 어느 하나라도 unknown/false이면 진행하지 않는다.
L12: 
L13: 별도 파일 `parent-e360-p10-capture-proposed.ps1`에 함수만 구현한다. 이 파일의 dot-source/실제 함수 호출/가짜 admission/실제 Ruff/compile/AST/Python/helper/tests/probes는 이번 준비에서 하지 않는다. native PS source parse와 source/hash 리뷰만 허용한다. 현재 P9 own9와 ledger82를 유지하고 FIRST/R1 사본과 인계만 저장한다. 기본 설계의 모든 cap/직접 identity/stream별1MiB/총2MiB+16KiB/소스 핀/새 grant0/F1–F8 제한은 그대로다.
```

### `requirements.txt` — 원본 L1–L5
SHA256: `9004ee39dac58b265fcd44c424b4fb2a8d44061368aea3ac429602572291304a`
```text
L1: # Python >=3.12,<3.15; Git required for the pinned upstream source.
L2: flygym @ git+https://github.com/NeLy-EPFL/flygym.git@ca65a510c2afe6ac61c51df4f274c8d190c2f95f
L3: mujoco>=3.9,<3.10
L4: numpy>=2,<3
L5: aiohttp>=3.12,<4
```

### `requirements-c.txt` — 원본 L1–L5
SHA256: `5472a5a851c8f70c6b1391cc7177eccbbec40543f86edc66fb187993a0e941e1`
```text
L1: -r requirements.txt
L2: scipy>=1.14,<2
L3: numba==0.67.0
L4: # Optional NVIDIA dependencies: requirements-cuda.txt (isolated CUDA runtime).
L5: # The CLI defaults to auto; --backend exp_lif_cpu_reference explicitly uses CPU.
```

### `requirements-research.txt` — 원본 L1–L4
SHA256: `a03ad6b14b9c7700c8bd743eed9e0d0df940a7cb93e9b06d90722117338e450b`
```text
L1: -r requirements-c.txt
L2: # ResearchLIF supports Torch CPU, MPS on Apple Silicon and NVIDIA CUDA.
L3: # For NVIDIA hosts use the matching CUDA wheel; see requirements-cuda.txt.
L4: torch==2.14.0
```

### `requirements-cuda.txt` — 원본 L1–L7
SHA256: `b724d8f5d0885f6c25645386b26b3c40230aef7782d0714210ec172ae778b504`
```text
L1: -r requirements-c.txt
L2: --extra-index-url https://download.pytorch.org/whl/cu130
L3: # Windows/Linux CUDA runtime isolated in this virtual environment.
L4: cupy-cuda13x[ctk]==14.2.0
L5: cuda-toolkit==13.0.2
L6: # Pin the CUDA build so a fresh Windows install cannot select a CPU wheel.
L7: torch==2.14.0+cu130
```

### `requirements-warp.txt` — 원본 L1–L4
SHA256: `7f5f6f9a8249a683b31bf47bfdc3ba55dd4db7566ddebae49df3911f1ec97349`
```text
L1: # Optional NVIDIA physics candidate. Neural CUDA remains requirements-cuda.txt.
L2: # Matches the pinned FlyGym 2.1.0 warp extra; does not replace CPU MuJoCo.
L3: warp-lang==1.14.0
L4: mujoco-warp==3.9.0
```

## 첨부 숫자에서 계산한 파생값

교사 cadence 2.382760007990865 Hz의 주기는 0.41968137649044923초입니다. 20ms 학습 창은 그 주기의 약 4.7655%입니다.
E359 producer startup qmax 0.0396360246919627 rad의 0.04 rad 제한까지 여유는 0.00036397530803729955 rad입니다.
E358 producer startup qmax 0.04220501637004681 rad는 0.04 rad보다 0.0022050163700468098 rad 큽니다.
두 startup 값은 독립 body audit가 완료되지 않은 기술적 무효 실행의 producer 기록입니다. 확정적 생물학적·신체 보행 비교로 사용하지 않습니다.

## 추가로 필요한 핵심 파일

다음 파일들은 최근 연구 factory/assemble에서 참조되나 이 ZIP에서 확인할 수 없습니다. 전체 dependency closure 목록을 대체하지 않습니다.
- `tools/banc_m3_checkpointed_rollout.py` — 첨부에 없음
- `tools/banc_m3_grouped_current_adjoint.py` — 첨부에 없음
- `tools/banc_m3_monotone_adjoint.py` — 첨부에 없음
- `tools/banc_m3_smooth_temporal.py` — 첨부에 없음
- `tools/banc_m3_ssp_adjoint.py` — 첨부에 없음
- `tools/banc_m3_ssp_efficacy_trial.py` — 첨부에 없음
- `tools/banc_m3_ssp_recovery_trial.py` — 첨부에 없음
- `tools/verify_c_banc_m3_e113_body.py` — 첨부에 없음
- `tools/banc_m3_phase_free_shape.py` — 첨부에 없음