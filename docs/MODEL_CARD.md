> v0.2.1 변경: 머리 프레임/벽 접촉을 수정하고, 감각 기반 회피 기억과 접촉 후진을 추가했습니다. 공학적 제어이며 생물학적 재현 주장으로 확대하지 않습니다. 현재 실행 근거는 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)를 확인하세요. 아래 제작 환경의 미실행 설명은 v0.2.0 이력입니다.

# Model card — B hybrid navigation + physical body integration

## Intended use
Education, software prototyping, intervention/monitoring tools, and a foundation for subsequent validated neuromechanical experiments. This release is an integration candidate with native physics execution **not verified in the build environment**.

## Not supported claims
No whole-brain emulation, biological validation, realistic flight aerodynamics, measured calcium/membrane voltages, learned world model, guaranteed realtime throughput, or in-vivo intervention prediction.

## Data and compute
98 anatomical IDs, 177 directed pairs from a biased hemibrain excerpt; 72 engineered state nodes. Incoming sqrt-count normalization and uniformly positive anatomical edge signs are design assumptions. Dynamics are bounded rate updates rather than validated spike/voltage models. Python and source-A JavaScript were compared on identical sensor sequences and timestep.

## Physical body path
FlyGym 2.1.0 NeuroMechFly + MuJoCo 3.9.x; engineered HybridTurningController; 42 active joint targets and six adhesion controls. Joint/contact readout code is implemented. Correct execution and stable walking on this integrated environment remain unverified until the native physics gate runs successfully.

## Sensory simplification
64 categorical brightness rays using the physical arena; 9 distance rays; analytic Gaussian odor field; idealized self-motion sensing. Not retina/light transport/olfactory kinetics. A finite landmark has parallax. Absolute 3D scene reconstruction is not claimed.

## Observation honesty
Neural values are dimensionless [0,1] model activations. Physical angles are rad, contact forces baseline body weights, actuator outputs raw engine units. Browser skin and shadows are schematic. Actual articulated endpoints are displayed only when provided by the backend. Native camera screenshots are optional and not controller input.

## Validation status
See TESTING.md. Unit and mock integration tests do not establish correctness of unexecuted MuJoCo code. Tests/fixture_body.py is marked test-only, returns physics=false, and has no production launcher switch.

## Reproducibility
Versioned graph/scene/model checkpoints and command timing. Same-environment deterministic continuation must be tested with native physics; cross-platform bit-identical reproduction is not promised.
