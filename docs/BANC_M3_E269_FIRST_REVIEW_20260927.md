# E269 FIRST root review

2026-09-27. Actual gpt-6-sol/xhigh FIRST source `0a5011dc4a674da401b82ba5fd003e868d4ef24945e757427889f257fac6ed25` and tests `17bc403aa7f82b02cb3bd31caa1c50af536275a3b52d41e494efd33ba28b6d2e` archived before correction. Generated CUDA supervisor completed in 5.019478700007312 s, peak tree RSS 424083456 bytes, original/BANC/body/solver calls zero. The source equations and tangent/decoder kernels agree with generated tests and a separately written root RK4/finite-difference oracle. No original E268 fitting has run.

First correction is required for these implementation contracts:

1. Native outgoing masks are binary (`tools/banc_m3_source_filtered_synapses.py` restore and `flylab/c/rate_observation.py`), while this implementation accepts fractional masks and tests `.4`. Require exact 0/1 before CUDA creation and use binary fixtures; do not silently extend the native model.
2. `solve` and `decode_rates` initialize completed_samples to 1 before successful allocation/kernel synchronization; first-chunk errors can report a row that is still NaN. Initialize synchronized completion counts to zero and advance them only once all corresponding outputs are written and synchronized. Include first-chunk and derivative-first-chunk generated failure tests.
3. CUDA command-Jacobian and cotangent helpers do not preserve primary exceptions with response receipts/partial output, and work is not separately metered beyond transfers. Add bounded product request/completion/wall accounting and attach the current input response/cost to failures, without masking the original error or automatic recovery. Avoid charging an old result's products to a newer solve: explicit engine/result ownership and operation receipt binding, or a clearly validated current-response-only contract, is needed.

Root requested same-Sol first correction; next is root SECOND review and only root may fix remaining issues thereafter. CUDA retesting will be reserved against the corrected source. No acceptance tolerances, F1–F8, original resource allocation, or biological claims change.
