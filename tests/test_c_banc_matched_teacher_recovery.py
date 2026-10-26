"""Pure gate checks for reusing completed parity without relabeling a partial teacher."""
from copy import deepcopy

import pytest

from tools.observe_c_banc_m3_matched_teacher_recovery import verify_reused_parity


def packet():
    proof = {"parity_validity": "independently_valid_exact", "complete_teacher_collection": False,
             "arms": {"parity": {"valid_saved_prefix": True, "completed_controls": 99,
                 "neural_ticks": 1980, "physics_steps": 4950, "stage_rows": 7920,
                 "minimum_upright": .9409362153417923, "complete_requested_arm": True}}}
    result = {"parity": "valid_exact", "teacher": "resource_limited", "validity": "resource_limited",
              "whole_goal_complete": False}
    ledger = {"storage_errors": [], "arms": [
        {"arm": "parity", "valid": True, "controls_completed": 99, "storage_errors": []},
        {"arm": "teacher", "controls_completed": 1936, "status": "resource_limited", "storage_errors": []}]}
    supervisor = {"exit_code": 1, "live": [], "terminated": [], "cumulative_after_s": 6278.4757079996525}
    return proof, result, ledger, supervisor


def test_reuses_valid_parity_without_promoting_teacher_prefix():
    values = packet()
    before = deepcopy(values)
    verify_reused_parity(*values)
    assert values == before


@pytest.mark.parametrize("kind", ["partial_parity", "teacher_promoted", "live_process", "storage_error"])
def test_rejects_incomplete_or_misrepresented_prior_evidence(kind):
    values = packet()
    if kind == "partial_parity":
        values[0]["arms"]["parity"]["completed_controls"] = 98
    elif kind == "teacher_promoted":
        values[0]["complete_teacher_collection"] = True
    elif kind == "live_process":
        values[3]["live"] = [123]
    else:
        values[2]["arms"][0]["storage_errors"] = ["failed raw write"]
    with pytest.raises(ValueError):
        verify_reused_parity(*values)
