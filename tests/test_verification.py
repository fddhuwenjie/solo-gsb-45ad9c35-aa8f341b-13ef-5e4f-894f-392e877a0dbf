"""Tests for off-site export and independent replay verification.

The verifier only ever sees the exported JSON package (never the source
SQLite); tests corrupt packages in every dimension and assert the first
failing version and machine-readable reason.
"""
import threading

import pytest
from fastapi.testclient import TestClient

from app import execution
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "rotation.db"
    monkeypatch.setenv("ROTATION_DB_PATH", str(db))
    execution.reset_store()
    yield
    execution.reset_store()


SCENARIO = {
    "certificates": [
        {"cert_id": "root-old", "subject": "CN=Old Root CA", "issuer": "CN=Old Root CA",
         "not_valid_before": "2020-01-01T00:00:00Z",
         "not_valid_after": "2030-01-01T00:00:00Z"},
        {"cert_id": "root-new", "subject": "CN=New Root CA", "issuer": "CN=New Root CA",
         "not_valid_before": "2024-01-01T00:00:00Z",
         "not_valid_after": "2035-01-01T00:00:00Z"},
        {"cert_id": "leaf-old", "subject": "CN=svc", "issuer": "CN=Old Root CA",
         "not_valid_before": "2025-01-01T00:00:00Z",
         "not_valid_after": "2026-12-31T00:00:00Z"},
        {"cert_id": "leaf-new", "subject": "CN=svc", "issuer": "CN=New Root CA",
         "not_valid_before": "2026-01-01T00:00:00Z",
         "not_valid_after": "2027-12-31T00:00:00Z"},
    ],
    "trust_anchors": [{"client_id": "c", "cert_id": "root-old"}],
    "endpoints": [
        {"endpoint_id": "svc",
         "current_chain": ["leaf-old", "root-old"],
         "target_chain": ["leaf-new", "root-new"]}
    ],
    "relationships": [{"client_id": "c", "endpoint_id": "svc"}],
    "actions": [
        {"action_id": "a1-add", "type": "add_trust", "client_id": "c",
         "anchor_cert_id": "root-new", "allowed_stages": [1, 2, 3]},
        {"action_id": "a2-replace", "type": "replace_cert", "endpoint_id": "svc",
         "new_chain": ["leaf-new", "root-new"], "allowed_stages": [2, 3]},
        {"action_id": "a3-remove", "type": "remove_trust", "client_id": "c",
         "anchor_cert_id": "root-old", "allowed_stages": [1, 2, 3]},
    ],
    "plan_time": "2026-06-01T00:00:00Z",
}

PLAN = [
    {"stage": 1, "action_ids": ["a1-add"]},
    {"stage": 2, "action_ids": ["a2-replace", "a3-remove"]},
]

# Plan with an explicit leading no-op (empty) stage.
PLAN_WITH_EMPTY_STAGE = [
    {"stage": 1, "action_ids": []},
    {"stage": 2, "action_ids": ["a1-add", "a2-replace", "a3-remove"]},
]


def _register(plan=PLAN, scenario=SCENARIO):
    r = client.post("/plan-executions", json={**scenario, "plan": plan})
    assert r.status_code == 201, r.text
    return r.json()["execution_id"]


def _export(eid):
    r = client.get(f"/plan-executions/{eid}/export")
    assert r.status_code == 200, r.text
    return r.json()


def _verify(pkg, expected_status=200):
    r = client.post("/execution-verifications", json=pkg)
    assert r.status_code == expected_status, r.text
    return r.json()


def _commit(eid, payload):
    r = client.post(f"/plan-executions/{eid}/stages", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


STAGE1 = {
    "expected_version": 0, "stage": 1, "action_ids": ["a1-add"],
    "anchor_observations": [
        {"client_id": "c", "anchor_cert_id": "root-old", "present": True},
        {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
    ],
    "chain_observations": [
        {"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}],
}

STAGE2 = {
    "expected_version": 1, "stage": 2,
    "action_ids": ["a3-remove", "a2-replace"],
    "anchor_observations": [
        {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
        {"client_id": "c", "anchor_cert_id": "root-old", "present": False},
    ],
    "chain_observations": [
        {"endpoint_id": "svc", "chain": ["leaf-new", "root-new"]}],
}


# ---------------------------------------------------------------------------
# Happy path: the exported package verifies independently at every version
# ---------------------------------------------------------------------------


def test_export_shape_is_self_contained_and_single_version():
    eid = _register()
    pkg = _export(eid)
    assert pkg["package_kind"] == "rotation-plan-execution-export"
    assert pkg["format_version"] == 1
    assert pkg["execution_id"] == eid
    assert pkg["version"] == 0
    assert pkg["stage_count"] == 2
    assert pkg["completed_stages"] == []
    assert pkg["next_stage"] == 1
    assert pkg["finished"] is False
    # verbatim scenario and plan are embedded
    assert pkg["scenario"]["plan_time"] == SCENARIO["plan_time"]
    assert pkg["plan"] == PLAN
    # exactly one committed version (version 0), with its snapshot
    assert [c["version"] for c in pkg["commits"]] == [0]
    entry = pkg["commits"][0]
    assert entry["stage"] == 0
    assert entry["action_ids"] == []
    assert entry["snapshot"]["stage"] == 0


def test_verify_version_zero_prefix_states_next_stage():
    eid = _register()
    body = _verify(_export(eid))
    assert body["status"] == "ok"
    assert body["verified_version"] == 0
    assert body["version"] == 0
    assert body["stage_count"] == 2
    assert body["next_stage"] == 1
    assert body["finished"] is False
    assert body["mismatch"] is None
    assert "package-internal consistency only" in body["scope"]


def test_verify_after_each_committed_stage():
    eid = _register()
    _commit(eid, STAGE1)
    body = _verify(_export(eid))
    assert body["status"] == "ok"
    assert body["verified_version"] == 1
    assert body["next_stage"] == 2
    assert body["finished"] is False

    _commit(eid, STAGE2)
    body = _verify(_export(eid))
    assert body["status"] == "ok"
    assert body["verified_version"] == 2
    assert body["next_stage"] is None
    assert body["finished"] is True
    assert [c["version"] for c in body["mismatch"] or []] == []  # mismatch None


def test_empty_noop_stage_is_verified_like_any_stage():
    eid = _register(plan=PLAN_WITH_EMPTY_STAGE)
    # commit the empty stage 1
    _commit(eid, {
        "expected_version": 0, "stage": 1, "action_ids": [],
        "anchor_observations": [
            {"client_id": "c", "anchor_cert_id": "root-old", "present": True},
            {"client_id": "c", "anchor_cert_id": "root-new", "present": False},
        ],
        "chain_observations": [
            {"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}],
    })
    pkg = _export(eid)
    assert len(pkg["commits"]) == 2
    assert pkg["commits"][1]["action_ids"] == []
    body = _verify(pkg)
    assert body["status"] == "ok"
    assert body["verified_version"] == 1
    assert body["next_stage"] == 2
    # the no-op snapshot carries the unchanged initial state
    snap = pkg["commits"][1]["snapshot"]
    assert snap["trust_anchors"] == [
        {"client_id": "c", "anchor_cert_id": "root-old"}]
    assert snap["endpoint_chains"] == [
        {"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}]
    assert snap["relationships"][0]["valid"] is True


# ---------------------------------------------------------------------------
# Commit sequence: missing / duplicate / out-of-order / extra
# ---------------------------------------------------------------------------


def test_missing_initial_entry_is_rejected():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"] = [c for c in pkg["commits"] if c["version"] != 0]
    body = _verify(pkg)
    assert body["status"] == "conflict"
    mm = body["mismatch"]
    assert mm["reason"] == "missing_initial_entry"
    assert mm["stage"] == 1


def test_missing_intermediate_entry_is_a_gap_and_reports_first_bad_version():
    eid = _register()
    _commit(eid, STAGE1)
    _commit(eid, STAGE2)
    pkg = _export(eid)
    pkg["commits"] = [c for c in pkg["commits"] if c["version"] != 1]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "commit_gap"
    assert mm["version"] == 2
    assert mm["expected"] == 1
    assert mm["found"] == 2
    # versions strictly before the gap were confirmed
    assert body["verified_version"] == 1
    assert body["status"] == "conflict"


def test_duplicate_version_entry_is_rejected():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"].append(pkg["commits"][1])
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "commit_duplicate_version"
    assert mm["version"] == 1


def test_out_of_order_entries_are_rejected():
    eid = _register()
    _commit(eid, STAGE1)
    _commit(eid, STAGE2)
    pkg = _export(eid)
    pkg["commits"][1], pkg["commits"][2] = pkg["commits"][2], pkg["commits"][1]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] in {"stage_out_of_order", "commit_gap"}
    assert mm["version"] in {1, 2}


def test_entry_newer_than_header_version_is_commit_extra():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["version"] = 0  # header claims version 0 but a v1 entry is present
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "commit_extra"
    assert mm["version"] == 1


def test_header_version_not_last_commit_is_header_mismatch():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    # drop the last entry but keep header at version 1
    pkg["commits"] = pkg["commits"][:1]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "header_mismatch"
    assert mm["expected"] == 1
    assert mm["found"] == 0


# ---------------------------------------------------------------------------
# Per-version content checks
# ---------------------------------------------------------------------------


def test_version_stage_mismatch_detected():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"][1]["stage"] = 2
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "version_stage_mismatch"
    assert mm["version"] == 1
    assert mm["expected"] == 1
    assert mm["found"] == 2


def test_initial_entry_with_actions_is_rejected():
    eid = _register()
    pkg = _export(eid)
    pkg["commits"][0]["action_ids"] = ["a1-add"]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "initial_entry_not_empty"
    assert mm["version"] == 0


def test_action_set_mismatch_reports_stage_and_expected_set():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"][1]["action_ids"] = ["a3-remove"]
    # keep snapshot internally aligned so the action check fires first
    pkg["commits"][1]["snapshot"]["action_ids"] = ["a3-remove"]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "action_set_mismatch"
    assert mm["stage"] == 1
    assert mm["expected"] == ["a1-add"]
    assert mm["found"] == ["a3-remove"]
    assert body["verified_version"] == 1  # v0 verified, v1 failed


def test_observation_mismatch_anchor_present_flipped():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    # claim the old anchor is gone after stage 1 (it is still present)
    for obs in pkg["commits"][1]["anchor_observations"]:
        if obs["anchor_cert_id"] == "root-old":
            obs["present"] = False
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "observation_mismatch"
    assert mm["version"] == 1
    assert mm["detail"]["kind"] == "anchor"
    assert mm["detail"]["submitted_present"] is False
    assert mm["detail"]["expected_present"] is True


def test_observation_mismatch_chain_wrong():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"][1]["chain_observations"][0]["chain"] = ["leaf-new", "root-new"]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "observation_mismatch"
    assert mm["detail"]["kind"] == "endpoint_chain"
    assert mm["detail"]["submitted_chain"] == ["leaf-new", "root-new"]
    assert mm["detail"]["expected_chain"] == ["leaf-old", "root-old"]


def test_unknown_observation_reference_detected():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"][1]["anchor_observations"].append(
        {"client_id": "c", "anchor_cert_id": "nope", "present": True})
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "unknown_reference"
    assert mm["version"] == 1


def test_duplicate_observation_detected():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    obs = pkg["commits"][1]["anchor_observations"][0]
    pkg["commits"][1]["anchor_observations"].append(dict(obs))
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "duplicate_observation"
    assert mm["version"] == 1


def test_unsafe_swapped_plan_replays_to_a_broken_relationship():
    # A structurally valid but unsafe plan: remove the old anchor at stage 1
    # (before the new chain is served) breaks the relationship. Rewrite both
    # the plan and that version's action set so replay reaches check_batch.
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["plan"] = [
        {"stage": 1, "action_ids": ["a3-remove"]},
        {"stage": 2, "action_ids": ["a1-add", "a2-replace"]},
    ]
    pkg["commits"][1]["action_ids"] = ["a3-remove"]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "untrusted_root"
    assert mm["stage"] == 1
    assert mm["blocking_action"] == "a3-remove"
    assert mm["relationship"] == {"client_id": "c", "endpoint_id": "svc"}
    # replay stops at stage 1; nothing later is trusted
    assert body["verified_version"] == 1


def test_structurally_invalid_plan_in_package_rejected():
    # A plan whose stage 1 schedules an unknown action: structurally invalid.
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["plan"][0]["action_ids"] = ["a1-add", "ghost-action"]
    pkg["commits"][1]["action_ids"] = ["a1-add", "ghost-action"]
    body = _verify(pkg)
    assert body["mismatch"]["reason"] == "invalid_plan"


def test_plan_missing_an_action_is_invalid_plan():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    # stage 1 schedules a3-remove instead of a1-add; a1-add never scheduled
    pkg["plan"] = [
        {"stage": 1, "action_ids": ["a3-remove"]},
        {"stage": 2, "action_ids": ["a2-replace"]},
    ]
    pkg["commits"][1]["action_ids"] = ["a3-remove"]
    body = _verify(pkg)
    assert body["mismatch"]["reason"] == "invalid_plan"


def test_invalid_scenario_in_package_rejected():
    eid = _register()
    pkg = _export(eid)
    # self-signed issuer mismatch makes the scenario invalid
    pkg["scenario"]["certificates"][0]["issuer"] = "CN=Someone Else"
    body = _verify(pkg)
    assert body["mismatch"]["reason"] == "invalid_scenario"


# ---------------------------------------------------------------------------
# Snapshot tampering: every component is independently rebuilt and compared
# ---------------------------------------------------------------------------


def test_snapshot_trust_anchor_tampering_detected():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"][1]["snapshot"]["trust_anchors"] = [
        {"client_id": "c", "anchor_cert_id": "root-new"}]  # old anchor dropped
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "snapshot_mismatch"
    assert mm["component"] == "trust_anchors"
    assert mm["version"] == 1


def test_snapshot_endpoint_chain_tampering_detected():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"][1]["snapshot"]["endpoint_chains"][0]["chain"] = [
        "leaf-new", "root-new"]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "snapshot_mismatch"
    assert mm["component"] == "endpoint_chains"


def test_snapshot_relationship_validity_tampering_detected():
    eid = _register()
    pkg = _export(eid)
    pkg["commits"][0]["snapshot"]["relationships"][0]["valid"] = False
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "snapshot_mismatch"
    assert mm["component"] == "relationships"
    assert mm["version"] == 0


def test_snapshot_relationship_anchor_list_tampering_detected():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    rel = pkg["commits"][1]["snapshot"]["relationships"][0]
    rel["trusted_anchors"] = ["root-new", "root-old", "evil-root"]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "snapshot_mismatch"
    assert mm["component"] == "relationships"


def test_snapshot_actions_tampering_detected_even_when_commit_ok():
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    pkg["commits"][1]["snapshot"]["action_ids"] = ["a3-remove"]
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "snapshot_mismatch"
    assert mm["component"] == "action_ids"


def test_initial_snapshot_anchor_tampering_detected():
    eid = _register()
    pkg = _export(eid)
    pkg["commits"][0]["snapshot"]["trust_anchors"] = []
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "snapshot_mismatch"
    assert mm["component"] == "trust_anchors"
    assert body["verified_version"] == 0


# ---------------------------------------------------------------------------
# Header bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("completed_stages", [1]),
    ("next_stage", 9),
    ("finished", True),
])
def test_header_progress_fields_must_match_prefix(field, value):
    eid = _register()  # version 0, nothing committed
    pkg = _export(eid)
    pkg[field] = value
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "header_mismatch"
    assert mm["version"] == pkg["version"]


def test_stage_count_must_match_plan_length():
    eid = _register()
    pkg = _export(eid)
    pkg["stage_count"] = 9
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "header_mismatch"
    assert mm["expected"] == 2
    assert mm["found"] == 9


# ---------------------------------------------------------------------------
# First-error semantics + envelope validation
# ---------------------------------------------------------------------------


def test_only_first_bad_version_is_reported():
    eid = _register()
    _commit(eid, STAGE1)
    _commit(eid, STAGE2)
    pkg = _export(eid)
    # corrupt BOTH stage 1 (action set) and stage 2 (snapshot)
    pkg["commits"][1]["action_ids"] = ["a3-remove"]
    pkg["commits"][2]["snapshot"]["trust_anchors"] = []
    body = _verify(pkg)
    mm = body["mismatch"]
    assert mm["reason"] == "action_set_mismatch"
    assert mm["stage"] == 1
    assert body["verified_version"] == 1


def test_malformed_envelope_is_422_not_a_conflict():
    r = client.post("/execution-verifications", json={"nope": True})
    assert r.status_code == 422


def test_verifier_never_touches_source_store(tmp_path, monkeypatch):
    # Pointing the verifier-side process at an empty, nonexistent DB while
    # verifying a package still works: the package alone is sufficient.
    eid = _register()
    _commit(eid, STAGE1)
    pkg = _export(eid)
    monkeypatch.setenv("ROTATION_DB_PATH", str(tmp_path / "other.db"))
    execution.reset_store()
    body = _verify(pkg)
    assert body["status"] == "ok"
    assert body["verified_version"] == 1
    assert not (tmp_path / "other.db").exists()


# ---------------------------------------------------------------------------
# Export read consistency under concurrent commits
# ---------------------------------------------------------------------------


def test_exports_stay_single_version_under_concurrent_commits():
    eid = _register()
    failures = []

    def commit_both():
        try:
            _commit(eid, STAGE1)
            _commit(eid, STAGE2)
        except AssertionError as exc:  # pragma: no cover - diagnostic only
            failures.append(exc)

    seen_versions: set[int] = set()
    t = threading.Thread(target=commit_both)
    t.start()
    while t.is_alive():
        pkg = _export(eid)
        seen_versions.add(pkg["version"])
        # the package must always be one coherent version ...
        assert [c["version"] for c in pkg["commits"]] == list(
            range(pkg["version"] + 1))
        # ... and must verify independently as that version
        body = _verify(pkg)
        assert body["status"] == "ok", body["mismatch"]
        assert body["verified_version"] == pkg["version"]
    t.join()
    assert not failures
    assert seen_versions  # observed at least one version during the run
