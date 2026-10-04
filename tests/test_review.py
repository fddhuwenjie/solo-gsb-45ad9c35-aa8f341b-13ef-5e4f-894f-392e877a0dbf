"""Tests for version-consistent execution export and off-site review."""
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import execution
from app.main import app

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Point the store at a throwaway SQLite file for every test."""
    db = tmp_path / "rotation.db"
    monkeypatch.setenv("ROTATION_DB_PATH", str(db))
    execution.reset_store()
    yield
    execution.reset_store()


def cert(cid, subject, issuer):
    return {
        "cert_id": cid,
        "subject": subject,
        "issuer": issuer,
        "not_valid_before": (NOW - timedelta(days=365)).isoformat(),
        "not_valid_after": (NOW + timedelta(days=365)).isoformat(),
    }


SCENARIO = {
    "certificates": [
        cert("root-old", "CN=Old Root CA", "CN=Old Root CA"),
        cert("root-new", "CN=New Root CA", "CN=New Root CA"),
        cert("leaf-old", "CN=svc", "CN=Old Root CA"),
        cert("leaf-new", "CN=svc", "CN=New Root CA"),
    ],
    "trust_anchors": [{"client_id": "c", "cert_id": "root-old"}],
    "endpoints": [
        {
            "endpoint_id": "svc",
            "current_chain": ["leaf-old", "root-old"],
            "target_chain": ["leaf-new", "root-new"],
        }
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
    "plan_time": NOW.isoformat(),
}

PLAN = [
    {"stage": 1, "action_ids": ["a1-add"]},
    {"stage": 2, "action_ids": ["a2-replace", "a3-remove"]},
]

STAGE1_COMMIT = {
    "expected_version": 0,
    "stage": 1,
    "action_ids": ["a1-add"],
    "anchor_observations": [
        {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
        {"client_id": "c", "anchor_cert_id": "root-old", "present": True},
    ],
    "chain_observations": [
        {"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}
    ],
}

STAGE2_COMMIT = {
    "expected_version": 1,
    "stage": 2,
    "action_ids": ["a2-replace", "a3-remove"],
    "anchor_observations": [
        {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
        {"client_id": "c", "anchor_cert_id": "root-old", "present": False},
    ],
    "chain_observations": [
        {"endpoint_id": "svc", "chain": ["leaf-new", "root-new"]}
    ],
}


def register(plan=PLAN, scenario=SCENARIO):
    r = client.post("/plan-executions", json={**scenario, "plan": plan})
    assert r.status_code == 201, r.text
    return r.json()["execution_id"]


def commit(exec_id, payload):
    r = client.post(f"/plan-executions/{exec_id}/stages", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def export(exec_id):
    r = client.get(f"/plan-executions/{exec_id}/export")
    assert r.status_code == 200, r.text
    return r.json()


def review(pkg):
    return client.post("/execution-reviews", json=pkg)


def finished_package():
    exec_id = register()
    commit(exec_id, STAGE1_COMMIT)
    commit(exec_id, STAGE2_COMMIT)
    return export(exec_id)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_fresh_registration_is_version_zero():
    exec_id = register()
    pkg = export(exec_id)
    assert pkg["format"] == "rotation-execution-export/v1"
    assert pkg["execution_id"] == exec_id
    assert pkg["version"] == 0
    assert pkg["stage_count"] == 2
    assert len(pkg["snapshots"]) == 1
    assert pkg["snapshots"][0]["stage"] == 0
    # the scenario and plan round-trip (datetimes may be re-serialized in a
    # different but equivalent ISO-8601 form)
    assert [c["cert_id"] for c in pkg["scenario"]["certificates"]] == [
        c["cert_id"] for c in SCENARIO["certificates"]
    ]
    assert pkg["scenario"]["plan_time"].replace("+00:00", "Z") == \
        SCENARIO["plan_time"].replace("+00:00", "Z")
    assert pkg["plan"] == PLAN


def test_export_unknown_execution_is_404():
    r = client.get("/plan-executions/does-not-exist/export")
    assert r.status_code == 404


def test_export_during_commits_always_yields_one_complete_version():
    """Concurrent commits must never tear an export across a version boundary."""
    exec_id = register()
    stop = threading.Event()
    problems = []

    def reader():
        while not stop.is_set():
            pkg = export(exec_id)
            stages = [s["stage"] for s in pkg["snapshots"]]
            if stages != list(range(pkg["version"] + 1)):
                problems.append((pkg["version"], stages))
            r = review(pkg)
            if r.status_code != 200:
                problems.append(("review", r.status_code, r.text))

    t = threading.Thread(target=reader)
    t.start()
    for payload in (STAGE1_COMMIT, STAGE2_COMMIT):
        # a few export rounds between the commits
        time.sleep(0.02)
        commit(exec_id, payload)
    time.sleep(0.02)
    stop.set()
    t.join()
    assert problems == []


# ---------------------------------------------------------------------------
# Review: consistent packages
# ---------------------------------------------------------------------------


def test_review_version_zero_package_is_consistent_and_names_next_stage():
    pkg = export(register())
    r = review(pkg)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "consistent"
    assert body["verified_version"] == 0
    assert body["stage_count"] == 2
    assert body["next_stage"] == 1
    assert body["finished"] is False
    assert body["scope"] == "package_internal_consistency"
    assert "not the authenticity of its source" in body["message"]


def test_review_mid_plan_package_reports_verified_version_and_next_stage():
    exec_id = register()
    commit(exec_id, STAGE1_COMMIT)
    pkg = export(exec_id)
    r = review(pkg)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verified_version"] == 1
    assert body["next_stage"] == 2
    assert body["finished"] is False


def test_review_finished_package():
    pkg = finished_package()
    r = review(pkg)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verified_version"] == 2
    assert body["next_stage"] is None
    assert body["finished"] is True


def test_review_needs_no_database(tmp_path, monkeypatch):
    """The verifying instance must not touch the source (or any) SQLite store."""
    pkg = finished_package()
    foreign_db = tmp_path / "foreign" / "rotation.db"
    monkeypatch.setenv("ROTATION_DB_PATH", str(foreign_db))
    execution.reset_store()
    r = review(pkg)
    assert r.status_code == 200, r.text
    assert not foreign_db.exists()


def test_review_zero_action_plan_is_finished_at_version_zero():
    scenario = {
        "certificates": [cert("r0", "CN=r0", "CN=r0")],
        "trust_anchors": [{"client_id": "c", "cert_id": "r0"}],
        "endpoints": [{"endpoint_id": "svc", "current_chain": ["r0"],
                       "target_chain": ["r0"]}],
        "relationships": [{"client_id": "c", "endpoint_id": "svc"}],
        "actions": [],
        "plan_time": NOW.isoformat(),
    }
    pkg = export(register(plan=[], scenario=scenario))
    r = review(pkg)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verified_version"] == 0
    assert body["finished"] is True
    assert body["next_stage"] is None


def test_review_noop_stage_package_is_consistent():
    plan = [
        {"stage": 1, "action_ids": []},
        {"stage": 2, "action_ids": ["a1-add", "a2-replace", "a3-remove"]},
    ]
    scenario = {**SCENARIO, "actions": [
        dict(SCENARIO["actions"][0], allowed_stages=[2]),
        dict(SCENARIO["actions"][1], allowed_stages=[2]),
        dict(SCENARIO["actions"][2], allowed_stages=[2]),
    ]}
    exec_id = register(plan=plan, scenario=scenario)
    commit(exec_id, {"expected_version": 0, "stage": 1, "action_ids": []})
    pkg = export(exec_id)
    r = review(pkg)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verified_version"] == 1
    assert body["next_stage"] == 2


# ---------------------------------------------------------------------------
# Review: sequence problems (missing / duplicate / out-of-order / unexpected)
# ---------------------------------------------------------------------------


def test_missing_middle_stage_is_reported():
    pkg = finished_package()
    del pkg["snapshots"][1]  # drop the stage-1 snapshot, keep version=2
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "inconsistent"
    assert body["reason"] == "missing_stage"
    assert body["stage"] == 1
    assert body["verified_version"] is None
    assert body["scope"] == "package_internal_consistency"


def test_missing_initial_snapshot_is_stage_zero():
    pkg = finished_package()
    del pkg["snapshots"][0]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "missing_stage"
    assert body["stage"] == 0


def test_duplicate_stage_is_reported():
    pkg = finished_package()
    pkg["snapshots"].insert(2, pkg["snapshots"][1])  # stage 1 twice
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "duplicate_stage"
    assert body["stage"] == 1


def test_out_of_order_stages_are_reported():
    pkg = finished_package()
    pkg["snapshots"][1], pkg["snapshots"][2] = (
        pkg["snapshots"][2],
        pkg["snapshots"][1],
    )
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "stage_out_of_order"
    assert body["stage"] == 1


def test_snapshot_beyond_declared_version_is_unexpected():
    pkg = finished_package()
    pkg["version"] = 1  # snapshots still cover stages 0..2
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "unexpected_stage"
    assert body["stage"] == 2


def test_stage_count_mismatch_is_reported():
    pkg = finished_package()
    pkg["stage_count"] = 5
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "stage_count_mismatch"
    assert body["stage"] == 0


# ---------------------------------------------------------------------------
# Review: content mismatches
# ---------------------------------------------------------------------------


def test_wrong_action_set_is_reported_with_expected_and_actual():
    pkg = finished_package()
    pkg["snapshots"][1]["action_ids"] = ["a1-add", "a3-remove"]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "action_set_mismatch"
    assert body["stage"] == 1
    assert body["verified_version"] == 0
    assert body["expected"] == ["a1-add"]
    assert body["actual"] == ["a1-add", "a3-remove"]


def test_reordered_action_ids_are_a_snapshot_mismatch():
    pkg = finished_package()
    # same set as the plan stage, but not in the canonical plan order
    pkg["snapshots"][2]["action_ids"] = ["a3-remove", "a2-replace"]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "snapshot_mismatch"
    assert body["snapshot_field"] == "action_ids"
    assert body["stage"] == 2
    assert body["verified_version"] == 1


def test_tampered_trust_anchors_are_a_snapshot_mismatch():
    pkg = finished_package()
    pkg["snapshots"][1]["trust_anchors"] = [
        {"client_id": "c", "anchor_cert_id": "root-old"}
    ]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "snapshot_mismatch"
    assert body["snapshot_field"] == "trust_anchors"
    assert body["stage"] == 1
    assert body["actual"] == [{"client_id": "c", "anchor_cert_id": "root-old"}]
    assert {tuple(sorted(a.items())) for a in body["expected"]} == {
        tuple(sorted(a.items()))
        for a in [
            {"client_id": "c", "anchor_cert_id": "root-new"},
            {"client_id": "c", "anchor_cert_id": "root-old"},
        ]
    }


def test_tampered_endpoint_chain_is_a_snapshot_mismatch():
    pkg = finished_package()
    pkg["snapshots"][2]["endpoint_chains"] = [
        {"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}
    ]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "snapshot_mismatch"
    assert body["snapshot_field"] == "endpoint_chains"
    assert body["stage"] == 2


def test_tampered_relationship_is_a_snapshot_mismatch():
    pkg = finished_package()
    pkg["snapshots"][2]["relationships"][0]["trusted_anchors"] = ["root-old"]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "snapshot_mismatch"
    assert body["snapshot_field"] == "relationships"
    assert body["stage"] == 2


def test_tampered_snapshot_version_is_a_snapshot_mismatch():
    pkg = finished_package()
    pkg["snapshots"][1]["version"] = 7
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "snapshot_mismatch"
    assert body["snapshot_field"] == "version"
    assert body["stage"] == 1


def test_tampered_anchor_observation_is_an_observation_mismatch():
    pkg = finished_package()
    obs = pkg["snapshots"][1]["anchor_observations"]
    root_new = next(o for o in obs if o["anchor_cert_id"] == "root-new")
    root_new["present"] = False
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "observation_mismatch"
    assert body["stage"] == 1
    assert body["observation"]["kind"] == "anchor"
    assert body["observation"]["submitted_present"] is False
    assert body["observation"]["expected_present"] is True


def test_tampered_chain_observation_is_an_observation_mismatch():
    pkg = finished_package()
    pkg["snapshots"][2]["chain_observations"] = [
        {"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}
    ]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "observation_mismatch"
    assert body["stage"] == 2
    assert body["observation"]["kind"] == "endpoint_chain"
    assert body["observation"]["expected_chain"] == ["leaf-new", "root-new"]


def test_noop_stage_with_unexpected_actions_is_an_action_set_mismatch():
    plan = [
        {"stage": 1, "action_ids": []},
        {"stage": 2, "action_ids": ["a1-add", "a2-replace", "a3-remove"]},
    ]
    scenario = {**SCENARIO, "actions": [
        dict(SCENARIO["actions"][0], allowed_stages=[2]),
        dict(SCENARIO["actions"][1], allowed_stages=[2]),
        dict(SCENARIO["actions"][2], allowed_stages=[2]),
    ]}
    exec_id = register(plan=plan, scenario=scenario)
    commit(exec_id, {"expected_version": 0, "stage": 1, "action_ids": []})
    pkg = export(exec_id)
    # claim the no-op stage executed an action
    pkg["snapshots"][1]["action_ids"] = ["a1-add"]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "action_set_mismatch"
    assert body["stage"] == 1
    assert body["expected"] == []


def test_noop_stage_with_tampered_state_is_a_snapshot_mismatch():
    plan = [
        {"stage": 1, "action_ids": []},
        {"stage": 2, "action_ids": ["a1-add", "a2-replace", "a3-remove"]},
    ]
    scenario = {**SCENARIO, "actions": [
        dict(SCENARIO["actions"][0], allowed_stages=[2]),
        dict(SCENARIO["actions"][1], allowed_stages=[2]),
        dict(SCENARIO["actions"][2], allowed_stages=[2]),
    ]}
    exec_id = register(plan=plan, scenario=scenario)
    commit(exec_id, {"expected_version": 0, "stage": 1, "action_ids": []})
    pkg = export(exec_id)
    # a no-op stage must carry the trust state over unchanged
    pkg["snapshots"][1]["trust_anchors"] = []
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "snapshot_mismatch"
    assert body["snapshot_field"] == "trust_anchors"
    assert body["stage"] == 1


# ---------------------------------------------------------------------------
# Review: trust replay failures
# ---------------------------------------------------------------------------


def test_plan_whose_committed_stage_breaks_a_relationship_is_reported():
    pkg = finished_package()
    # Craft a package whose (structurally valid) plan removes the old root at
    # stage 1; the committed prefix then breaks the only relationship.
    pkg["plan"] = [
        {"stage": 1, "action_ids": ["a3-remove"]},
        {"stage": 2, "action_ids": ["a1-add", "a2-replace"]},
    ]
    pkg["snapshots"][1]["action_ids"] = ["a3-remove"]
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "untrusted_root"
    assert body["stage"] == 1
    assert body["verified_version"] == 0
    assert body["relationship"] == {"client_id": "c", "endpoint_id": "svc"}
    assert body["blocking_action"] == "a3-remove"
    assert body["trusted_anchors"] == []


def test_broken_initial_state_is_reported_at_stage_zero():
    scenario = {**SCENARIO, "trust_anchors": []}
    # registration would reject this plan, but a fabricated package can claim
    # it; the review must catch the broken initial state on its own
    pkg = export(register())
    pkg["scenario"] = {**pkg["scenario"], "trust_anchors": []}
    pkg["snapshots"] = pkg["snapshots"][:1]
    pkg["version"] = 0
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "untrusted_root"
    assert body["stage"] == 0
    assert body["blocking_action"] is None


# ---------------------------------------------------------------------------
# Review: malformed packages (422)
# ---------------------------------------------------------------------------


def test_structurally_bad_plan_is_422():
    pkg = finished_package()
    pkg["plan"] = [{"stage": 1, "action_ids": ["a1-add"]}]  # actions missing
    r = review(pkg)
    assert r.status_code == 422
    assert "exactly once" in r.json()["detail"]


def test_unknown_action_in_plan_is_422():
    pkg = finished_package()
    pkg["plan"][0]["action_ids"] = ["ghost"]
    r = review(pkg)
    assert r.status_code == 422


def test_unknown_observation_references_are_422():
    pkg = finished_package()
    pkg["snapshots"][1]["anchor_observations"] = [
        {"client_id": "nobody", "anchor_cert_id": "root-new", "present": True}
    ]
    r = review(pkg)
    assert r.status_code == 422

    pkg = finished_package()
    pkg["snapshots"][1]["chain_observations"] = [
        {"endpoint_id": "ghost", "chain": ["leaf-old", "root-old"]}
    ]
    r = review(pkg)
    assert r.status_code == 422


def test_wrong_format_tag_is_422():
    pkg = finished_package()
    pkg["format"] = "something-else"
    r = review(pkg)
    assert r.status_code == 422


def test_scenario_violating_scene_rules_is_422():
    pkg = finished_package()
    pkg["scenario"]["relationships"] = [
        {"client_id": "c", "endpoint_id": "ghost"}
    ]
    r = review(pkg)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Review does not continue on top of a bad snapshot
# ---------------------------------------------------------------------------


def test_first_failing_stage_wins_and_later_stages_are_not_reported():
    pkg = finished_package()
    # corrupt stage 1 *and* stage 2 differently; only stage 1 may be reported
    pkg["snapshots"][1]["trust_anchors"] = []
    pkg["snapshots"][2]["endpoint_chains"] = []
    r = review(pkg)
    assert r.status_code == 409
    body = r.json()
    assert body["stage"] == 1
    assert body["snapshot_field"] == "trust_anchors"
    assert body["verified_version"] == 0


def test_exported_package_matches_documented_example_shape():
    """The export must carry scenario, plan and per-stage content+snapshot."""
    pkg = finished_package()
    assert set(pkg) >= {
        "format", "execution_id", "version", "stage_count",
        "scenario", "plan", "snapshots",
    }
    snap = pkg["snapshots"][1]
    # committed content and the resulting state live in one snapshot object
    assert snap["action_ids"] == ["a1-add"]
    assert {tuple(sorted(o.items())) for o in snap["anchor_observations"]}
    assert snap["trust_anchors"]
    assert snap["endpoint_chains"]
    assert snap["relationships"]
