"""Independent off-site replay verification of an exported execution package.

A verifier instance receives an :class:`ExecutionExport` over HTTP and replays
the committed prefix **without any access to the originating SQLite file**: the
package's own scenario and plan are parsed and structurally validated, then
every committed version (starting with version 0) is replayed from scratch with
the same deterministic engine used at commit time.

For each version the verifier checks, in order:

* the commit sequence is complete, unique and consecutive (no missing,
  duplicate, out-of-order or trailing-extra versions);
* the stage header is consistent with the version and with the plan;
* the committed action set equals exactly what the immutable plan schedules
  there (empty for a no-op stage — empty stages are verified like any other);
* the stage replays safely: allowed_stages, no add/remove contradiction, and
  every required communication relationship still has a valid trust path;
* every submitted field observation (trust anchors and endpoint chains)
  attests exactly the post-stage state the replay produces;
* the stored snapshot equals the snapshot the verifier independently builds
  for that state — actions, observations, complete trust-anchor set, complete
  endpoint-chain set and every relationship.

The first failing version ends verification: the reported ``verified_version``
is the number of confirmed-good versions and replay is **never** continued from
a faulty snapshot.

Scope: this establishes package-internal consistency only. It does not
authenticate the exporter and is no proof of origin; transport authenticity is
the caller's responsibility.
"""
from __future__ import annotations

from typing import Any, Optional

from .engine import (
    Scenario,
    ScenarioValidationError,
    TrustEdge,
    build_scenario,
    check_batch,
    first_broken_relationship,
    initial_state,
    state_for_mask_applied,
)
from .execution import _build_snapshot, _known_clients, _replay_failure_dto, _structural_batches
from .models import (
    CommitAnchorObservation,
    CommitChainObservation,
    ExecutionExport,
    ExecutionVerificationResult,
    PlannedStage,
    VerificationMismatch,
)


# ---------------------------------------------------------------------------
# Normalization helpers (order-independent content comparison)
# ---------------------------------------------------------------------------


def _anchor_obs_tuples(obss: list[CommitAnchorObservation]) -> list[tuple]:
    return sorted(
        ((o.client_id, o.anchor_cert_id, o.present) for o in obss),
        key=lambda t: (t[0], t[1]),
    )


def _chain_obs_tuples(obss: list[CommitChainObservation]) -> list[tuple]:
    return sorted(
        ((o.endpoint_id, tuple(o.chain)) for o in obss),
        key=lambda t: t[0],
    )


def _anchor_tuples(anchors) -> list[tuple]:
    return sorted((a.client_id, a.anchor_cert_id) for a in anchors)


def _endpoint_chain_tuples(chains) -> list[tuple]:
    return sorted((c.endpoint_id, tuple(c.chain)) for c in chains)


def _relationship_tuples(rels) -> list[tuple]:
    return sorted(
        (
            r.client_id,
            r.endpoint_id,
            r.valid,
            tuple(
                (c.cert_id, c.subject, c.issuer) for c in r.chain
            ),
            tuple(r.trusted_anchors),
        )
        for r in rels
    )


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _result(
    pkg: ExecutionExport,
    *,
    verified_version: int,
    mismatch: Optional[VerificationMismatch],
) -> ExecutionVerificationResult:
    finished = verified_version >= pkg.stage_count
    return ExecutionVerificationResult(
        status="ok" if mismatch is None else "conflict",
        execution_id=pkg.execution_id,
        verified_version=verified_version,
        version=pkg.version,
        stage_count=pkg.stage_count,
        next_stage=None if finished else verified_version + 1,
        finished=finished,
        mismatch=mismatch,
    )


def _fail(
    pkg: ExecutionExport,
    *,
    verified_version: int,
    version: int,
    stage: int,
    reason: str,
    message: str,
    expected: Any = None,
    found: Any = None,
    component: Optional[str] = None,
    relationship=None,
    blocking_action: Optional[str] = None,
    chain=None,
    trusted_anchors=None,
    detail: Any = None,
) -> ExecutionVerificationResult:
    mismatch = VerificationMismatch(
        version=version,
        stage=stage,
        reason=reason,
        message=message,
        expected=expected,
        found=found,
        component=component,
        relationship=relationship,
        blocking_action=blocking_action,
        chain=chain or [],
        trusted_anchors=trusted_anchors or [],
        detail=detail,
    )
    return _result(pkg, verified_version=verified_version, mismatch=mismatch)


# ---------------------------------------------------------------------------
# Snapshot comparison
# ---------------------------------------------------------------------------


def _compare_snapshot(
    pkg: ExecutionExport, *, verified_version: int, expected, found
) -> Optional[ExecutionVerificationResult]:
    """Return a conflict result if ``found`` differs from the expected snapshot.

    ``committed_at`` is local bookkeeping, not replay-derived content, so it is
    deliberately not compared. Components are checked in a fixed order so the
    reported first disagreement is stable.
    """

    version = found.version
    stage = found.stage

    if found.version != expected.version or found.stage != expected.stage:
        return _fail(
            pkg,
            verified_version=verified_version,
            version=version,
            stage=stage,
            reason="snapshot_mismatch",
            component="version",
            message=(
                f"snapshot header (version={found.version}, stage={found.stage}) "
                f"does not match the replay boundary "
                f"(version={expected.version}, stage={expected.stage})."
            ),
            expected={"version": expected.version, "stage": expected.stage},
            found={"version": found.version, "stage": found.stage},
        )

    checks = [
        ("action_ids", sorted(expected.action_ids), sorted(found.action_ids)),
        (
            "anchor_observations",
            _anchor_obs_tuples(expected.anchor_observations),
            _anchor_obs_tuples(found.anchor_observations),
        ),
        (
            "chain_observations",
            _chain_obs_tuples(expected.chain_observations),
            _chain_obs_tuples(found.chain_observations),
        ),
        (
            "trust_anchors",
            _anchor_tuples(expected.trust_anchors),
            _anchor_tuples(found.trust_anchors),
        ),
        (
            "endpoint_chains",
            _endpoint_chain_tuples(expected.endpoint_chains),
            _endpoint_chain_tuples(found.endpoint_chains),
        ),
        (
            "relationships",
            _relationship_tuples(expected.relationships),
            _relationship_tuples(found.relationships),
        ),
    ]
    for component, want, got in checks:
        if got != want:
            return _fail(
                pkg,
                verified_version=verified_version,
                version=version,
                stage=stage,
                reason="snapshot_mismatch",
                component=component,
                message=(
                    f"snapshot {component} at version {version} do not match "
                    f"the state the independent replay produces."
                ),
                expected=want,
                found=got,
            )
    return None


# ---------------------------------------------------------------------------
# Reference validation of observations (mirrors the commit-time checks)
# ---------------------------------------------------------------------------


def _validate_observation_refs(
    pkg: ExecutionExport, sc: Scenario, entry
) -> Optional[ExecutionVerificationResult]:
    clients = _known_clients(sc)
    seen_anchors: set[tuple[str, str]] = set()
    for obs in entry.anchor_observations:
        key = (obs.client_id, obs.anchor_cert_id)
        if key in seen_anchors:
            return _fail(
                pkg,
                verified_version=entry.version,
                version=entry.version,
                stage=entry.stage,
                reason="duplicate_observation",
                message=(
                    f"anchor observation {key} is reported more than once at "
                    f"version {entry.version}."
                ),
                detail={"kind": "anchor", "client_id": obs.client_id,
                        "anchor_cert_id": obs.anchor_cert_id},
            )
        seen_anchors.add(key)
        if obs.client_id not in clients:
            return _unknown_ref(
                pkg, entry, kind="anchor", ref="client_id", value=obs.client_id
            )
        if obs.anchor_cert_id not in sc.certs:
            return _unknown_ref(
                pkg, entry, kind="anchor", ref="anchor_cert_id",
                value=obs.anchor_cert_id,
            )
    seen_endpoints: set[str] = set()
    for obs in entry.chain_observations:
        if obs.endpoint_id in seen_endpoints:
            return _fail(
                pkg,
                verified_version=entry.version,
                version=entry.version,
                stage=entry.stage,
                reason="duplicate_observation",
                message=(
                    f"chain observation for endpoint '{obs.endpoint_id}' is "
                    f"reported more than once at version {entry.version}."
                ),
                detail={"kind": "endpoint_chain",
                        "endpoint_id": obs.endpoint_id},
            )
        seen_endpoints.add(obs.endpoint_id)
        if obs.endpoint_id not in sc.endpoint_chains:
            return _unknown_ref(
                pkg, entry, kind="endpoint_chain", ref="endpoint_id",
                value=obs.endpoint_id,
            )
        for cid in obs.chain:
            if cid not in sc.certs:
                return _unknown_ref(
                    pkg, entry, kind="endpoint_chain", ref="chain", value=cid,
                    detail={"endpoint_id": obs.endpoint_id},
                )
    return None


def _unknown_ref(pkg, entry, *, kind: str, ref: str, value: str, detail=None):
    return _fail(
        pkg,
        verified_version=entry.version,
        version=entry.version,
        stage=entry.stage,
        reason="unknown_reference",
        message=(
            f"{kind} observation at version {entry.version} references unknown "
            f"{ref} '{value}'."
        ),
        detail=detail or {"kind": kind, ref: value},
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_export(pkg: ExecutionExport) -> ExecutionVerificationResult:
    # ---- parse the scenario carried in the package -------------------------
    try:
        sc = build_scenario(pkg.scenario)
    except ScenarioValidationError as exc:
        return _fail(
            pkg,
            verified_version=0,
            version=0,
            stage=0,
            reason="invalid_scenario",
            message=f"the exported scenario fails validation: {exc}",
        )

    # ---- structurally validate the plan carried in the package -------------
    try:
        batches = _structural_batches(sc, list(pkg.plan))
    except ScenarioValidationError as exc:
        return _fail(
            pkg,
            verified_version=0,
            version=0,
            stage=0,
            reason="invalid_plan",
            message=f"the exported plan fails structural validation: {exc}",
        )

    # ---- package header vs. plan bookkeeping -------------------------------
    if pkg.stage_count != len(pkg.plan):
        return _fail(
            pkg,
            verified_version=0,
            version=0,
            stage=0,
            reason="header_mismatch",
            message=(
                f"package stage_count {pkg.stage_count} does not match the "
                f"number of plan stages {len(pkg.plan)}."
            ),
            expected=len(pkg.plan),
            found=pkg.stage_count,
        )

    # ---- commit sequence must be exactly versions 0..version ---------------
    if not pkg.commits or pkg.commits[0].version != 0:
        return _fail(
            pkg,
            verified_version=0,
            version=(pkg.commits[0].version if pkg.commits else None) or 0,
            stage=(pkg.commits[0].stage if pkg.commits else 0),
            reason="missing_initial_entry",
            message=(
                "the export has no version-0 initial entry; a package must "
                "start with the registration snapshot."
            ),
            expected=0,
            found=pkg.commits[0].version if pkg.commits else None,
        )

    seen_versions: set[int] = set()
    for pos, entry in enumerate(pkg.commits):
        expect_v = pos
        if entry.version in seen_versions:
            return _fail(
                pkg,
                verified_version=pos,
                version=entry.version,
                stage=entry.stage,
                reason="commit_duplicate_version",
                message=f"version {entry.version} appears more than once.",
                expected=None,
                found=entry.version,
            )
        seen_versions.add(entry.version)
        if entry.version != expect_v:
            # Covers both gaps (a missing version leaves a hole) and
            # out-of-order entries (version smaller than its position).
            reason = (
                "commit_gap"
                if entry.version > expect_v
                else "stage_out_of_order"
            )
            return _fail(
                pkg,
                verified_version=expect_v,
                version=entry.version,
                stage=entry.stage,
                reason=reason,
                message=(
                    f"commits must be consecutive from 0; expected version "
                    f"{expect_v} at position {pos}, found {entry.version}."
                ),
                expected=expect_v,
                found=entry.version,
            )
        if entry.version > pkg.version:
            return _fail(
                pkg,
                verified_version=pkg.version,
                version=entry.version,
                stage=entry.stage,
                reason="commit_extra",
                message=(
                    f"entry at version {entry.version} is newer than the "
                    f"package's declared version {pkg.version}; the package "
                    f"is not a single consistent version."
                ),
                expected=pkg.version,
                found=entry.version,
            )

    # Header declared version must itself be the last committed version.
    if pkg.commits[-1].version != pkg.version:
        return _fail(
            pkg,
            verified_version=min(pkg.version, pkg.commits[-1].version),
            version=pkg.commits[-1].version,
            stage=pkg.commits[-1].stage,
            reason="header_mismatch",
            message=(
                f"the last committed version {pkg.commits[-1].version} does "
                f"not match the package header version {pkg.version}."
            ),
            expected=pkg.version,
            found=pkg.commits[-1].version,
        )

    # ---- independent replay, one verified version at a time ----------------
    state = initial_state(sc)

    broken = first_broken_relationship(sc, state)
    if broken is not None:
        from .engine import ReplayFailure

        ridx, pf = broken
        rf = ReplayFailure(
            stage=0,
            relationship_index=ridx,
            blocking_action_index=None,
            reason=pf.reason,
            chain=pf.chain,
            failure_index=pf.failure_index,
            trusted_anchors=pf.trusted_anchors,
        )
        failure = _replay_failure_dto(sc, rf, status="invalid")
        return _fail(
            pkg,
            verified_version=0,
            version=0,
            stage=0,
            reason=failure.reason,
            message=failure.message,
            relationship=failure.relationship,
            blocking_action=failure.blocking_action,
            chain=failure.chain,
            trusted_anchors=failure.trusted_anchors,
        )

    for entry in pkg.commits:
        v = entry.version
        bad = _verify_entry(pkg, sc, batches, state, entry)
        if bad is not None:
            return bad
        # Advance the replay state only after the version fully verified, so
        # replay never proceeds from a faulty snapshot.
        if v >= 1:
            bits = 0
            index = {aid: i for i, aid in enumerate(sc.action_ids)}
            for aid in entry.action_ids:
                bits |= 1 << index[aid]
            state = state_for_mask_applied(sc, state, bits)

    # ---- header bookkeeping must agree with the verified prefix -----------
    expected_completed = list(range(1, pkg.version + 1))
    finished = pkg.version >= pkg.stage_count
    expected_next = None if finished else pkg.version + 1
    for field_name, want, got in (
        ("completed_stages", expected_completed, list(pkg.completed_stages)),
        ("next_stage", expected_next, pkg.next_stage),
        ("finished", finished, pkg.finished),
    ):
        if got != want:
            return _fail(
                pkg,
                verified_version=pkg.version,
                version=pkg.version,
                stage=pkg.version,
                reason="header_mismatch",
                message=(
                    f"package field {field_name} does not match the verified "
                    f"prefix."
                ),
                expected=want,
                found=got,
            )

    return _result(pkg, verified_version=pkg.version, mismatch=None)


def _verify_entry(pkg: ExecutionExport, sc: Scenario, batches, state, entry):
    """Verify one commit entry against the replayed prefix state.

    Returns a conflict result or None. Does NOT mutate ``state``.
    """

    v = entry.version
    verified_before = v  # versions 0..v-1 already confirmed

    # version/stage header consistency on the entry itself
    if entry.stage != v:
        return _fail(
            pkg,
            verified_version=verified_before,
            version=v,
            stage=entry.stage,
            reason="version_stage_mismatch",
            message=(
                f"entry at version {v} carries stage {entry.stage}; stage must "
                f"equal version for every committed stage (0 for the initial)."
            ),
            expected=v,
            found=entry.stage,
        )

    # ---- version 0: the registration/initial entry -------------------------
    if v == 0:
        if entry.action_ids or entry.anchor_observations or entry.chain_observations:
            return _fail(
                pkg,
                verified_version=0,
                version=0,
                stage=0,
                reason="initial_entry_not_empty",
                message=(
                    "the version-0 initial entry must carry no actions and no "
                    "observations."
                ),
                found={
                    "action_ids": list(entry.action_ids),
                    "anchor_observations": len(entry.anchor_observations),
                    "chain_observations": len(entry.chain_observations),
                },
            )
        expected_snapshot = _build_snapshot(
            sc,
            state,
            version=0,
            stage=0,
            action_ids=[],
            anchor_observations=[],
            chain_observations=[],
            committed_at=entry.committed_at,
        )
        return _compare_snapshot(
            pkg,
            verified_version=0,
            expected=expected_snapshot,
            found=entry.snapshot,
        )

    # ---- versions >= 1 must map onto the plan ------------------------------
    if v > pkg.stage_count:
        return _fail(
            pkg,
            verified_version=verified_before,
            version=v,
            stage=entry.stage,
            reason="version_ahead_of_plan",
            message=(
                f"version {v} commits a stage beyond the plan's stage_count "
                f"{pkg.stage_count}."
            ),
            expected=pkg.stage_count,
            found=v,
        )

    planned: PlannedStage = pkg.plan[v - 1]
    if entry.stage != planned.stage:
        return _fail(
            pkg,
            verified_version=verified_before,
            version=v,
            stage=entry.stage,
            reason="stage_out_of_order",
            message=(
                f"entry at version {v} must commit plan stage {planned.stage}, "
                f"not stage {entry.stage}."
            ),
            expected=planned.stage,
            found=entry.stage,
        )

    # ---- action set must equal exactly the plan stage ----------------------
    planned_actions = [sc.action_ids[i] for i in batches[v - 1]]
    if sorted(entry.action_ids) != sorted(planned_actions):
        return _fail(
            pkg,
            verified_version=verified_before,
            version=v,
            stage=entry.stage,
            reason="action_set_mismatch",
            message=(
                f"committed actions at version {v} do not equal the plan "
                f"stage's action set (empty actions required for a no-op "
                f"stage)."
            ),
            expected=sorted(planned_actions),
            found=sorted(entry.action_ids),
        )

    # ---- observation references must resolve in the scenario ---------------
    ref_bad = _validate_observation_refs(pkg, sc, entry)
    if ref_bad is not None:
        return ref_bad

    # ---- replay this stage against the verified prefix ---------------------
    index = {aid: i for i, aid in enumerate(sc.action_ids)}
    bits = 0
    for aid in entry.action_ids:
        bits |= 1 << index[aid]

    rf = check_batch(sc, state, bits, entry.stage)
    if rf is not None:
        failure = _replay_failure_dto(sc, rf, status="invalid")
        return _fail(
            pkg,
            verified_version=verified_before,
            version=v,
            stage=entry.stage,
            reason=failure.reason,
            message=failure.message,
            relationship=failure.relationship,
            blocking_action=failure.blocking_action,
            chain=failure.chain,
            trusted_anchors=failure.trusted_anchors,
        )

    end_state = state_for_mask_applied(sc, state, bits)

    # ---- observations must attest exactly the post-stage state -------------
    for obs in entry.anchor_observations:
        edge = TrustEdge(obs.client_id, obs.anchor_cert_id)
        expected_present = edge in end_state.anchors
        if expected_present != obs.present:
            return _fail(
                pkg,
                verified_version=verified_before,
                version=v,
                stage=entry.stage,
                reason="observation_mismatch",
                message=(
                    f"anchor observation ({obs.client_id}, "
                    f"{obs.anchor_cert_id}) present={str(obs.present).lower()} "
                    f"does not match the replayed post-stage state "
                    f"(present={str(expected_present).lower()})."
                ),
                detail={
                    "kind": "anchor",
                    "client_id": obs.client_id,
                    "anchor_cert_id": obs.anchor_cert_id,
                    "submitted_present": obs.present,
                    "expected_present": expected_present,
                },
            )
    for obs in entry.chain_observations:
        expected_chain = list(end_state.chains[obs.endpoint_id])
        if list(obs.chain) != expected_chain:
            return _fail(
                pkg,
                verified_version=verified_before,
                version=v,
                stage=entry.stage,
                reason="observation_mismatch",
                message=(
                    f"chain observation for endpoint '{obs.endpoint_id}' does "
                    f"not match the replayed post-stage chain."
                ),
                detail={
                    "kind": "endpoint_chain",
                    "endpoint_id": obs.endpoint_id,
                    "submitted_chain": list(obs.chain),
                    "expected_chain": expected_chain,
                },
            )

    # ---- the stored snapshot must equal an independently built one ---------
    expected_snapshot = _build_snapshot(
        sc,
        end_state,
        version=v,
        stage=entry.stage,
        action_ids=[sc.action_ids[i] for i in batches[v - 1]],
        anchor_observations=list(entry.anchor_observations),
        chain_observations=list(entry.chain_observations),
        committed_at=entry.committed_at,
    )
    return _compare_snapshot(
        pkg,
        verified_version=verified_before,
        expected=expected_snapshot,
        found=entry.snapshot,
    )
