"""Off-site review of an exported plan-execution audit package.

``POST /execution-reviews`` receives a package produced by
``GET /plan-executions/{id}/export`` — typically on a *different* instance —
and independently replays the committed prefix without any access to the
source SQLite store. Every committed stage (no-op stages included) is
checked:

* the committed action set against the immutable plan stage;
* the trust replay of those actions (allowed stages, no same-stage
  add/remove contradiction, every communication relationship keeps a valid
  trust path);
* the field observations against the replayed post-stage state;
* the exported snapshot (version, committed content, trust anchors,
  endpoint chains and every relationship) against the snapshot the reviewer
  recomputes on its own.

The replay state is always derived from the scenario plus the committed
actions — an exported snapshot is never adopted as the replay state — and
the review stops at the first stage that is missing, duplicated, out of
order or content-mismatched instead of continuing on top of a bad snapshot.

The verdict attests **package-internal consistency only**: a consistent
package could still have been fabricated by whoever produced it. It is not
a proof of the source execution's authenticity.
"""
from __future__ import annotations

from typing import Any, NoReturn, Optional

from .engine import (
    ReplayFailure,
    Scenario,
    ScenarioValidationError,
    TrustEdge,
    TrustState,
    build_scenario,
    check_batch,
    first_broken_relationship,
    initial_state,
    state_for_mask_applied,
)
from .execution import (
    _build_snapshot,
    _known_clients,
    _replay_message,
    _structural_batches,
)
from .models import (
    ChainCert,
    ExecutionExport,
    ExecutionReviewFailure,
    ExecutionReviewSuccess,
    RelationshipRef,
    StageSnapshot,
)


class ReviewInconsistent(Exception):
    """The package failed verification at some stage (HTTP 409)."""

    def __init__(self, failure: ExecutionReviewFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


# ---------------------------------------------------------------------------
# Failure helpers
# ---------------------------------------------------------------------------


def _fail(
    pkg: ExecutionExport,
    *,
    stage: int,
    reason: str,
    message: str,
    verified_version: Optional[int] = None,
    **context: Any,
) -> NoReturn:
    raise ReviewInconsistent(
        ExecutionReviewFailure(
            execution_id=pkg.execution_id,
            stage=stage,
            reason=reason,
            message=message,
            verified_version=verified_version,
            **context,
        )
    )


def _fail_replay(
    pkg: ExecutionExport,
    sc: Scenario,
    rf: ReplayFailure,
    *,
    verified_version: Optional[int],
) -> NoReturn:
    """Map a trust-replay failure to the review's 409 shape."""

    rel = sc.relationships[rf.relationship_index]
    _fail(
        pkg,
        stage=rf.stage,
        reason=rf.reason,
        message=_replay_message(sc, rf),
        verified_version=verified_version,
        relationship=RelationshipRef(
            client_id=rel.client_id, endpoint_id=rel.endpoint_id
        ),
        blocking_action=(
            sc.action_ids[rf.blocking_action_index]
            if rf.blocking_action_index is not None
            else None
        ),
        chain=[
            ChainCert(
                cert_id=cid,
                subject=sc.certs[cid].subject,
                issuer=sc.certs[cid].issuer,
            )
            for cid in rf.chain
        ],
        trusted_anchors=rf.trusted_anchors,
    )


def _jsonable(value: Any) -> list:
    """JSON-comparable form of a snapshot field, for expected/actual."""

    if isinstance(value, list):
        return [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in value
        ]
    return [value]


# ---------------------------------------------------------------------------
# Package-level reference validation (422) and stage-sequence checks (409)
# ---------------------------------------------------------------------------


def _validate_references(pkg: ExecutionExport, sc: Scenario) -> None:
    """Unknown ids inside committed observations are malformed input (422),
    exactly as they are on the live commit endpoint."""

    clients = _known_clients(sc)
    for snap in pkg.snapshots:
        for obs in snap.anchor_observations:
            if obs.client_id not in clients:
                raise ScenarioValidationError(
                    f"snapshots: unknown client_id '{obs.client_id}' in an "
                    f"anchor observation of stage {snap.stage}"
                )
            if obs.anchor_cert_id not in sc.certs:
                raise ScenarioValidationError(
                    f"snapshots: unknown certificate '{obs.anchor_cert_id}' "
                    f"in an anchor observation of stage {snap.stage}"
                )
        for obs in snap.chain_observations:
            if obs.endpoint_id not in sc.endpoint_chains:
                raise ScenarioValidationError(
                    f"snapshots: unknown endpoint_id '{obs.endpoint_id}' in "
                    f"a chain observation of stage {snap.stage}"
                )
            for cid in obs.chain:
                if cid not in sc.certs:
                    raise ScenarioValidationError(
                        f"snapshots: unknown certificate '{cid}' in the "
                        f"observed chain of endpoint '{obs.endpoint_id}' "
                        f"at stage {snap.stage}"
                    )


def _check_sequence(pkg: ExecutionExport) -> None:
    """The snapshots must be exactly stages 0..version, each once, in order."""

    if pkg.stage_count != len(pkg.plan):
        _fail(
            pkg,
            stage=0,
            reason="stage_count_mismatch",
            message=(
                f"the declared stage_count {pkg.stage_count} does not match "
                f"the exported plan, which has {len(pkg.plan)} stage(s)."
            ),
            expected=[len(pkg.plan)],
            actual=[pkg.stage_count],
        )

    seen: set[int] = set()
    for snap in pkg.snapshots:
        if snap.stage in seen:
            _fail(
                pkg,
                stage=snap.stage,
                reason="duplicate_stage",
                message=(
                    f"stage {snap.stage} has more than one snapshot in the "
                    f"package; every stage must appear exactly once."
                ),
            )
        seen.add(snap.stage)

    for k in range(pkg.version + 1):
        if k not in seen:
            what = "the initial snapshot" if k == 0 else f"the stage {k} snapshot"
            _fail(
                pkg,
                stage=k,
                reason="missing_stage",
                message=(
                    f"the package declares version {pkg.version} but "
                    f"{what} is missing."
                ),
            )

    for snap in pkg.snapshots:
        if (
            snap.stage < 0
            or snap.stage > pkg.version
            or snap.stage > pkg.stage_count
        ):
            _fail(
                pkg,
                stage=snap.stage,
                reason="unexpected_stage",
                message=(
                    f"stage {snap.stage} cannot belong to an export at "
                    f"version {pkg.version} of a {pkg.stage_count}-stage "
                    f"plan."
                ),
            )

    for prev, cur in zip(pkg.snapshots, pkg.snapshots[1:]):
        if cur.stage <= prev.stage:
            _fail(
                pkg,
                stage=cur.stage,
                reason="stage_out_of_order",
                message=(
                    f"snapshots are not in ascending stage order: stage "
                    f"{cur.stage} follows stage {prev.stage}."
                ),
            )


# ---------------------------------------------------------------------------
# Per-stage content checks
# ---------------------------------------------------------------------------


def _check_observations(
    pkg: ExecutionExport,
    end_state: TrustState,
    snap: StageSnapshot,
    stage: int,
) -> None:
    """Committed field observations must attest the replayed post-stage state."""

    for obs in snap.anchor_observations:
        edge = TrustEdge(obs.client_id, obs.anchor_cert_id)
        expected_present = edge in end_state.anchors
        if expected_present != obs.present:
            _fail(
                pkg,
                stage=stage,
                reason="observation_mismatch",
                message=(
                    f"stage {stage}: anchor observation ({obs.client_id}, "
                    f"{obs.anchor_cert_id}) present="
                    f"{str(obs.present).lower()} does not match the "
                    f"replayed post-stage state (present="
                    f"{str(expected_present).lower()})."
                ),
                verified_version=stage - 1,
                observation={
                    "kind": "anchor",
                    "client_id": obs.client_id,
                    "anchor_cert_id": obs.anchor_cert_id,
                    "submitted_present": obs.present,
                    "expected_present": expected_present,
                },
            )
    for obs in snap.chain_observations:
        expected_chain = list(end_state.chains[obs.endpoint_id])
        if list(obs.chain) != expected_chain:
            _fail(
                pkg,
                stage=stage,
                reason="observation_mismatch",
                message=(
                    f"stage {stage}: chain observation for endpoint "
                    f"'{obs.endpoint_id}' ['"
                    + "' -> '".join(obs.chain)
                    + "'] does not match the replayed post-stage chain ['"
                    + "' -> '".join(expected_chain)
                    + "']."
                ),
                verified_version=stage - 1,
                observation={
                    "kind": "endpoint_chain",
                    "endpoint_id": obs.endpoint_id,
                    "submitted_chain": list(obs.chain),
                    "expected_chain": expected_chain,
                },
            )


# Snapshot fields compared against the independent replay, in the order the
# first mismatch is reported.
_SNAPSHOT_FIELDS = (
    "version",
    "stage",
    "action_ids",
    "anchor_observations",
    "chain_observations",
    "trust_anchors",
    "endpoint_chains",
    "relationships",
)


def _check_snapshot(
    pkg: ExecutionExport,
    sc: Scenario,
    state: TrustState,
    snap: StageSnapshot,
    stage: int,
    *,
    planned_actions: list[str],
    anchor_observations: list,
    chain_observations: list,
) -> None:
    """The exported snapshot must equal the reviewer's own recomputation.

    The expected snapshot is built by the very same serializer the live
    service uses, so canonical ordering (sorted anchors/observations, plan
    order for action_ids, scenario order for relationships) is enforced
    exactly; ``committed_at`` is metadata and is not part of the comparison.
    For a committed stage the expected committed content is the package's
    own (it is validated against the replayed state separately); the
    initial snapshot must carry no committed content at all.
    """

    expected = _build_snapshot(
        sc,
        state,
        version=stage,
        stage=stage,
        action_ids=planned_actions,
        anchor_observations=anchor_observations,
        chain_observations=chain_observations,
        committed_at=snap.committed_at,
    )
    for field in _SNAPSHOT_FIELDS:
        exp = getattr(expected, field)
        act = getattr(snap, field)
        if act != exp:
            _fail(
                pkg,
                stage=stage,
                reason="snapshot_mismatch",
                message=(
                    f"stage {stage}: the exported snapshot's {field} does "
                    f"not match the independently replayed state."
                ),
                verified_version=stage - 1,
                snapshot_field=field,
                expected=_jsonable(exp),
                actual=_jsonable(act),
            )


# ---------------------------------------------------------------------------
# Review entry point
# ---------------------------------------------------------------------------


def review_export(pkg: ExecutionExport) -> ExecutionReviewSuccess:
    """Independently replay and verify an exported execution package.

    Raises ``ScenarioValidationError`` (HTTP 422) when the package cannot
    even be interpreted and ``ReviewInconsistent`` (HTTP 409) at the first
    stage that is missing, duplicated, out of order or content-mismatched.
    """

    sc = build_scenario(pkg.scenario)
    batches = _structural_batches(sc, pkg.plan)
    _validate_references(pkg, sc)
    _check_sequence(pkg)

    index = {aid: i for i, aid in enumerate(sc.action_ids)}
    state = initial_state(sc)

    # ---- version 0: the initial snapshot ----------------------------------
    initial = pkg.snapshots[0]
    broken = first_broken_relationship(sc, state)
    if broken is not None:
        ridx, pf = broken
        _fail_replay(
            pkg,
            sc,
            ReplayFailure(
                stage=0,
                relationship_index=ridx,
                blocking_action_index=None,
                reason=pf.reason,
                chain=pf.chain,
                failure_index=pf.failure_index,
                trusted_anchors=pf.trusted_anchors,
            ),
            verified_version=None,
        )
    _check_snapshot(
        pkg, sc, state, initial, 0,
        planned_actions=[], anchor_observations=[], chain_observations=[],
    )

    # ---- committed prefix: replay every stage, no-op stages included ------
    for k in range(1, pkg.version + 1):
        snap = pkg.snapshots[k]
        planned = [sc.action_ids[i] for i in batches[k - 1]]
        if sorted(snap.action_ids) != sorted(planned):
            _fail(
                pkg,
                stage=k,
                reason="action_set_mismatch",
                message=(
                    f"stage {k}: the committed action set "
                    f"[{', '.join(sorted(snap.action_ids)) or 'none'}] does "
                    f"not match the plan stage "
                    f"[{', '.join(sorted(planned)) or 'none'}]."
                ),
                verified_version=k - 1,
                expected=sorted(planned),
                actual=sorted(snap.action_ids),
            )

        bits = 0
        for aid in snap.action_ids:
            bits |= 1 << index[aid]

        # Trust replay of the committed actions: allowed_stages, same-stage
        # add/remove contradictions and every relationship's trust path.
        rf = check_batch(sc, state, bits, k)
        if rf is not None:
            _fail_replay(pkg, sc, rf, verified_version=k - 1)

        # The replay state is always computed from the committed actions,
        # never adopted from the exported snapshot.
        end_state = state_for_mask_applied(sc, state, bits)
        _check_observations(pkg, end_state, snap, k)
        _check_snapshot(
            pkg, sc, end_state, snap, k,
            planned_actions=planned,
            anchor_observations=list(snap.anchor_observations),
            chain_observations=list(snap.chain_observations),
        )
        state = end_state

    finished = pkg.version >= pkg.stage_count
    return ExecutionReviewSuccess(
        execution_id=pkg.execution_id,
        verified_version=pkg.version,
        stage_count=pkg.stage_count,
        next_stage=None if finished else pkg.version + 1,
        finished=finished,
        message=_success_message(pkg, finished=finished),
    )


def _success_message(pkg: ExecutionExport, *, finished: bool) -> str:
    base = (
        f"all {pkg.version} committed stage(s) of execution "
        f"'{pkg.execution_id}' (including any no-op stages) replay exactly "
        f"to their exported snapshots; the package is internally consistent"
    )
    if finished:
        progress = "the plan is fully executed"
    else:
        progress = (
            f"the plan is not finished: verified through version "
            f"{pkg.version} of {pkg.stage_count} stages; the next stage to "
            f"commit is {pkg.version + 1}"
        )
    return (
        f"{base}; {progress}. This verdict attests only the internal "
        f"consistency of the exported package, not the authenticity of its "
        f"source."
    )
