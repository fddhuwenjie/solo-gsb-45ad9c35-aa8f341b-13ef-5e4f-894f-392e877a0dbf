#!/usr/bin/env python3
"""Regenerate examples/execution_review_request.json.

Builds a real export package with the service itself: the scenario and plan
of examples/execution_register_request.json are registered, stage 1 of 2 is
committed (examples/execution_commit_stage1.json) and the mid-plan export is
written out. The package therefore verifies as ``consistent`` with
``verified_version: 1`` and ``next_stage: 2`` against any fresh instance:

    curl -s -X POST http://localhost:8080/execution-reviews \\
      -H 'Content-Type: application/json' \\
      --data @examples/execution_review_request.json

Run from the repository root:  python3 examples/gen_execution_review_example.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# The store must live somewhere throwaway before app.execution is used.
_tmpdir = tempfile.TemporaryDirectory()
os.environ["ROTATION_DB_PATH"] = str(Path(_tmpdir.name) / "rotation.db")

from app import execution  # noqa: E402
from app.models import ExecutionRegisterRequest, StageCommitRequest  # noqa: E402
from app.review import review_export  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent


def main() -> None:
    register_req = json.loads(
        (EXAMPLES / "execution_register_request.json").read_text()
    )
    stage1_commit = json.loads(
        (EXAMPLES / "execution_commit_stage1.json").read_text()
    )

    status = execution.register_execution(
        ExecutionRegisterRequest.model_validate(register_req)
    )
    execution.commit_stage(
        status.execution_id, StageCommitRequest.model_validate(stage1_commit)
    )

    # Export mid-plan (version 1 of 2): the review of this package reports
    # the verified version and the next stage to commit.
    pkg = execution.export_execution(status.execution_id)

    # Sanity: the generated package must verify consistently.
    report = review_export(pkg)
    assert report.status == "consistent"
    assert report.verified_version == 1 and report.next_stage == 2
    assert report.finished is False

    out = EXAMPLES / "execution_review_request.json"
    out.write_text(
        json.dumps(json.loads(pkg.model_dump_json()), indent=2) + "\n"
    )
    print(f"wrote {out} (execution_id={pkg.execution_id})")


if __name__ == "__main__":
    main()
