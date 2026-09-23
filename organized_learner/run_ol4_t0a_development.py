"""Launch the prospectively frozen OL4-T0a identity and outer-run stages.

Usage from the repository root::

    python -m organized_learner.run_ol4_t0a_development freeze-identities
    python -m organized_learner.run_ol4_t0a_development run --execute-frozen

The second command opens the four registered run IDs only after verifying the
committed PASS preflight, source inventory, and frozen identity archive.
"""
from __future__ import annotations

import argparse
import json

from .ol4_training import freeze_identities, run_registered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze-identities")
    run = commands.add_parser("run")
    run.add_argument("--execute-frozen", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "freeze-identities":
        outcome = freeze_identities()
        print(json.dumps({"stage": "identities", "life_count": outcome["life_count"],
                          "archive_sha256": outcome["identity_archive_sha256"]},
                         sort_keys=True))
        return 0
    outcome = run_registered()
    print(json.dumps({"stage": "development", "verdict": outcome["verdict"]},
                     sort_keys=True))
    return 0 if outcome["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
