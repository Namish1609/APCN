from __future__ import annotations
import argparse, json
from apcn_v09.session import SemanticSessionV09


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("memory")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--held-out-templates", action="store_true")
    a = p.parse_args()
    s = SemanticSessionV09.load(a.memory)
    rep = s.test(a.samples, a.held_out_templates)
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
