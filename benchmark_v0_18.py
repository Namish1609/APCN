from __future__ import annotations

import json

from apcn_v18.benchmark import run_natural_conversation_benchmark


if __name__ == "__main__":
    report = run_natural_conversation_benchmark()
    print(json.dumps(report.to_dict(), indent=2))
