"""Run from rfp-v0: .venv/Scripts/python -m eval.v4.run_offline"""

from __future__ import annotations

import json

from backend.v4.evaluation import evaluate


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
