from __future__ import annotations

import argparse

from .core import compute_metrics_sl


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute table-level schema linking metrics.")
    parser.add_argument("--linked-json", required=True)
    parser.add_argument("--db-path", required=True)
    args = parser.parse_args()
    compute_metrics_sl(args.linked_json, args.db_path)


if __name__ == "__main__":
    main()
