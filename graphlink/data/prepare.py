from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paths import load_paths, validate_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GraphLink dataset layout and create output directories.")
    parser.add_argument("--config", default="configs/paths.example.yaml")
    parser.add_argument("--require-existing", action="store_true")
    parser.add_argument("--create-output-dirs", action="store_true")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    paths = load_paths(args.config)
    if args.create_output_dirs:
        for path in (paths.data_root, paths.output_root):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
    summary = validate_paths(paths, require_existing=args.require_existing)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.summary:
        Path(args.summary).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
