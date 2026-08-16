from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_config
from app.container import build_rag_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one document into the configured store")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    summary = build_rag_service(get_config()).ingest(args.path)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

