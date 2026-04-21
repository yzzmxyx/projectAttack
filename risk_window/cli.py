"""Command-line interface for risk_window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .assets import validate_asset_root
from .config import load_risk_window_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="risk_window utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-assets", help="Validate an asset root")
    validate_parser.add_argument("--asset-root", required=True, type=str)

    inspect_parser = subparsers.add_parser("inspect-config", help="Inspect a JSON/YAML config or defaults")
    inspect_parser.add_argument("--config", default="", type=str)

    for command_name in ("replay-libero", "replay-bridge"):
        replay_parser = subparsers.add_parser(command_name, help=f"Run detector over an offline {command_name.split('-')[1]} video")
        replay_parser.add_argument("--asset-root", required=True, type=str)
        replay_parser.add_argument("--video", required=True, type=str)
        replay_parser.add_argument("--config", default="", type=str)
        replay_parser.add_argument("--task-id", default="", type=str)
        replay_parser.add_argument("--episode-id", default="", type=str)
        replay_parser.add_argument("--init-state-idx", default="", type=str)
        replay_parser.add_argument("--log-dir", default="", type=str)
    return parser


def _run_replay(args) -> dict:
    from .model import RiskWindowDetector
    try:  # pragma: no cover - imageio availability is environment-specific
        import imageio.v2 as imageio
    except Exception:  # pragma: no cover
        import imageio

    detector = RiskWindowDetector.from_config(
        config_path=str(args.config),
        asset_root=str(args.asset_root),
        log_dir=str(args.log_dir),
    )
    detector.reset(
        task_id=str(args.task_id) if str(args.task_id).strip() else "default",
        episode_id=str(args.episode_id) if str(args.episode_id).strip() else None,
        init_state_idx=int(args.init_state_idx) if str(args.init_state_idx).strip() else None,
    )
    reader = imageio.get_reader(Path(args.video).expanduser().resolve().as_posix())
    for index, frame in enumerate(reader):
        detector.predict(frame=frame, timestamp=float(index))
    reader.close()
    return detector.flush()


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-assets":
        print(json.dumps(validate_asset_root(args.asset_root), indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if args.command == "inspect-config":
        print(json.dumps(load_risk_window_config(args.config).to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if args.command in ("replay-libero", "replay-bridge"):
        print(json.dumps(_run_replay(args), indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
