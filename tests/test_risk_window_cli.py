import json
from pathlib import Path

from risk_window.cli import main


def test_cli_validate_assets_and_inspect_config(tmp_path: Path, capsys):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "top_windows.json").write_text(json.dumps({"top_windows": []}), encoding="utf-8")

    rc = main(["validate-assets", "--asset-root", str(asset_root)])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["top_windows_exists"] is True

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"runtime": {"log_dir": "/tmp/risk_window"}}), encoding="utf-8")
    rc = main(["inspect-config", "--config", str(config_path)])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["runtime"]["log_dir"] == "/tmp/risk_window"
