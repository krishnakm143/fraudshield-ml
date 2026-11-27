import json

from fraudshield.cli import main


def test_cli_train_then_drift(tmp_path, small_df, next_month_df, capsys):
    data = tmp_path / "tx.csv.gz"
    cur = tmp_path / "nm.csv.gz"
    small_df.to_csv(data, index=False)
    next_month_df.to_csv(cur, index=False)
    reg = tmp_path / "registry"
    assert main(["train", "--data", str(data), "--registry", str(reg)]) == 0
    assert (reg / "v0001" / "model.joblib").exists()
    meta = json.loads((reg / "v0001" / "metadata.json").read_text())
    assert meta["data"]["rows"] == len(small_df) and meta["threshold"] > 0
    out = tmp_path / "drift.json"
    assert main(["drift", "--data", str(data), "--current", str(cur), "--registry", str(reg), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["model_version"] == "v0001"
    assert "Recommended action" in capsys.readouterr().out
