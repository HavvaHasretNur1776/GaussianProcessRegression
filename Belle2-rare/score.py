"""Evaluate one frozen validation-selected checkpoint on the bundled test split."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "studies" / "inducing_scan"))
sys.path.insert(0, str(ROOT / "studies" / "gpr"))

def score(run, device):
    import numpy as np
    import tensorflow as tf
    from gpr_common import load_data, apply_restored, residual_metrics

    info = json.loads((run / "result.json").read_text())
    from swan_runner import DATA
    data = load_data(DATA)
    model_path = run / "model"
    model_info = json.loads((model_path / "model.json").read_text())
    if data["sha256"] != model_info["data_sha256"] or data["sha256"] != info["input_sha256"]:
        raise ValueError("Checkpoint or run belongs to another dataset")
    rows = np.flatnonzero(data["split"] == 2)
    truth = data["q2_true"][rows]
    baseline = data["q2_energy_mass"][rows]
    with tf.device(device):
        pred, _, _ = apply_restored(model_path, data, rows)
    metrics = {name: residual_metrics(q - truth) for name, q in
               (("angular_gp", pred["q2"]), ("energy_only", baseline))}
    metrics["rmse_reduction_percent"] = 100 * (1 - metrics["angular_gp"]["rmse"] / metrics["energy_only"]["rmse"])
    np.savez_compressed(run / "test_predictions.npz", event_id=data["event_id"][rows],
                        generator_seed=data["generator_seed"][rows], q2_true=truth,
                        q2_gp=pred["q2"], q2_energy=baseline, cos_prediction=pred["cos_mean"])
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    from device import configure_device
    device, _ = configure_device("gpu", 4)
    score(Path(sys.argv[1]), device)
