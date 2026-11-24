"""Compare saved validation predictions from two trainable, 11-input fits."""
import argparse
import json
from pathlib import Path
import numpy as np
from swan_runner import DATA_SHA256

def compare(reference, current):
    reports = [json.loads((p / "result.json").read_text()) for p in (reference, current)]
    config = [r["config"] for r in reports]
    for report in reports:
        if report["input_sha256"] != DATA_SHA256 or not report["config"]["train_inducing"]:
            raise ValueError("Both fits must use the bundled data and trainable locations")
    if config[0]["inducing"] != 300:
        raise ValueError("Reference must be the trainable M=300 fit")
    for key in ("train_seed", "init_seed", "train_cap", "maxiter", "validate_every", "device", "threads"):
        if config[0][key] != config[1][key]:
            raise ValueError(f"Fit configurations differ in {key}")
    with np.load(reference/"model/training_state.npz") as a, np.load(current/"model/training_state.npz") as b:
        for key in ("training_event_id", "training_generator_seed", "X_train", "y_train"):
            np.testing.assert_array_equal(a[key], b[key])
    with np.load(reference/"validation_predictions.npz") as a, np.load(current/"validation_predictions.npz") as b:
        for key in ("event_id", "generator_seed", "q2_true"):
            np.testing.assert_array_equal(a[key], b[key])
        a2 = (a["q2"] - a["q2_true"])**2
        b2 = (b["q2"] - b["q2_true"])**2
    rng = np.random.default_rng(120)
    differences = []
    for _ in range(2500):
        rows = rng.integers(len(a2), size=len(a2))
        differences.append(np.sqrt(b2[rows].mean()) - np.sqrt(a2[rows].mean()))
    old, new = float(np.sqrt(a2.mean())), float(np.sqrt(b2.mean()))
    result = dict(reference_M=300, current_M=config[1]["inducing"], validation_rows=len(a2),
                  reference_rmse=old, current_rmse=new, delta_rmse=new-old,
                  paired_95_interval=np.quantile(differences, [.025,.975]).tolist(), units="GeV²")
    (current/"paired_vs_M300.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("current", type=Path)
    args = p.parse_args()
    compare(args.reference, args.current)
