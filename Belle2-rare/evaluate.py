"""Compare a completed fit with the Belle II inclusive-tag estimator on matched toy events."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swan_runner import DATA

def evaluate(run, split, data_path=DATA):
    info = json.loads((run/"result.json").read_text())
    if info["input_sha256"] != hashlib.sha256(data_path.read_bytes()).hexdigest():
        raise ValueError("The supplied dataset does not match the dataset used for this fit")
    split_id, field = {"validation": (1, "q2"), "test": (2, "q2_gp")}[split]
    with np.load(data_path) as data, np.load(run/f"{split}_predictions.npz") as pred:
        keep = data["split"] == split_id
        for key in ("event_id", "generator_seed", "q2_true"):
            np.testing.assert_array_equal(data[key][keep], pred[key])
        truth = pred["q2_true"]
        gp = pred[field]
        ita = data["q2_energy_belle"][keep]
        gen = json.loads(str(data["metadata_json"]))["generator"]
        k_energy = data["X"][keep, data["feature_names"].tolist().index("k_E_cm")]
        sqrt_s = gen["sqrt_s_GeV"]
        np.testing.assert_allclose(ita, sqrt_s**2/4 + gen["mK_GeV"]**2 - sqrt_s*k_energy, atol=1e-12, rtol=0)

    models = {"ITA estimator on toy": ita, "Angular GP on toy": gp}
    metrics = {}
    for label, values in models.items():
        residual = values - truth
        lower, upper = np.quantile(residual, [0.16, 0.84])
        metrics[label] = dict(rmse=float(np.sqrt(np.mean(residual**2))), bias=float(residual.mean()),
                              central_68_half_width=float((upper-lower)/2))

    base_sq = (ita-truth)**2
    gp_sq = (gp-truth)**2
    gain = 100*(1-np.sqrt(gp_sq.mean()/base_sq.mean()))
    rng = np.random.default_rng(120)
    bootstrap = []
    for _ in range(2500):
        rows = rng.integers(len(truth), size=len(truth))
        bootstrap.append(100*(1-np.sqrt(gp_sq[rows].mean()/base_sq[rows].mean())))
    summary = dict(split=split, n=len(truth), units="GeV²", metrics=metrics, rmse_reduction_percent=gain,
                   paired_bootstrap_95_percent=np.quantile(bootstrap, [0.025,0.975]).tolist())
    (run/f"paired_{split}.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, (spectrum, error) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bins = np.histogram_bin_edges(np.concatenate([truth, ita, gp]), bins=45)
    residual_bins = np.histogram_bin_edges(np.concatenate([ita-truth, gp-truth]), bins=60)
    spectrum.hist(truth, bins=bins, histtype="step", ls=":", lw=2, color="black", label="Generator truth")
    for (label, values), colour in zip(models.items(), ["tab:orange","tab:blue"]):
        spectrum.hist(values, bins=bins, histtype="step", lw=1.6, color=colour, label=label)
        error.hist(values-truth, bins=residual_bins, histtype="step", lw=1.6, color=colour, label=label)
    spectrum.set(xlabel="q² [GeV²]", title=f"{split.capitalize()} spectrum", ylabel="Events / bin")
    error.set(xlabel="Reconstructed − true q² [GeV²]", title="Reconstruction residual", ylabel="Events / bin")
    for ax in (spectrum, error):
        ax.legend(fontsize=8)
    fig.savefig(run/f"paired_{split}.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--split", choices=["validation","test"], default="validation")
    parser.add_argument("--input", type=Path, default=DATA, help="Feature NPZ used to fit the model")
    args = parser.parse_args()
    evaluate(args.run, args.split, args.input)
