"""Shared data, GPflow SGPR and evaluation functions."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_data(path):
    with np.load(path, allow_pickle=False) as f:
        required = ["X", "y_cos_bk", "q2_true", "event_id", "generator_seed", "split",
                    "feature_names", "kaon_only_columns", "q2_energy_belle", "q2_energy_mass",
                    "q2_roe_direction", "metadata_json"]
        missing = [k for k in required if k not in f]
        if missing:
            raise ValueError("Missing fields: " + ", ".join(missing))
        d = {k: np.array(f[k]) for k in required}
    d["X"] = d["X"].astype(np.float64)
    d["y_cos_bk"] = d["y_cos_bk"].astype(np.float64).reshape(-1, 1)
    d["feature_names"] = [str(k) for k in d["feature_names"]]
    d["metadata"] = json.loads(str(d.pop("metadata_json")))
    n = len(d["X"])
    if d["X"].ndim != 2 or d["X"].shape[1] != len(d["feature_names"]):
        raise ValueError("Feature matrix/schema mismatch")
    if len(set(d["feature_names"])) != len(d["feature_names"]):
        raise ValueError("Duplicate feature names")
    for k in ["y_cos_bk", "q2_true", "event_id", "generator_seed", "split",
              "q2_energy_belle", "q2_energy_mass", "q2_roe_direction"]:
        if len(d[k]) != n:
            raise ValueError("Row mismatch for " + k)
    for k in ["X", "y_cos_bk", "q2_true", "q2_energy_belle", "q2_energy_mass", "q2_roe_direction"]:
        if not np.isfinite(d[k]).all():
            raise ValueError("Non-finite values in " + k)
    if np.max(np.abs(d["y_cos_bk"])) > 1+1e-8:
        raise ValueError("Angular targets outside [-1,1]")
    if not np.isin(d["split"], [0, 1, 2]).all():
        raise ValueError("split must be 0=train, 1=validation, 2=test")
    groups = np.column_stack([d["generator_seed"], d["event_id"]])
    if len(np.unique(groups, axis=0)) != n:
        raise ValueError("Duplicate generator-seed/event groups")
    gen = d["metadata"]["generator"]
    if gen["frame"] != "ee_CM":
        raise ValueError("Expected CM-frame features")
    d["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return d


def choose_training_rows(data, cap, seed):
    rows = np.flatnonzero(data["split"] == 0)
    if len(rows) < 20:
        raise ValueError("Need at least 20 training events")
    if cap > 0 and cap < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(rows, cap, replace=False))
    return rows


def fit_scaler(x, y):
    mean, std = x.mean(axis=0), x.std(axis=0)
    active = np.flatnonzero(std > 1e-10)
    if not len(active):
        raise ValueError("No varying training features")
    ymean, ystd = float(y.mean()), float(y.std())
    if ystd < 1e-10:
        raise ValueError("Constant angular training target")
    return {"active": active, "x_mean": mean[active], "x_std": std[active],
            "y_mean": ymean, "y_std": ystd}


def standardise_x(x, scaler):
    return (x[:, scaler["active"]]-scaler["x_mean"])/scaler["x_std"]


def build_model(x, y, z, train_inducing=False):
    import gpflow
    gpflow.config.set_default_float(np.float64)
    gpflow.config.set_default_jitter(1e-6)
    d = x.shape[1]
    kernel = (gpflow.kernels.Matern52(variance=1., lengthscales=np.ones(d))
              + gpflow.kernels.Linear(variance=0.05)
              + gpflow.kernels.Constant(variance=0.05))
    mean = gpflow.mean_functions.Linear(A=np.zeros((d, 1)), b=np.zeros(1))
    model = gpflow.models.SGPR((x, y), kernel=kernel, inducing_variable=z,
                              mean_function=mean, noise_variance=0.5)
    model.likelihood.variance = gpflow.Parameter(
        0.5, transform=gpflow.utilities.positive(lower=1e-5))
    gpflow.set_trainable(model.inducing_variable, train_inducing)
    return model


def parameter_values(model):
    import gpflow
    return {k: np.array(v.numpy(), copy=True) for k, v in gpflow.utilities.parameter_dict(model).items()}


def predict_angle(model, x, scaler, batch_size=2048):
    means, observation_variances, latent_variances = [], [], []
    noise = float(model.likelihood.variance.numpy())
    # Gaussian predict_y variance includes the learned event-level residual
    # variance. predict_f alone describes uncertainty in the latent mean.
    for start in range(0, len(x), batch_size):
        mu, var = model.predict_f(x[start:start+batch_size])
        means.append(mu.numpy().reshape(-1))
        latent_variances.append(var.numpy().reshape(-1))
        observation_variances.append(var.numpy().reshape(-1)+noise)
    mu = np.concatenate(means)*scaler["y_std"]+scaler["y_mean"]
    vy = np.maximum(np.concatenate(observation_variances), 0)*scaler["y_std"]**2
    vf = np.maximum(np.concatenate(latent_variances), 0)*scaler["y_std"]**2
    return {"cos_mean_raw": mu, "cos_mean": np.clip(mu, -1, 1),
            "cos_sigma_predictive": np.sqrt(vy), "cos_sigma_latent_mean": np.sqrt(vf)}


def make_q2(data, rows, cos_mean):
    gen = data["metadata"]["generator"]
    mb, mk, sqrts = gen["mB_GeV"], gen["mK_GeV"], gen["sqrt_s_GeV"]
    energy_column = data["feature_names"].index("k_E_cm")
    energy = data["X"][rows, energy_column]
    pk = np.sqrt(np.maximum(energy**2-mk**2, 0))
    pb = np.sqrt(sqrts**2/4-mb**2)
    return data["q2_energy_mass"][rows]+2*pb*pk*np.asarray(cos_mean).reshape(-1)


def residual_metrics(residual):
    r = np.asarray(residual)
    q16, q84 = np.quantile(r, [0.16, 0.84])
    return {"n": len(r), "bias": float(r.mean()), "std": float(r.std()),
            "rmse": float(np.sqrt(np.mean(r*r))), "half_central_68": float((q84-q16)/2),
            "abs_residual_p95": float(np.quantile(np.abs(r), .95)),
            "fraction_abs_residual_above_1_GeV2": float(np.mean(np.abs(r) > 1))}


def binned_metrics(truth, predictions, edges):
    output = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (truth >= lo) & ((truth < hi) if i < len(edges)-2 else (truth <= hi))
        row = {"low": float(lo), "high": float(hi), "n": int(mask.sum())}
        if mask.any():
            row.update(residual_metrics(predictions[mask]-truth[mask]))
        output.append(row)
    return output


def response_matrix(truth, prediction, edges):
    # All reconstructed underflows/overflows are retained as explicit columns.
    row = np.searchsorted(edges, truth, side="right")-1
    row = np.clip(row, 0, len(edges)-2)
    column = np.searchsorted(edges, prediction, side="right")
    matrix = np.zeros((len(edges)-1, len(edges)+1), dtype=np.int64)
    np.add.at(matrix, (row, column), 1)
    totals = matrix.sum(axis=1, keepdims=True)
    probability = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
    return matrix, probability


def paired_rmse_bootstrap(truth, new, reference, seed, repeats=1000):
    rng = np.random.default_rng(seed)
    new2, ref2 = (new-truth)**2, (reference-truth)**2
    changes = np.empty(repeats)
    for i in range(repeats):
        rows = rng.integers(0, len(truth), len(truth))
        changes[i] = 1-np.sqrt(new2[rows].mean()/ref2[rows].mean())
    lo, hi = np.quantile(changes, [.025, .975])
    return {"fractional_rmse_reduction": float(1-np.sqrt(new2.mean()/ref2.mean())),
            "paired_test_bootstrap_95_interval": [float(lo), float(hi)],
            "bootstrap_repetitions": repeats,
            "scope": "Test-sample statistics for these fixed trained models; no training/systematic uncertainty"}


def angular_calibration(y, prediction):
    from scipy.stats import norm
    mu, sigma = prediction["cos_mean_raw"], prediction["cos_sigma_predictive"]
    residual = np.asarray(y).reshape(-1)-mu
    coverage = {str(c): float(np.mean(np.abs(residual) <= norm.ppf((1+c)/2)*sigma))
                for c in [.50, .68, .90, .95]}
    pull = residual/np.maximum(sigma, 1e-12)
    return {"gaussian_angular_interval_coverage": coverage,
            "angular_pull_mean": float(pull.mean()), "angular_pull_std": float(pull.std()),
            "fraction_means_clipped_to_physical_cosine": float(np.mean(np.abs(mu) > 1)),
            "note": "Uses observation predictive variance, not latent-mean variance; not a full q2 uncertainty"}


def save_model(directory, model, x, y, scaler, columns, data, rows, info):
    directory = Path(directory)
    directory.mkdir(parents=True)
    np.savez_compressed(directory/"parameters.npz", **parameter_values(model))
    np.savez_compressed(directory/"training_state.npz", X_train=x, y_train=y,
                        x_mean=scaler["x_mean"], x_std=scaler["x_std"],
                        y_mean=scaler["y_mean"], y_std=scaler["y_std"],
                        training_event_id=data["event_id"][rows], training_generator_seed=data["generator_seed"][rows])
    selected_names = [data["feature_names"][columns[i]] for i in scaler["active"]]
    metadata = {"schema_version": 1, "feature_names": selected_names,
                "inducing_points": int(model.inducing_variable.Z.shape[0]),
                "data_sha256": data["sha256"], "input_metadata": data["metadata"], **info}
    (directory/"model.json").write_text(json.dumps(metadata, indent=2)+"\n")


def restore_model(directory):
    import gpflow
    directory = Path(directory)
    metadata = json.loads((directory/"model.json").read_text())
    with np.load(directory/"training_state.npz", allow_pickle=False) as f:
        state = {k: np.array(f[k]) for k in f.files}
    with np.load(directory/"parameters.npz", allow_pickle=False) as f:
        params = {k: np.array(f[k]) for k in f.files}
    x, y = state["X_train"], state["y_train"]
    model = build_model(x, y, x[:metadata["inducing_points"]].copy(), train_inducing=metadata.get("train_inducing", False))
    gpflow.utilities.multiple_assign(model, params)
    scaler = {"active": np.arange(x.shape[1]), "x_mean": state["x_mean"], "x_std": state["x_std"],
              "y_mean": float(state["y_mean"]), "y_std": float(state["y_std"])}
    return model, scaler, metadata, state


def apply_restored(directory, data, rows):
    model, scaler, metadata, state = restore_model(directory)
    columns = [data["feature_names"].index(name) for name in metadata["feature_names"]]
    x = standardise_x(data["X"][rows][:, columns], scaler)
    predictions = predict_angle(model, x, scaler)
    predictions["q2"] = make_q2(data, rows, predictions["cos_mean"])
    return predictions, metadata, state
