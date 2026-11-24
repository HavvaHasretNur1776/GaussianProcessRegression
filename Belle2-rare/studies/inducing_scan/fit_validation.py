#!/usr/bin/env python3
"""Fit one angular SGPR using training/validation only; never evaluate test labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"gpr"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--inducing", type=int, required=True)
    p.add_argument("--train-cap", type=int, default=5000)
    p.add_argument("--train-seed", type=int, default=120)
    p.add_argument("--init-seed", type=int, default=120)
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--validate-every", type=int, default=25)
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--train-inducing", action="store_true")
    p.add_argument("--initial-centres", type=Path, help="Reuse a sibling fit’s saved k-means initialization")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.outdir.exists():
        raise ValueError("Output exists")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    from device import configure_device
    device, hardware = configure_device(args.device, args.threads)
    import numpy as np
    import tensorflow as tf
    import gpflow
    from gpr_common import (load_data, choose_training_rows, fit_scaler, standardise_x, build_model,
                            predict_angle, make_q2, residual_metrics, binned_metrics,
                            parameter_values, save_model, angular_calibration)
    from inducing import inducing_kmeans
    data = load_data(args.input)
    rows = choose_training_rows(data, args.train_cap, args.train_seed)
    validation = np.flatnonzero(data["split"] == 1)
    if len(rows) < args.inducing or len(validation) < 20:
        raise ValueError("Insufficient training/validation rows")
    args.outdir.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.outdir/"run_config.json").write_text(json.dumps(config, indent=2)+"\n")
    np.random.seed(args.init_seed);tf.random.set_seed(args.init_seed)
    scaler = fit_scaler(data["X"][rows], data["y_cos_bk"][rows])
    x = standardise_x(data["X"][rows], scaler)
    y = (data["y_cos_bk"][rows]-scaler["y_mean"])/scaler["y_std"]
    xv = standardise_x(data["X"][validation], scaler)
    started = time.monotonic()
    design_hash = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    if args.initial_centres is None:
        z = inducing_kmeans(x, args.inducing, args.init_seed)
    else:
        with np.load(args.initial_centres, allow_pickle=False) as initial:
            if str(initial["design_hash"]) != design_hash or int(initial["seed"]) != args.init_seed:
                raise ValueError("Initial centres belong to a different training design or seed")
            z = initial["Z"]
        if z.shape != (args.inducing, x.shape[1]) or not np.isfinite(z).all():
            raise ValueError("Invalid saved initial inducing locations")
    np.savez(args.outdir/"initial_centres.npz", Z=z, design_hash=design_hash, seed=args.init_seed)
    history = []
    best = {"score": float("inf"), "step": -1, "parameters": None}
    gen = data["metadata"]["generator"]
    edges = np.linspace(0, (gen["mB_GeV"]-gen["mK_GeV"])**2, 13)
    # Everything TensorFlow-related, including model variables, is created
    # under the explicit device context. NumPy/k-means remain CPU operations.
    with tf.device(device):
        model = build_model(x, y, z, train_inducing=args.train_inducing)

        def checkpoint(step):
            prediction = predict_angle(model, xv, scaler)
            q2 = make_q2(data, validation, prediction["cos_mean"])
            qm = residual_metrics(q2-data["q2_true"][validation])
            cm = float(np.sqrt(np.mean((prediction["cos_mean"]-data["y_cos_bk"][validation].reshape(-1))**2)))
            loss = float(model.training_loss())
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite objective")
            row = {"step": int(step), "loss": loss, "validation_q2_rmse": qm["rmse"],
                   "validation_cosine_rmse": cm, "seconds": time.monotonic()-started}
            history.append(row)
            if qm["rmse"] < best["score"]:
                best.update(score=qm["rmse"], step=int(step), parameters=parameter_values(model))
            print(f"M={args.inducing} init={args.init_seed} step={step} loss={loss:.6f} val_q2={qm['rmse']:.6f}", flush=True)
            (args.outdir/"history.json").write_text(json.dumps(history, indent=2)+"\n")

        def callback(step, variables, values):
            if (step+1) % args.validate_every == 0:
                for variable, value in zip(variables, values):variable.assign(value)
                checkpoint(step+1)

        checkpoint(0)
        result = gpflow.optimizers.Scipy().minimize(
            model.training_loss, model.trainable_variables, method="L-BFGS-B", step_callback=callback,
            options={"maxiter": args.maxiter, "ftol": 1e-9, "maxls": 30})
        checkpoint(result.nit)
        # Record final hyperparameters/gradient separately from the selected
        # predictive checkpoint, so convergence can be audited or continued.
        final_parameters = parameter_values(model)
        jacobian = np.asarray(result.jac)
        gpflow.utilities.multiple_assign(model, best["parameters"])
        prediction = predict_angle(model, xv, scaler)
        q2 = make_q2(data, validation, prediction["cos_mean"])
        qm = residual_metrics(q2-data["q2_true"][validation])
        by_bin = binned_metrics(data["q2_true"][validation], q2, edges)
        selected_parameters = parameter_values(model)
    elapsed = time.monotonic()-started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024)
    report = {"scope": "Validation-only angular SGPR; no test evaluation",
              "config": config, "hardware": hardware, "input_sha256": data["sha256"],
              "training_rows": len(rows), "validation_rows": len(validation),
              "inducing_points": args.inducing, "best_validation_step": best["step"],
              "best_validation_q2_rmse": qm["rmse"],
              "selected_Z_rms_displacement": float(np.sqrt(np.mean((selected_parameters[".inducing_variable.Z"]-z)**2))),
              "final_Z_rms_displacement": float(np.sqrt(np.mean((final_parameters[".inducing_variable.Z"]-z)**2))),
              "validation_q2_metrics": qm, "validation_by_true_q2": by_bin,
              "validation_cosine_rmse": float(np.sqrt(np.mean((prediction["cos_mean"]-data["y_cos_bk"][validation].reshape(-1))**2))),
              "angular_calibration": angular_calibration(data["y_cos_bk"][validation], prediction),
              "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
              "optimizer_iterations": int(result.nit), "optimizer_final_loss": float(result.fun),
              "optimizer_final_gradient_inf_norm": float(np.max(np.abs(jacobian))),
              "seconds": elapsed, "peak_process_RSS_MiB": peak,
              "selected_lengthscales": selected_parameters[".kernel.kernels[0].lengthscales"].tolist(),
              "selected_noise_variance_standardized": float(selected_parameters[".likelihood.variance"])}
    info = {"model_name": "kaon_roe", "versions": {"gpflow": gpflow.__version__, "tensorflow": tf.__version__},
            "best_validation_step": best["step"], "best_validation_q2_rmse": qm["rmse"],
            "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
            "training_rows": len(rows), "validation_rows": len(validation),
            "kernel": "Matern52 ARD + Linear + Constant", "train_inducing": args.train_inducing,
            "training_seed": args.train_seed, "initialization_seed": args.init_seed}
    save_model(args.outdir/"model", model, x, y, scaler, list(range(data["X"].shape[1])), data, rows, info)
    np.savez_compressed(args.outdir/"optimizer_final_parameters.npz", **final_parameters)
    np.savez_compressed(args.outdir/"validation_predictions.npz", q2=q2,
                        q2_true=data["q2_true"][validation], **prediction,
                        event_id=data["event_id"][validation], generator_seed=data["generator_seed"][validation])
    (args.outdir/"result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ["inducing_points", "best_validation_step", "best_validation_q2_rmse", "optimizer_success", "optimizer_iterations", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
