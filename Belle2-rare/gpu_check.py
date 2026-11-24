"""Check the actual training interpreter, GPU, FP64 kernels and GPflow gradient."""
from importlib import import_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "studies" / "inducing_scan"))
sys.path.insert(0, str(ROOT / "studies" / "gpr"))

def main():
    from device import configure_device
    device, report = configure_device("gpu", 4)
    import numpy as np
    import tensorflow as tf
    from gpr_common import build_model

    gpu = tf.config.get_visible_devices("GPU")[0]
    report["gpu_details"] = tf.config.experimental.get_device_details(gpu)
    report["tf_build"] = tf.sysconfig.get_build_info()
    report["python_executable"] = sys.executable
    print(json.dumps(report, indent=2, default=str), flush=True)
    report["versions"] = {}
    for name in ("gpflow", "tensorflow_probability", "numpy", "scipy"):
        module = import_module(name)
        report["versions"][name] = getattr(module, "__version__", "unreported")
    from swan_runner import DATA
    with np.load(DATA, allow_pickle=False) as data:
        features = data["feature_names"].tolist()
        dimension = data["X"].shape[1]
    if dimension != 11 or len(features) != dimension:
        raise ValueError("Expected eleven input columns in the bundled dataset")
    report["feature_names"] = features
    rng = np.random.default_rng(120)
    x = rng.normal(size=(256, dimension))
    y = np.sin(x[:, [0]])
    with tf.device(device):
        a = tf.constant(x, dtype=tf.float64)
        cov = tf.matmul(a, a, transpose_a=True) + tf.eye(dimension, dtype=tf.float64)
        chol = tf.linalg.cholesky(cov)
        for op in (cov, chol):
            if "GPU:" not in op.device.upper():
                raise RuntimeError(f"Expected FP64 GPU tensor, got {op.device}")
        np.testing.assert_allclose((chol @ tf.transpose(chol)).numpy(), cov.numpy(), rtol=1e-10, atol=1e-10)
        model = build_model(x, y, x[:32].copy(), train_inducing=True)
        with tf.GradientTape() as tape:
            loss = model.training_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        if not np.isfinite(loss.numpy()) or any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
            raise RuntimeError("SGPR loss or gradients failed the finite-value check")
        report["sgpr_loss"] = float(loss.numpy())
        report["cholesky_device"] = chol.device
        report["loss_device"] = loss.device
    print(json.dumps(report, indent=2, default=str), flush=True)

if __name__ == "__main__":
    main()
