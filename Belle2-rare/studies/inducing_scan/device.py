"""Explicit TensorFlow device selection and double-precision placement check."""
import platform


def configure_device(requested="cpu", threads=4):
    import tensorflow as tf
    physical = tf.config.list_physical_devices("GPU")
    if requested == "gpu" and not physical:
        raise RuntimeError("GPU requested but TensorFlow sees no GPU. Run on an allocated GPU worker with a compatible TensorFlow/CUDA environment.")
    use_gpu = requested == "gpu" or (requested == "auto" and bool(physical))
    if use_gpu:
        tf.config.set_visible_devices(physical[0], "GPU")
        tf.config.experimental.set_memory_growth(physical[0], True)
    else:
        tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    name = "/GPU:0" if use_gpu else "/CPU:0"
    # Test actual float64 placement. Do not infer acceleration from CUDA build metadata.
    with tf.device(name):
        matrix = tf.constant([[2., .1], [.1, 1.]], dtype=tf.float64)
        factor = tf.linalg.cholesky(matrix)
        factor.numpy()
    if use_gpu and "GPU:" not in factor.device.upper():
        raise RuntimeError("Float64 Cholesky silently fell back to CPU")
    return name, {"requested": requested, "chosen": name, "visible_physical_gpus": [str(x) for x in physical],
                  "float64_cholesky_device": factor.device, "tensorflow": tf.__version__,
                  "python": platform.python_version(), "gpu_execution_verified": use_gpu}
