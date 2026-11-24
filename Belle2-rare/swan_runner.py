"""Launch each training stage in a fresh, logged Python process from Jupyter."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "eleven.npz"
DATA_SHA256 = "713af18144aa4a50bda7dd81633e2104ae16c454dc617a25ed2cda6f662bf9a5"

def create_session():
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError(f"Bundled dataset checksum mismatch: {DATA}")
    parent = ROOT / "runs"
    parent.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=parent))

def run_logged(command, log, env):
    command = [str(arg) for arg in command]
    print("Writing", log, flush=True)
    with log.open("a", encoding="utf-8") as stream:
        with subprocess.Popen(command, cwd=ROOT, env=env, text=True, errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1) as child:
            try:
                for line in child.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                status = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                raise
    if status:
        raise RuntimeError(f"{command[0]} exited with {status}. See {log}")

@lru_cache(maxsize=1)
def private_workspace():
    return Path(tempfile.mkdtemp(prefix="swan-gpflow-"))

def training_env(session, isolated=True, existing_workspace=None):
    env = os.environ.copy()
    env.update(TF_USE_LEGACY_KERAS="1", TF_CPP_MIN_LOG_LEVEL="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    if not isolated:
        return Path(sys.executable), env

    if existing_workspace is None:
        work = private_workspace()
    else:
        work = Path(existing_workspace).expanduser().absolute()
        if not (work / "train/bin/python").is_file():
            raise FileNotFoundError(f"Training Python missing from {work}. Set EXISTING_WORKSPACE = None to install.")

    log = session / "setup.log"
    uv = shutil.which("uv")
    if uv is None:
        target = work / "bootstrap"
        uv = target / "bin" / "uv"
        if not uv.is_file():
            # Use the kernel's pip only to obtain the standalone uv executable.
            run_logged([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                        "--ignore-installed", "--no-deps", "--target", target, "uv==0.8.22"], log, env)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "UV_PYTHON"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", UV_CACHE_DIR=str(work / "cache"),
               UV_PYTHON_INSTALL_DIR=str(work / "interpreters"), UV_LINK_MODE="hardlink")
    venv = work / "train"
    python = venv / "bin" / "python"
    if not python.is_file():
        run_logged([uv, "venv", "--python", "3.12", venv], log, env)
    run_logged([python, "-c", "import sys; print(sys.executable, sys.version); "
                "assert sys.version_info[:2] == (3, 12)"], log, env)
    run_logged([uv, "pip", "install", "--python", python, "-r",
                ROOT / "requirements-gpu.txt"], log, env)
    env["PATH"] = str(venv / "bin") + os.pathsep + env["PATH"]
    env["VIRTUAL_ENV"] = str(venv)
    packages = Path(subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        env=env, text=True).strip())
    libs = sorted((packages / "nvidia").glob("*/lib"))
    # The TF extension has DT_RUNPATH. An inherited LCG path can otherwise
    # supply a different libtensorflow_cc.so.2 or libtensorflow_framework.so.2.
    paths = [str(packages / "tensorflow"), *(str(p) for p in libs)]
    paths.extend(env.get("LD_LIBRARY_PATH", "").split(os.pathsep))
    env.pop("LD_PRELOAD", None)  # Keep the parent SWAN kernel environment intact.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in paths if p)
    nvcc_bin = packages / "nvidia" / "cuda_nvcc" / "bin"
    if nvcc_bin.is_dir():
        env["PATH"] = str(nvcc_bin) + os.pathsep + env["PATH"]
    return python, env

def preflight(python, env, session):
    smi = shutil.which("nvidia-smi", path=env["PATH"])
    if smi:
        run_logged([smi, "--query-gpu=name,driver_version,memory.total",
                    "--format=csv"], session / "gpu.log", env)
    if "VIRTUAL_ENV" in env and shutil.which("ldd"):
        # Capture resolved paths before the import, even if the import fails.
        site = Path(subprocess.check_output(
            [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            env=env, text=True).strip())
        extension = site / "tensorflow/python/_pywrap_tensorflow_internal.so"
        if extension.is_file():
            run_logged(["ldd", extension], session / "gpu.log", env)
    run_logged([python, ROOT / "gpu_check.py"], session / "gpu.log", env)

def fit(python, env, out, inducing):
    if type(inducing) is not int or not 1 <= inducing <= 5000:
        raise ValueError("M must be an integer between 1 and the 5,000 training events.")
    if out.exists():
        raise FileExistsError(f"Fit already exists at {out}. Start a new session for another fit.")
    run_logged([python, ROOT / "studies/inducing_scan/fit_validation.py", "--input", DATA,
                "--outdir", out, "--device", "gpu", "--inducing", inducing, "--train-inducing",
                "--threads", 4, "--train-seed", 120, "--init-seed", 120, "--train-cap", 5000,
                "--maxiter", 1000, "--validate-every", 25], out.with_suffix(".log"), env)
