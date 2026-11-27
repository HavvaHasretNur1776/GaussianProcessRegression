# Reconstructing q² in B⁺ → K⁺νν̄ with Gaussian process regression

In B⁺ → K⁺νν̄, both neutrinos escape detection. Belle II’s inclusive-tag analysis estimates q², the invariant mass squared of the neutrino pair, from the kaon energy by treating the signal B as stationary in the collision centre-of-mass frame. The B has a small momentum in this frame. As a result, even two perfectly measured kaons with the same energy can correspond to different true values of q². [Belle II, Eq. (2), Phys. Rev. D 109, 112006 (2024)](https://doi.org/10.1103/PhysRevD.109.112006)

Keeping the B momentum in the calculation introduces the term $2 p_B p_K \cos\theta_{BK}$. The beam energy fixes the B momentum magnitude, so the missing information is the angle between the B and the kaon. The two B mesons recoil against each other, which means the visible decay products of the companion B carry information about the signal B direction. We train a sparse Gaussian process (GP) on 11 reconstructed quantities to predict $\cos\theta_{BK}$ and use that prediction to calculate q² for each event.

In a differential measurement, limited q² resolution moves events between neighbouring bins and can wash out differences between predicted spectra. A more precise event-by-event estimate could help preserve those differences. On 3,513 signal-only toy validation events, the GP reduced the q² root-mean-square error by **16.36%** relative to the Belle II estimator applied to the same events.

## How we reconstruct q²

With the stationary-B approximation, Belle II’s inclusive-tag estimator is

```math
q^2_{\mathrm{ITA}} = \frac{s}{4} + m_K^2 - \sqrt{s} E_K
```

Using the B mass and the predicted direction cosine gives

```math
q^2_{\mathrm{GP}} = m_B^2 + m_K^2 - \sqrt{s} E_K + 2 p_B p_K \hat{c}_{BK}
```

All energies and momenta are in the collision centre-of-mass frame. $E_K$ and $p_K$ are the reconstructed kaon energy and momentum magnitude, $p_B$ is the B momentum magnitude, and $\hat{c}_{BK}$ is the GP prediction of $\cos\theta_{BK}$.

## Model configuration

- **Target:** the cosine between the **true B direction and true kaon direction** in simulated events.
- **Model:** GPflow sparse Gaussian process regression with a Matérn-5/2 kernel, linear and constant kernel terms, and a linear mean.
- **Inducing points:** 1,000 by default. The fit starts from k-means centres and learns their positions. Set `M` in the notebook to change the count.
- **Training:** 5,000 events sampled from a pool of 10,842. We select the fit with the lowest q² RMSE on 3,513 validation events and reserve 3,593 events for testing.

### Input features

The GP receives 11 reconstructed quantities. The *rest of the event* (ROE) consists of the visible particles left after removing the signal kaon.

| Source | Inputs |
|---|---|
| Kaon (2) | Energy and polar-angle cosine relative to the beam |
| ROE momentum (4) | Total visible energy, magnitude of the summed momentum, cosine of the angle to the kaon, and polar-angle cosine relative to the beam |
| ROE composition (2) | Net charge and scalar sum of individual momentum magnitudes |
| Missing energy and momentum (3) | Missing energy, missing momentum magnitude, and polar-angle cosine of the missing momentum |

## Workflow

**To run the fit on CERN SWAN**, open the training notebook in a GPU session and run the cells in order. It reads the bundled feature table, fits the GP, and plots the comparison with the Belle II estimator. [Training notebook](SWAN_trainable11.ipynb) · [11-input dataset](data/eleven.npz)

**To reproduce the study from event generation**, follow the stages below.

| Stage | What it does | Files |
|---|---|---|
| Check the signal decay | Generate 100,000 signal-only decays with RapidSim and validate the KNUNU q² spectrum. This is an independent check. | [RapidSim setup and cards](rapidsim/README.md) |
| Generate the training sample | Generate 20,000 charged B pairs at 10.58 GeV. EvtGen forces the signal decay and generates the companion B decay. Pythia handles modes that require hadronisation. | [Pair generator](simulation/BPairGenerator.cc) · [Decay table](simulation/signal.dec) |
| Reconstruct the events | Apply acceptance, efficiency, thresholds and smearing to tracks and photons. The signal kaon survives in 17,948 events. Export the 11 inputs and the simulation labels. | [Response settings](simulation/response.json) · [Feature exporter](simulation/export_features.py) |
| Train and evaluate | Fit the angular GP and compare the two q² estimates event by event. | [CPU/GPU fit](studies/inducing_scan/fit_validation.py) · [Paired evaluation](evaluate.py) |

The optional kinematic notebook illustrates why the B–kaon angle matters, even with a perfectly measured kaon. You can skip it when generating events or training the GP. [Kinematic illustration](notebooks/01_kinematic_baseline.ipynb)

<details>
<summary>Commands for event generation and reconstruction</summary>

Run from the repository root with Git, Bash and Pixi. To start at reconstruction, use `data/reference/bpair_20k.root` as the exporter input.

```bash
bash rapidsim/setup.sh
bash rapidsim/generate.sh 100000 runs/rapidsim_signal
sim() { pixi run --manifest-path vendor/RapidSim/pixi.toml --frozen -- "$@"; }
sim python rapidsim/validate_signal.py runs/rapidsim_signal/BToKnunu_tree.root

sim cmake -S simulation -B build/bpair -GNinja
sim cmake --build build/bpair
sim build/bpair/BPairGenerator.exe --events 20000 --seed 1776 --output runs/bpair.root
sim python simulation/validate_bpair.py --input runs/bpair.root --outdir runs/bpair_validation
sim python simulation/export_features.py runs/bpair.root runs/regenerated.npz
sim python simulation/test_response.py
```

</details>

<details>
<summary>Commands for training on a Linux CPU</summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
export TF_USE_LEGACY_KERAS=1 OPENBLAS_NUM_THREADS=1
python studies/inducing_scan/fit_validation.py --input data/eleven.npz --outdir runs/cpu_M1000 --inducing 1000 --train-inducing --device cpu
python evaluate.py runs/cpu_M1000
```

For a freshly exported dataset, pass `--input runs/regenerated.npz` to both the fit and the evaluation. Each fit needs a fresh output directory. It saves its own checkpoint and event-level predictions. The original SWAN run is archived here as a figure and summary metrics.

</details>

## Results

The table compares the two estimators on the same events used for validation during training. The residual is reconstructed q² minus true q².

| Validation metric (GeV²) | Belle II estimator on toy | GP on toy |
|---|---:|---:|
| Root-mean-square error | 0.7255 | 0.6068 |
| Central 68% half-width | 0.7719 | 0.5966 |
| Mean residual | +0.0340 | +0.0119 |

The central 68% of the residual distribution narrowed by **22.71%**. For the **16.36% reduction in RMSE**, the paired 95% bootstrap interval is **14.76–18.02%** across 2,500 resamples of the validation events.

![Reconstructed and true q² spectra on the left, and event-by-event residuals for both estimators on the right](results/validation.png)

These figures describe a signal-only toy with a simplified detector response. The generator identifies the signal kaon. Assessing performance in Belle II data requires realistic kaon selection, physics backgrounds and beam-background occupancy.

[Development studies and model comparisons](results/development/README.md) · [Generator checks](results/bpair_validation_20k.json) · [Simulation credits](simulation/NOTICE.md) · [Related LHCb momentum reconstruction](https://github.com/HavvaHasretNur1776/GaussianProcessRegression/tree/main/LHCb-semileptonic)
