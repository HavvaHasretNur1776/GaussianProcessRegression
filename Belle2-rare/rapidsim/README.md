# RapidSim signal validation

This first sample isolates the B⁺ → K⁺ νₑ ν̄ₑ decay. The [decay descriptor](BToKnunu.decay) sets the daughters. The [card](BToKnunu.config) selects the Belle II KNUNU matrix element through EvtGen, treats both neutrinos as invisible, and saves their truth four-vectors. It uses 4π acceptance, no track smearing, and a parent generation range of 0–1 GeV in transverse momentum and −1 to 1 in pseudorapidity. These parent settings are specific to the signal check. The full-pair sample uses Υ(4S) production at fixed CM energy.

From the top of this repository, with Git, Bash and Pixi installed:

```bash
bash rapidsim/setup.sh
bash rapidsim/generate.sh 100000 runs/rapidsim_signal
pixi run --manifest-path vendor/RapidSim/pixi.toml --frozen -- python rapidsim/validate_signal.py runs/rapidsim_signal/BToKnunu_tree.root
```

Setup creates a project-local upstream checkout at `vendor/RapidSim`, pins the source revision and uses the included Pixi lockfile. The first patch makes the optional external engines switchable. The second registers the KNUNU model supplied in [simulation/models](../simulation/models). For the signal-only run we switch off PHOTOS, Pythia and TAUOLA. The independent pair generator uses Pythia for generic tag-side partonic decays.

The generation command takes an event count and a fresh output folder. It writes the ROOT tree, histograms and EvtGen DEC card there. To rebuild an already prepared checkout use `(cd vendor/RapidSim && pixi run --frozen build)`. To run the closure check directly on the archived sample, pass `data/reference/BToKnunu_tree.root` to the validator.

The card and model registration were reconstructed from the stored sample and source. A fresh 100,000-event run reproduced the stored event-by-event q² to within 4.0 × 10⁻⁷ GeV². The validator tests four-momentum conservation, both invariant-mass definitions of q², physical support, and the normalized KNUNU spectrum. Use the [B-pair generator](../simulation/BPairGenerator.cc) to generate the companion decay for GP training.

[Upstream RapidSim at the pinned revision](https://github.com/lhcb/RapidSim/tree/5fcfc328f0924f8ecb7a9d0129aed7c415a265a6) · [RapidSim license](RapidSim-LICENSE.md) · [KNUNU source credits](../simulation/NOTICE.md)
