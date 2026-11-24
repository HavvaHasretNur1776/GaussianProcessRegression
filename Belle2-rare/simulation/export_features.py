"""Export the nominal eleven reconstructed inputs and generator-level labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import ROOT
from response import FEATURE_NAMES, event_noise, event_split, observable_features, reconstruct

NAMES = [name for name in FEATURE_NAMES if name not in
         {"roe_n_tracks", "roe_n_photons", "event_missing_mass2", "event_R2"}]

def export(source_path, output):
    if output.exists():
        raise FileExistsError(output)
    src = ROOT.TFile.Open(str(source_path))
    if not src or src.IsZombie() or not src.Get("Events"):
        raise ValueError("Expected the full BPairGenerator Events tree")
    gen = json.loads(str(src.Get("ConfigJSON").GetString()))
    response = json.loads(Path(__file__).with_name("response.json").read_text())
    scenario = next(s for s in response["scenarios"] if s["name"] == "toy_nominal")
    records = {k: [] for k in ("X", "event_id", "split", "q2_true", "y_cos_bk",
                               "q2_energy_belle", "q2_energy_mass", "q2_roe_direction")}
    columns = [FEATURE_NAMES.index(name) for name in NAMES]
    for ev in src.Get("Events"):
        terminal = np.flatnonzero(np.asarray(ev.n_daughters) == 0)
        kpos = np.flatnonzero(terminal == ev.kaon_index).item()
        p = np.column_stack([np.asarray(getattr(ev, k)) for k in ("E_cm", "px_cm", "py_cm", "pz_cm")])
        event = int(ev.event)
        noise = event_noise(gen["seed"], event, response["response_seed"], len(terminal))
        measured = reconstruct(p[terminal], np.asarray(ev.pdg)[terminal], np.asarray(ev.charge3)[terminal],
                               kpos, noise, scenario, response, gen["mK_GeV"])
        obs = observable_features(*measured, kpos, gen["sqrt_s_GeV"], gen["mB_GeV"], gen["mK_GeV"])
        if obs is None:
            continue
        b, k = p[ev.signal_index, 1:], p[ev.kaon_index, 1:]
        records["y_cos_bk"].append(np.dot(b, k)/(np.linalg.norm(b)*np.linalg.norm(k)))
        records["X"].append(obs["X"][columns])
        records["event_id"].append(event)
        records["split"].append(event_split(gen["seed"], event, response["split_seed"]))
        records["q2_true"].append(float(ev.q2_true))
        for key in ("q2_energy_belle", "q2_energy_mass", "q2_roe_direction"):
            records[key].append(obs[key])
    src.Close()
    meta = dict(generator=gen, response=response, scenario=scenario["name"],
                angle_target="True B and true K CM directions")
    np.savez_compressed(output, **records, generator_seed=np.full(len(records["X"]),gen["seed"]),
                        metadata_json=json.dumps(meta), feature_names=NAMES, kaon_only_columns=[0,1])
    print(f"Exported {len(records['X'])} selected kaons to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
