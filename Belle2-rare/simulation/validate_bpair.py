#!/usr/bin/env python3
"""Validate B-pair event records; results describe generation, not a detector."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import ROOT


def rate(q2, mb, mk):
    """Independent normalised-shape input: lambda^(3/2) times f+ squared."""
    tp, tm = (mb+mk)**2, (mb-mk)**2
    t0 = tp*(1-np.sqrt(1-tm/tp))
    z = (np.sqrt(tp-q2)-np.sqrt(tp-t0))/(np.sqrt(tp-q2)+np.sqrt(tp-t0))
    a0, a1, a2 = 0.432, -0.664, -1.2
    fp = (a0+a1*z+a2*z**2+(-a1+2*a2)*z**3/3)/(1-q2/(mb+0.046)**2)
    return np.maximum((tp-q2)*(tm-q2), 0)**1.5 * fp**2


def bin_integrals(edges, mb, mk):
    nodes, weights = np.polynomial.legendre.leggauss(64)
    half = np.diff(edges)/2
    grid = (edges[1:]+edges[:-1])[:, None]/2 + half[:, None]*nodes
    return half*(rate(grid, mb, mk) @ weights)


def chi_square(counts, probabilities):
    expected = np.asarray(probabilities)*np.sum(counts)
    expected *= np.sum(counts)/np.sum(expected)
    chi2 = float(np.sum((counts-expected)**2/expected))
    ndf = len(counts)-1
    return {"chi2": chi2, "ndf": ndf,
            "p_value": float(ROOT.TMath.Prob(chi2, ndf)),
            "minimum_expected_count": float(expected.min())}


def plot_summary(q2, cosb, nch, enu, mb, mk, outdir):
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("bpair_summary", "B-pair generator checks; no detector", 1100, 850)
    canvas.Divide(2, 2)
    keep = []
    for pad, name, values, bins, low, high, label in [
        (1, "q2", q2, 40, 0, (mb-mk)**2, ";Signal q^{2} [GeV^{2}];Events / bin"),
        (2, "angle", cosb, 30, -1, 1, ";cos #theta_{B} in CM;Events / bin"),
        (3, "ncharged", nch, 22, -0.5, 21.5, ";Tag charged final particles;Events / bin"),
        (4, "neutrinos", enu, 40, 0, 5.5, ";Tag neutrino energy sum in CM [GeV];Events / bin"),
    ]:
        canvas.cd(pad)
        ROOT.gPad.SetLeftMargin(0.15)
        ROOT.gPad.SetRightMargin(0.04)
        ROOT.gPad.SetBottomMargin(0.14)
        ROOT.gPad.SetTopMargin(0.08)
        if pad == 4:
            ROOT.gPad.SetLogy(True)
        h = ROOT.TH1D(name, label, bins, low, high)
        h.SetDirectory(0)
        for v in values:
            h.Fill(float(v))
        h.SetLineColor(ROOT.kBlack)
        h.SetMarkerStyle(20)
        h.SetMarkerSize(0.45)
        h.SetMaximum(h.GetMaximum()*1.18)
        if pad == 4:
            h.SetMinimum(0.5)
            h.SetMaximum(h.GetMaximum()*1.6)
        h.Draw("E1" if pad < 3 else "HIST")
        keep.append(h)
        if pad in [1, 2]:
            expected = ROOT.TH1D(name+"_expected", "", bins, low, high)
            expected.SetDirectory(0)
            edges = np.linspace(low, high, bins+1)
            prob = bin_integrals(edges, mb, mk) if pad == 1 else np.diff(0.75*(edges-edges**3/3))
            prob /= prob.sum()
            for i, v in enumerate(prob, 1):
                expected.SetBinContent(i, len(values)*float(v))
            expected.SetLineColor(ROOT.kBlue+1)
            expected.SetLineWidth(2)
            expected.Draw("HIST SAME")
            legend = ROOT.TLegend(0.53, 0.73, 0.94, 0.89)
            legend.SetBorderSize(0)
            legend.SetTextSize(0.032)
            legend.AddEntry(h, "Generated events", "lep")
            legend.AddEntry(expected, "KNUNU expectation" if pad == 1 else "sin^{2}#theta expectation", "l")
            legend.Draw()
            keep.extend([expected, legend])
    for ext in ["png", "pdf"]:
        canvas.SaveAs(str(outdir/("bpair_validation."+ext)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="bpair.root")
    parser.add_argument("--outdir", default="bpair_validation")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    f = ROOT.TFile.Open(args.input)
    if not f or f.IsZombie():
        raise ValueError("Cannot open input ROOT file")
    tree = f.Get("Events")
    config = json.loads(str(f.Get("ConfigJSON").GetString()))
    n = tree.GetEntries()
    if n < 10000:
        raise ValueError("Use at least 10,000 events for distribution checks")
    checks = {k: True for k in ["event_count", "finite", "ancestry", "daughter_counts",
              "charged_pair", "signal_topology", "charge_conservation", "fully_hadronised"]}
    checks["event_count"] = n == config["events"]
    q2, cosb, phi, nch, enu = [], [], [], [], []
    models, tagmodels, leaves, flavour = Counter(), Counter(), Counter(), Counter()
    sizes = []
    max_vertex, max_leaf, max_q2, max_beam_energy = 0., 0., 0., 0.
    max_nonlegacy_vertex = 0.
    closure_exceptions = []
    legacy_three_pion = {"CB3PI-P00", "CB3PI-MPP", "BTO3PI_CP"}
    tag_neutrino_events, tau_events = 0, 0
    for index, ev in enumerate(tree):
        pdg = np.array(ev.pdg, dtype=int)
        mother = np.array(ev.mother, dtype=int)
        side = np.array(ev.side, dtype=int)
        nd = np.array(ev.n_daughters, dtype=int)
        charge = np.array(ev.charge3, dtype=int)
        p = np.stack([np.array(getattr(ev, k)) for k in ["E_cm", "px_cm", "py_cm", "pz_cm"]], axis=1)
        position = np.stack([np.array(getattr(ev, k)) for k in ["ct_mm", "x_mm", "y_mm", "z_mm"]], axis=1)
        signal, tag, kaon = int(ev.signal_index), int(ev.tag_index), int(ev.kaon_index)
        names = [str(name) for name in ev.model]
        checks["finite"] &= bool(np.isfinite(p).all() and np.isfinite(position).all()
                                 and np.isfinite(ev.q2_true) and np.isfinite(ev.cos_theta_B))
        checks["ancestry"] &= bool(ev.event == index and mother[0] == -1 and side[0] == 0
                                   and np.all((mother[1:] >= 0) & (mother[1:] < np.arange(1, len(pdg)))))
        checks["daughter_counts"] &= bool(np.array_equal(np.bincount(mother[1:], minlength=len(pdg)), nd))
        checks["charged_pair"] &= bool(pdg[0] == 300553 and pdg[signal] == 521 and pdg[tag] == -521
                                       and mother[signal] == 0 and mother[tag] == 0 and nd[0] == 2)
        is_neutrino = np.isin(np.abs(pdg), [12, 14, 16])
        sig_nu = (mother == signal) & is_neutrino
        checks["signal_topology"] &= bool(nd[signal] == 3 and pdg[kaon] == 321 and mother[kaon] == signal
                                         and np.sum(sig_nu) == 2 and np.sum(pdg[sig_nu]) == 0)
        for j in range(1, len(pdg)):
            expected_side = 1 if j == signal else 2 if j == tag else side[mother[j]]
            checks["ancestry"] &= side[j] == expected_side
        terminal = nd == 0
        tag_terminal = terminal & (side == 2)
        parton = ((np.abs(pdg) >= 1) & (np.abs(pdg) <= 8)) | (np.abs(pdg) == 21)
        diquark = ((np.abs(pdg) >= 1000) & (np.abs(pdg) <= 9999)
                   & ((np.abs(pdg)//10) % 10 == 0))
        checks["fully_hadronised"] &= not bool(np.any(terminal & (parton | diquark)))
        for j in np.flatnonzero(nd):
            children = mother == j
            error = float(np.max(np.abs(p[j]-p[children].sum(axis=0))))
            max_vertex = max(max_vertex, error)
            if names[j] not in legacy_three_pion:
                max_nonlegacy_vertex = max(max_nonlegacy_vertex, error)
            if error > 1e-7:
                closure_exceptions.append({"event": index, "pdg": int(pdg[j]),
                                           "model": names[j], "error_GeV": error})
            checks["charge_conservation"] &= charge[j] == charge[children].sum()
        max_leaf = max(max_leaf, float(np.max(np.abs(p[0]-p[terminal].sum(axis=0)))))
        recoil = p[signal]-p[kaon]
        pair = p[sig_nu].sum(axis=0)
        recoil_q2 = recoil[0]**2-np.dot(recoil[1:], recoil[1:])
        pair_q2 = pair[0]**2-np.dot(pair[1:], pair[1:])
        max_q2 = max(max_q2, abs(recoil_q2-ev.q2_true), abs(pair_q2-ev.q2_true))
        max_beam_energy = max(max_beam_energy, abs(p[signal, 0]-config["sqrt_s_GeV"]/2),
                              abs(p[tag, 0]-config["sqrt_s_GeV"]/2))
        q2.append(float(ev.q2_true)); cosb.append(float(ev.cos_theta_B))
        phi.append(float(np.arctan2(p[signal, 2], p[signal, 1])))
        nch.append(int(np.sum(tag_terminal & (charge != 0))))
        enu.append(float(p[tag_terminal & is_neutrino, 0].sum()))
        tag_neutrino_events += bool(np.any(tag_terminal & is_neutrino))
        tau_events += bool(np.any((side == 2) & (np.abs(pdg) == 15)))
        models.update(name for name in names if name)
        tagmodels[names[tag]] += 1
        leaves.update(int(v) for v in pdg[tag_terminal])
        flavour[int(np.abs(pdg[sig_nu][0]))] += 1
        sizes.append(int(np.sum(terminal)))
    q2, cosb, phi = map(np.asarray, [q2, cosb, phi])
    mb, mk = config["mB_GeV"], config["mK_GeV"]
    qmax = (mb-mk)**2
    edges = np.linspace(0, qmax, min(40, max(5, n//2000))+1)
    qprob = bin_integrals(edges, mb, mk)
    qtest = chi_square(np.histogram(q2, edges)[0], qprob/qprob.sum())
    cedges = np.linspace(-1, 1, 21)
    ctest = chi_square(np.histogram(cosb, cedges)[0], np.diff(0.75*(cedges-cedges**3/3)))
    phitest = chi_square(np.histogram(phi, np.linspace(-np.pi, np.pi, 21))[0], np.full(20, 1/20))
    ftest = chi_square(np.array([flavour[v] for v in [12, 14, 16]]), np.full(3, 1/3))
    # Legacy three-pion helper uses mB=5.2794 versus 5.27933 in this table.
    # Report raw residuals; allow 0.1 MeV globally, 0.1 keV for all other models.
    checks.update({"vertex_four_momentum_within_0p1_MeV": max_vertex < 1e-4,
                   "leaf_four_momentum_within_0p1_MeV": max_leaf < 1e-4,
                   "nonlegacy_vertex_four_momentum": max_nonlegacy_vertex < 1e-7,
                   "q2_identity": max_q2 < 1e-8,
                   "beam_constrained_B_energy": max_beam_energy < 1e-8,
                   "physical_q2_range": bool(q2.min() >= 0 and q2.max() <= qmax),
                   "q2_shape": qtest["p_value"] > 0.001,
                   "sin2_production_angle": ctest["p_value"] > 0.001,
                   "uniform_production_phi": phitest["p_value"] > 0.001,
                   "equal_signal_neutrino_flavours": ftest["p_value"] > 0.001,
                   "pythia_used": models["PYTHIA"] > 0,
                   "tag_neutrinos_present": tag_neutrino_events > 0,
                   "tag_tau_decays_present": tau_events > 0})
    report = {"config": config, "events": n, "passed": bool(all(checks.values())),
              "checks": {key: bool(value) for key, value in checks.items()},
              "q2_shape_test": qtest, "production_angle_test": ctest,
              "production_phi_test": phitest, "neutrino_flavour_test": ftest,
              "max_vertex_error_GeV": max_vertex, "max_leaf_error_GeV": max_leaf,
              "max_nonlegacy_vertex_error_GeV": max_nonlegacy_vertex,
              "vertices_exceeding_1e_minus_7_GeV": closure_exceptions,
              "max_q2_error_GeV2": max_q2, "max_B_energy_error_GeV": max_beam_energy,
              "tag_neutrino_event_fraction": tag_neutrino_events/n,
              "tag_tau_event_fraction": tau_events/n,
              "mean_tag_charged_final_particles": float(np.mean(nch)),
              "final_particle_count_range": [min(sizes), max(sizes)],
              "model_counts": dict(models), "primary_tag_model_counts": dict(tagmodels),
              "tag_final_pdg_counts": dict(leaves),
              "input_sha256": {key: hashlib.sha256(str(f.Get(key).GetString()).encode()).hexdigest()
                               for key in ["GenericDecayTable", "ParticleTable", "ForcedDecayTable"]},
              "scope": "Historical EvtGen generic table; fixed-energy CM generator study, no detector response"}
    (outdir/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    plot_summary(q2, cosb, nch, enu, mb, mk, outdir)
    if not report["passed"]:
        raise SystemExit("B-pair validation failed; inspect validation.json")


if __name__ == "__main__":
    main()
