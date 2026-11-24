"""Check generator-level invariants and compare the signal spectrum with KNUNU."""
import argparse
import sys
from pathlib import Path

import numpy as np
import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
from validate_bpair import bin_integrals, chi_square


def check(path):
    columns = ["q2_true"] + [f"{obj}_{component}_TRUE"
                             for obj in ("B", "K", "nu", "nubar")
                             for component in ("E", "PX", "PY", "PZ")]
    arrays = ROOT.RDataFrame("DecayTree", str(path)).AsNumpy(columns)
    vectors = {obj: np.column_stack([arrays[f"{obj}_{c}_TRUE"] for c in ("E", "PX", "PY", "PZ")])
               for obj in ("B", "K", "nu", "nubar")}

    def mass2(p):
        return p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)

    q2 = arrays["q2_true"]
    if not q2.size or not np.isfinite(q2).all():
        raise ValueError("Expected a nonempty tree with finite q²")
    invisible = vectors["nu"] + vectors["nubar"]
    recoil = vectors["B"] - vectors["K"]
    np.testing.assert_allclose(invisible, recoil, rtol=0, atol=1e-7)
    np.testing.assert_allclose(mass2(invisible), q2, rtol=0, atol=1e-7)
    np.testing.assert_allclose(mass2(recoil), q2, rtol=0, atol=1e-7)
    mb, mk = (float(np.median(np.sqrt(mass2(vectors[name])))) for name in ("B", "K"))
    bins = np.linspace(0, (mb - mk) ** 2, 21)
    counts, _ = np.histogram(q2, bins)
    if counts.sum() != q2.size:
        raise ValueError("q² exceeds physical support")
    expected = bin_integrals(bins, mb, mk)
    shape = chi_square(counts, expected / expected.sum())
    print(f"{q2.size:,} events passed four-momentum and q² closure. "
          f"KNUNU shape χ²/ndf = {shape['chi2']:.2f}/{shape['ndf']}, p = {shape['p_value']:.3g}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    check(parser.parse_args().tree)
