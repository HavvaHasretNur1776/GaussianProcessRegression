#!/usr/bin/env python3
"""Small physical and information-flow checks for the response layer."""
import json
from pathlib import Path
import unittest

import numpy as np

from response import (boost_z, event_noise, event_split, mass2,
                          observable_features, reconstruct)


class ResponseChecks(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads(Path(__file__).with_name("response.json").read_text())
        self.scenario = self.settings["scenarios"][2]
        # Correctly assigned K, a charged ROE object, a photon, K_L and neutrino.
        # This tests the response mechanics; event physics comes from EvtGen.
        self.pdg = np.array([321, -211, 22, 130, 12])
        self.charge = np.array([3, -3, 0, 0, 0])
        momentum = np.array([[1., 0.2, 0.3], [-0.4, 0.1, -0.2],
                             [0.1, 0.3, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.3]])
        masses = np.array([0.493677, 0.13957039, 0., 0.497611, 0.])
        self.p4 = np.column_stack([np.sqrt(np.sum(momentum**2, axis=1)+masses**2), momentum])
        self.noise = event_noise(1776, 42, self.settings["response_seed"], len(self.p4))
        self.noise["keep"][:] = 0.01

    def features(self, p4):
        observed, retained, track, charges = reconstruct(
            p4, self.pdg, self.charge, 0, self.noise, self.scenario, self.settings, 0.493677)
        return observable_features(observed, retained, track, charges, 0, 10.58, 5.27933, 0.493677), observed, retained

    def test_lorentz_roundtrip_and_invariant_mass(self):
        lab = boost_z(self.p4, self.settings["longitudinal_beta"])
        np.testing.assert_allclose(boost_z(lab, -self.settings["longitudinal_beta"]), self.p4, atol=1e-14)
        np.testing.assert_allclose(mass2(lab), mass2(self.p4), atol=1e-14)

    def test_unobserved_particle_truth_cannot_change_features(self):
        first, _, _ = self.features(self.p4)
        modified = self.p4.copy()
        modified[3:, :] *= 3.0  # change only unobserved K_L/neutrino momenta
        second, _, _ = self.features(modified)
        np.testing.assert_array_equal(first["X"], second["X"])

    def test_measured_objects_are_on_shell(self):
        _, observed, retained = self.features(self.p4)
        expected = np.array([0.493677**2 if i == 0 else 0. if self.pdg[i] == 22
                             else self.settings["roe_track_mass_GeV"]**2 for i in retained])
        np.testing.assert_allclose(mass2(observed), expected, atol=1e-13)
        self.assertNotIn(3, retained)
        self.assertNotIn(4, retained)

    def test_response_and_splits_do_not_depend_on_processing_order(self):
        forward = {i: event_noise(1776, i, 123, 5) for i in [0, 1, 100]}
        backward = {i: event_noise(1776, i, 123, 5) for i in [100, 1, 0]}
        for i in forward:
            for key in forward[i]:
                np.testing.assert_array_equal(forward[i][key], backward[i][key])
        split = np.array([event_split(1776, i, 717) for i in range(10000)])
        for category, target in enumerate([0.6, 0.2, 0.2]):
            self.assertLess(abs(np.mean(split == category)-target), 0.02)


if __name__ == "__main__":
    unittest.main()
