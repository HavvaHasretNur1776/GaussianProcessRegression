"""Nominal object response and observable functions from the original toy. Units are GeV."""
import hashlib
import numpy as np

FEATURE_NAMES = [
    "k_E_cm", "k_cos_theta_cm", "roe_E_cm", "roe_p_cm", "roe_cos_k",
    "roe_cos_theta_cm", "roe_n_tracks", "roe_n_photons", "roe_charge",
    "roe_scalar_p_cm", "event_missing_E_cm", "event_missing_p_cm",
    "event_missing_cos_theta_cm", "event_missing_mass2", "event_R2"
]
KAON_ONLY_COLUMNS = [0, 1]
METHODS = ["q2_energy_belle", "q2_energy_mass", "q2_roe_direction", "q2_truth_direction_control"]


def mass2(p):
    p = np.asarray(p)
    return p[..., 0]**2 - np.sum(p[..., 1:]**2, axis=-1)


def boost_z(p, beta):
    p = np.asarray(p, dtype=float)
    result = p.copy()
    gamma = 1/np.sqrt(1-beta**2)
    result[..., 0] = gamma*(p[..., 0]+beta*p[..., 3])
    result[..., 3] = gamma*(p[..., 3]+beta*p[..., 0])
    return result


def event_split(generator_seed, event, split_seed):
    # Same generator seed/event group stays in the same split across response
    # scenarios and overlapping generator runs. 0=train, 1=validation, 2=test.
    key = f"{generator_seed}:{event}:{split_seed}".encode()
    u = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")/2**64
    return 0 if u < 0.6 else 1 if u < 0.8 else 2


def event_noise(generator_seed, event, response_seed, size):
    rng = np.random.default_rng(np.random.SeedSequence([response_seed, generator_seed, event]))
    return {"keep": rng.random(size), "scale": rng.normal(size=size),
            "angle": rng.normal(size=(size, 3))}


def fox_wolfram_r2(momentum):
    lengths = np.linalg.norm(momentum, axis=1)
    good = lengths > 0
    if not np.any(good):
        return 0.
    unit = momentum[good]/lengths[good, None]
    weight = np.outer(lengths[good], lengths[good])
    cosine = np.clip(unit @ unit.T, -1, 1)
    return float(np.sum(weight*(3*cosine**2-1)/2)/np.sum(weight))


def reconstruct(p4_cm, pdg, charge3, kaon_position, noise, scenario, settings, mk):
    """Simulation uses particle type; output contains only measured-object data.

    All arrays here describe terminal particles. No B direction, q2, ancestry,
    tag-side identity, or tag-side missing-energy label enters reconstruction.
    """
    lab = boost_z(p4_cm, settings["longitudinal_beta"])
    p = np.linalg.norm(lab[:, 1:], axis=1)
    track = charge3 != 0
    photon = pdg == 22
    eligible = track | photon
    good = eligible & (p > 0)
    unit = np.zeros_like(lab[:, 1:])
    unit[good] = lab[good, 1:]/p[good, None]
    theta = np.degrees(np.arccos(np.clip(unit[:, 2], -1, 1)))
    if scenario["fiducial"]:
        low, high = settings["charged_theta_deg"]
        accepted_photon = np.zeros(len(p), dtype=bool)
        for lowg, highg in settings["photon_regions_deg"]:
            accepted_photon |= (theta >= lowg) & (theta <= highg)
        good &= (track & (theta >= low) & (theta <= high)) | (photon & accepted_photon)
    scale = scenario["smear_scale"]
    sigma_track = np.hypot(settings["track_resolution_a_per_GeV"]*np.linalg.norm(lab[:, 1:3], axis=1),
                           settings["track_resolution_b"])
    safe_E = np.maximum(lab[:, 0], 1e-9)
    sigma_photon = np.sqrt((settings["photon_resolution_a_GeV"]/safe_E)**2
                           + settings["photon_resolution_b_sqrtGeV"]**2/safe_E
                           + settings["photon_resolution_c"]**2)
    sigma = scale*np.where(photon, sigma_photon, sigma_track)
    # Positive, mean-preserving momentum/energy response; width is the configured
    # fractional RMS. Direction jitter is projected into the tangent plane.
    log_sigma = np.sqrt(np.log1p(sigma**2))
    factor = np.exp(log_sigma*noise["scale"]-0.5*log_sigma**2)
    measured_p = np.where(photon, lab[:, 0], p)*factor
    tangent = noise["angle"]-np.sum(noise["angle"]*unit, axis=1)[:, None]*unit
    angle_sigma = scale*np.where(photon, settings["photon_angle_sigma_rad"], settings["track_angle_sigma_rad"])
    measured_unit = unit+angle_sigma[:, None]*tangent
    norm = np.linalg.norm(measured_unit, axis=1)
    measured_unit[good] /= norm[good, None]
    measured_lab = np.zeros_like(lab)
    measured_lab[:, 1:] = measured_p[:, None]*measured_unit
    mass = np.where(track, settings["roe_track_mass_GeV"], 0.)
    mass[kaon_position] = mk
    measured_lab[:, 0] = np.sqrt(measured_p**2+mass**2)
    if scenario["thresholds"]:
        pass_track = np.linalg.norm(measured_lab[:, 1:3], axis=1) >= settings["min_track_pt_GeV"]
        pass_photon = measured_lab[:, 0] >= settings["min_photon_E_GeV"]
        good &= (track & pass_track) | (photon & pass_photon)
    efficiency = np.where(track, scenario["track_efficiency"], scenario["photon_efficiency"])
    good &= noise["keep"] < efficiency
    retained = np.flatnonzero(good)
    observed = boost_z(measured_lab[good], -settings["longitudinal_beta"])
    return observed, retained, track[good], charge3[good]


def observable_features(observed, retained, track, charge3, kaon_position, sqrts, mb, mk):
    """Pure feature construction: receives reconstructed objects and constants."""
    where = np.flatnonzero(retained == kaon_position)
    if not len(where):
        return None
    k_index = int(where[0])
    k = observed[k_index]
    pk = float(np.linalg.norm(k[1:]))
    ku = k[1:]/pk
    roe_mask = np.arange(len(observed)) != k_index
    roe_objects = observed[roe_mask]
    roe = roe_objects.sum(axis=0)
    pr = float(np.linalg.norm(roe[1:]))
    roe_cos_k = float(np.dot(roe[1:], ku)/pr) if pr > 0 else 0.
    missing = np.array([sqrts, 0., 0., 0.])-observed.sum(axis=0)
    pm = float(np.linalg.norm(missing[1:]))
    features = np.array([
        k[0], ku[2], roe[0], pr, roe_cos_k, roe[3]/pr if pr > 0 else 0.,
        np.sum(track[roe_mask]), np.sum(~track[roe_mask]), np.sum(charge3[roe_mask])/3,
        np.sum(np.linalg.norm(roe_objects[:, 1:], axis=1)), missing[0], pm,
        missing[3]/pm if pm > 0 else 0., mass2(missing), fox_wolfram_r2(observed[:, 1:])
    ], dtype=float)
    pb2 = sqrts**2/4-mb**2
    q_belle = sqrts**2/4+mk**2-sqrts*k[0]
    q_mass = q_belle-pb2
    q_roe = q_mass-2*np.sqrt(pb2)*pk*roe_cos_k
    return {"X": features, "k_unit": ku, "k_p": pk,
            "q2_energy_belle": q_belle, "q2_energy_mass": q_mass,
            "q2_roe_direction": q_roe, "event_missing_mass2": float(mass2(missing))}


